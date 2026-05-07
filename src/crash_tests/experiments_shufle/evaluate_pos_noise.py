"""
Evaluate binary classification (ab vs wt) on the 9009 dataset
using encoders trained with position noise (from experiments_shufle/).

Uses the STANDARD loader: src.models.loader_model.load_encoder_from_folder
which reads hparams.yaml automatically.

Optionally adds GaussianPositionNoise(sigma) to node positions at
inference time so we can measure classifier robustness.
"""

import torch
from torch import nn
import pytorch_lightning as L
import numpy as np
import gc
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from torch_geometric.nn import global_add_pool
from torch_geometric.data import Batch
from sklearn.metrics import f1_score as sklearn_f1_score

# ─── Standard loader (reads hparams.yaml, no manual network_cfg needed) ───
from src.models.loader_model import load_encoder_from_folder

from src.models.classificator import ClassifierLightModule, LinearClassifier
from src.models.encoder import GraphLatent
from src.data_utils.datamodule import GraphDataModule, GraphDataSet
from src.data_utils.transforms import (
    GenNormalize,
    NormNoEps,
    EdgeNorm,
    LocalPos,
    FeatureChoice,
    ConcatStructuralPE,
    GaussianPositionNoise,
)
from src.data_utils.structural_stats import ThesisMacroMetrics
from src.data_utils.stats import compute_macro_stats

torch.set_float32_matmul_precision('high')

SAVE_DIR = Path("src/crash_tests/experiments_shufle")


# ─── Utilities ────────────────────────────────────────────────────────────

def get_class_9009(file_path, **kwargs):
    """Binary label from folder name: ab=0, wt=1."""
    mapping = {"ab": 0, "wt": 1}
    folder_name = Path(file_path).parent.name.lower()
    if folder_name not in mapping:
        raise ValueError(
            f"Folder '{folder_name}' not in mapping {mapping}. "
            f"File: {file_path}"
        )
    return torch.tensor(mapping[folder_name], dtype=torch.long)


def _simple_collate(data_list):
    return Batch.from_data_list(data_list)


def load_stats(path):
    """Load precomputed normalization statistics."""
    mean_x = torch.load(path + "means.pt", map_location='cpu')
    std_x = torch.load(path + "stds.pt", map_location='cpu')
    mean_edge = torch.load(path + "mean_edge.pt", map_location='cpu')
    std_edge = torch.load(path + "std_edge.pt", map_location='cpu')
    return mean_x, std_x, mean_edge, std_edge


def build_transforms(cfg, mean_x, std_x, mean_edge, std_edge, pos_noise_sigma=0.0):
    """
    Build the transform pipeline for the 9009 classification dataset.
    Optionally adds Gaussian noise N(0, pos_noise_sigma) to node POSITIONS.
    """
    transforms = []

    features = cfg.get('features', None)
    if features is not None:
        features = list(features)
        transforms.append(FeatureChoice(feature=features))
        mean_x = mean_x[features]
        std_x = std_x[features]

    transforms.append(NormNoEps(mean=mean_x, std=std_x, eps=cfg.get('eps', 1e-6)))
    transforms.append(EdgeNorm(mean=mean_edge, std=std_edge))
    transforms.append(LocalPos())

    # Position noise AFTER LocalPos normalization
    if pos_noise_sigma > 0:
        transforms.append(GaussianPositionNoise(sigma=pos_noise_sigma))

    transforms.append(ThesisMacroMetrics())
    transforms.append(ConcatStructuralPE())

    return transforms


# ─── Classification runner ────────────────────────────────────────────────

def run_classification(cfg, encoder_path, pos_noise_sigma=0.0, run_name="classifier"):
    """
    Run binary classification (ab vs wt) on the 9009 dataset
    using a frozen encoder loaded from encoder_path.

    Args:
        cfg: Full Hydra DictConfig.
        encoder_path: Path to Lightning version folder with checkpoints/ and hparams.yaml.
        pos_noise_sigma: If > 0, adds Gaussian noise N(0, sigma) to node POSITIONS.
        run_name: Name for TensorBoard logger / checkpoint folder.
    """
    cls_cfg = cfg.classifier
    dm_cfg = cfg.datamodule

    # Load frozen encoder via the STANDARD loader (uses hparams.yaml)
    print(f"Loading encoder from: {encoder_path}")
    encoder = load_encoder_from_folder(encoder_path)
    encoder.eval()
    encoder.requires_grad_(False)

    num_classes = cls_cfg.get("num_classes", 2)

    # Build transforms with optional position noise
    mean_x, std_x, mean_edge, std_edge = load_stats(cls_cfg.stats_path)
    transforms = build_transforms(
        dm_cfg, mean_x, std_x, mean_edge, std_edge,
        pos_noise_sigma=pos_noise_sigma,
    )
    gen_normalize = GenNormalize(transforms=transforms, mask_transform=None)

    # Dataset
    ds = GraphDataSet(
        path=cls_cfg.path,
        get_class=get_class_9009,
        transform=gen_normalize,
    )

    print("Computing dynamic macro statistics for dataset...")
    macro_mean, macro_std = compute_macro_stats(ds)

    # Determine embedding dimension
    num_macro = macro_mean.shape[1] if macro_mean is not None else 0
    embed_dim = cfg.network.encoder.out_channels + num_macro
    print(f"Embedding dimension: {embed_dim} "
          f"(Encoder: {cfg.network.encoder.out_channels}, Macro: {num_macro})")

    # Classifier
    classifier_head = LinearClassifier(in_channels=embed_dim, num_classes=num_classes)
    encoder_graph = GraphLatent(
        encoder=encoder,
        macro_mean=macro_mean,
        macro_std=macro_std,
        pooling=global_add_pool,
        sigma=cls_cfg.get("sigma", 1.0),
    )

    module = ClassifierLightModule(
        cfg=cls_cfg,
        encoder_graph=encoder_graph,
        learning_rate=cls_cfg.get("learning_rate", 1e-3),
        classifier=classifier_head,
    )

    # DataModule
    datamodule = GraphDataModule(
        ds,
        batch_size=dm_cfg.batch_size,
        num_workers=dm_cfg.num_workers,
        seed=dm_cfg.seed,
        ratio=dm_cfg.ratio,
        collate_fn=_simple_collate,
    )

    # Trainer
    max_epochs = cls_cfg.get("max_epochs", 50)

    checkpoint_callback = L.callbacks.ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename=f"classifier-{run_name}-{{epoch:02d}}-{{val_acc:.4f}}",
    )

    logger = L.loggers.TensorBoardLogger(
        save_dir=str(SAVE_DIR),
        name=f"classifier_{run_name}",
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=cfg.trainer.get("accelerator", "gpu"),
        devices=cfg.trainer.get("devices", 1),
        log_every_n_steps=cfg.trainer.get("log_every_n_steps", 10),
        logger=logger,
        callbacks=[checkpoint_callback],
        deterministic=True,
    )

    # Train
    print(f"\nStarting classifier training: {run_name}")
    trainer.fit(module, datamodule=datamodule)

    # Test
    print(f"\nRunning evaluation on test set: {run_name}")
    trainer.test(module, datamodule=datamodule)

    # Cleanup
    del module, trainer, datamodule, encoder, encoder_graph, classifier_head
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ─── Main ─────────────────────────────────────────────────────────────────

@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)

    cls_cfg = cfg.classifier

    # ── Discover trained encoders in SAVE_DIR ──
    # Folders created by train_pos_noise.py: jepa_pos_sigma_{sigma}/version_X/
    encoder_folders = []
    for child in sorted(SAVE_DIR.iterdir()):
        if not child.is_dir() or not child.name.startswith("jepa_pos_sigma_"):
            continue
        # Extract sigma from folder name
        sigma_str = child.name.replace("jepa_pos_sigma_", "")
        try:
            sigma_val = float(sigma_str)
        except ValueError:
            continue
        # Pick latest version
        versions = sorted(
            [v for v in child.iterdir() if v.is_dir() and v.name.startswith("version_")],
            key=lambda p: int(p.name.split("_")[1]),
        )
        if not versions:
            print(f"⚠ Skipping {child.name}: no version_* subfolders")
            continue
        latest = versions[-1]
        encoder_folders.append((sigma_val, str(latest)))

    if not encoder_folders:
        # Fallback: use single encoder from config
        encoder_path = cls_cfg.checkpoint_path
        pos_noise_sigma = cls_cfg.get("noise_sigma", 0.0)
        print("No jepa_pos_sigma_* folders found. Using classifier.checkpoint_path from config.")
        encoder_folders = [(pos_noise_sigma, encoder_path)]

    print(f"Found {len(encoder_folders)} encoder(s):")
    for sigma_val, folder in encoder_folders:
        print(f"  pos_sigma={sigma_val:>8}  →  {folder}")
    print()

    # ── Run classification for each encoder ──
    for sigma_val, encoder_folder in encoder_folders:
        print("=" * 60)
        print(f"  Position Noise σ = {sigma_val}")
        print(f"  Encoder: {encoder_folder}")
        print("=" * 60)

        # Apply the same noise at inference that the encoder was trained with
        run_classification(
            cfg=cfg,
            encoder_path=encoder_folder,
            pos_noise_sigma=sigma_val,
            run_name=f"pos_sigma_{sigma_val}",
        )

        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
