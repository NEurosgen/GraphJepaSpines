"""
Robustness test: load ONE encoder from config, then classify 9009 data
with increasing levels of Gaussian position noise.

This answers the question: "How robust is this encoder to spine position
perturbation at inference time?"

Usage:
    python -m src.crash_tests.experiments_shufle.evaluate_noise_robustness

Reads encoder from:  classifier.checkpoint_path  (config.yaml)
Noise sigmas:        defined in SIGMA_VALUES below (can be overridden via CLI)
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

# Standard loader (reads hparams.yaml automatically)
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

# Noise levels to sweep
SIGMA_VALUES = [0, 10, 100, 1000, 10000]


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
    Optionally adds Gaussian noise N(0, pos_noise_sigma) to node POSITIONS
    AFTER LocalPos normalization.
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


# ─── Single classification run ───────────────────────────────────────────

def run_classification(cfg, encoder_path, pos_noise_sigma=0.0, run_name="classifier"):
    """
    Run binary classification (ab vs wt) on the 9009 dataset using a
    frozen encoder loaded from encoder_path with given position noise.

    Returns dict with test metrics.
    """
    cls_cfg = cfg.classifier
    dm_cfg = cfg.datamodule

    # Load frozen encoder via the STANDARD loader
    print(f"  Loading encoder from: {encoder_path}")
    encoder = load_encoder_from_folder(encoder_path)
    encoder.eval()
    encoder.requires_grad_(False)

    num_classes = cls_cfg.get("num_classes", 2)

    # Build transforms with position noise
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

    print("  Computing dynamic macro statistics for dataset...")
    macro_mean, macro_std = compute_macro_stats(ds)

    # Determine embedding dimension
    num_macro = macro_mean.shape[1] if macro_mean is not None else 0
    embed_dim = cfg.network.encoder.out_channels + num_macro
    print(f"  Embedding dim: {embed_dim} "
          f"(Encoder: {cfg.network.encoder.out_channels}, Macro: {num_macro})")

    # Classifier head + frozen encoder wrapper
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
        name=f"robustness_{run_name}",
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
    print(f"  Training classifier: {run_name}")
    trainer.fit(module, datamodule=datamodule)

    # Test
    print(f"  Evaluating on test set: {run_name}")
    results = trainer.test(module, datamodule=datamodule)
    test_metrics = results[0] if results else {}

    # Cleanup
    del module, trainer, datamodule, encoder, encoder_graph, classifier_head
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return test_metrics


# ─── Main ─────────────────────────────────────────────────────────────────

@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)

    cls_cfg = cfg.classifier
    encoder_path = cls_cfg.checkpoint_path

    # Allow overriding sigma list from config
    sigma_values = list(cls_cfg.get("sigma_values", SIGMA_VALUES))

    print("=" * 60)
    print("  Robustness Test: Position Noise Sweep")
    print(f"  Encoder:  {encoder_path}")
    print(f"  Sigmas:   {sigma_values}")
    print("=" * 60)
    print()

    summary = []

    for sigma in sigma_values:
        print("=" * 60)
        print(f"  pos_noise_sigma = {sigma}")
        print("=" * 60)

        metrics = run_classification(
            cfg=cfg,
            encoder_path=encoder_path,
            pos_noise_sigma=sigma,
            run_name=f"noise_{sigma}",
        )

        test_acc = metrics.get("test_acc", float('nan'))
        test_f1 = metrics.get("test_f1", float('nan'))
        test_loss = metrics.get("test_loss", float('nan'))

        summary.append((sigma, test_acc, test_f1, test_loss))

        print(f"\n  ✓ σ={sigma}  acc={test_acc:.4f}  f1={test_f1:.4f}  loss={test_loss:.4f}")
        print("-" * 60 + "\n")

    # ── Summary table ──
    print("\n" + "=" * 60)
    print("  ROBUSTNESS SUMMARY")
    print(f"  Encoder: {encoder_path}")
    print("=" * 60)
    print(f"  {'sigma':>10}  {'test_acc':>10}  {'test_f1':>10}  {'test_loss':>10}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")
    for sigma, acc, f1, loss in summary:
        print(f"  {sigma:>10}  {acc:>10.4f}  {f1:>10.4f}  {loss:>10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
