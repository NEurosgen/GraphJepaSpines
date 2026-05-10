import torch
from torch import nn
import pytorch_lightning as L
import numpy as np
import gc
import os
import glob
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from torch_geometric.nn import global_add_pool
from torch_geometric.data import Batch
from sklearn.metrics import f1_score as sklearn_f1_score

from src.models.jepa import JepaLight
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
    GaussianNoiseAugmentation,
)
from src.data_utils.structural_stats import ThesisMacroMetrics
from src.data_utils.stats import compute_macro_stats

torch.set_float32_matmul_precision('high')


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


def build_transforms(cfg, mean_x, std_x, mean_edge, std_edge, noise_sigma=0.0):
    """
    Build the transform pipeline for the 9009 classification dataset.
    Optionally adds Gaussian noise N(0, noise_sigma) to node features.
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
    transforms.append(ThesisMacroMetrics())
    transforms.append(ConcatStructuralPE())

    if noise_sigma > 0:
        transforms.append(GaussianNoiseAugmentation(sigma=noise_sigma))

    return transforms


def load_encoder_from_folder(folder_path, network_cfg=None):
    """
    Load a trained encoder from a checkpoint folder.
    
    Supports two layouts:
      1. Standard Lightning: folder/checkpoints/*.ckpt + folder/hparams.yaml
      2. Flat (crash test):  folder/*.ckpt  (no hparams.yaml)
    
    When hparams.yaml is missing, network_cfg (from main config) is used
    to instantiate the model architecture.
    """
    # Try standard layout first, then flat
    checkpoint_dir = os.path.join(folder_path, "checkpoints")
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))

    if not checkpoint_files:
        # Flat layout: .ckpt files directly in folder
        checkpoint_files = glob.glob(os.path.join(folder_path, "*.ckpt"))

    if not checkpoint_files:
        raise FileNotFoundError(
            f"No checkpoints found in {folder_path} or {checkpoint_dir}"
        )

    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    print(f"  Loading checkpoint: {latest_checkpoint}")

    # Try to load hparams.yaml for model config
    hparams_path = os.path.join(folder_path, "hparams.yaml")
    if os.path.exists(hparams_path):
        hparams = OmegaConf.load(hparams_path)
        if "cfg" in hparams and "network" in hparams.cfg:
            model = instantiate(hparams.cfg.network, _recursive_=True)
        elif "network" in hparams:
            model = instantiate(hparams.network, _recursive_=True)
        else:
            raise ValueError(f"Could not find 'network' config in {hparams_path}")
    elif network_cfg is not None:
        # Fallback: use the network config passed from main config
        print(f"  No hparams.yaml found, using network config from main config")
        model = instantiate(network_cfg, _recursive_=True)
    else:
        raise FileNotFoundError(
            f"No hparams.yaml in {folder_path} and no network_cfg provided"
        )

    jepa_light = JepaLight.load_from_checkpoint(
        checkpoint_path=latest_checkpoint,
        model=model,
        strict=False,
        weights_only=False
    )

    jepa_model = jepa_light.model
    if hasattr(jepa_model, 'student_encoder'):
        return jepa_model.student_encoder
    elif hasattr(jepa_model, 'encoder'):
        return jepa_model.encoder
    else:
        return jepa_model


def run_classification(cfg, encoder_path, noise_sigma=0.0, run_name="classifier"):
    """
    Run binary classification (ab vs wt) on the 9009 dataset
    using a frozen encoder loaded from encoder_path.

    Args:
        cfg: Full Hydra DictConfig.
        encoder_path: Path to Lightning version folder with checkpoints/.
        noise_sigma: If > 0, adds Gaussian noise N(0, sigma) to node features.
        run_name: Name for TensorBoard logger / checkpoint folder.
    """
    cls_cfg = cfg.classifier
    dm_cfg = cfg.datamodule

    # Load frozen encoder
    print(f"Loading encoder from: {encoder_path}")
    encoder = load_encoder_from_folder(encoder_path, network_cfg=cfg.network)
    encoder.eval()
    encoder.requires_grad_(False)

    num_classes = cls_cfg.get("num_classes", 2)

    # Build transforms with optional noise
    mean_x, std_x, mean_edge, std_edge = load_stats(cls_cfg.stats_path)
    transforms = build_transforms(dm_cfg, mean_x, std_x, mean_edge, std_edge, noise_sigma=noise_sigma)
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
    print(f"Embedding dimension: {embed_dim} (Encoder: {cfg.network.encoder.out_channels}, Macro: {num_macro})")

    # Classifier
    classifier_head = LinearClassifier(in_channels=embed_dim, num_classes=num_classes)
    encoder_graph = GraphLatent(
        encoder=encoder,
        macro_mean=macro_mean,
        macro_std=macro_std,
        pooling=global_add_pool,
        sigma=cls_cfg.get("sigma", 1.0)
    )

    module = ClassifierLightModule(
        cfg=cls_cfg,
        encoder_graph=encoder_graph,
        learning_rate=cls_cfg.get("learning_rate", 1e-3),
        classifier=classifier_head
    )

    # DataModule
    datamodule = GraphDataModule(
        ds,
        batch_size=dm_cfg.batch_size,
        num_workers=dm_cfg.num_workers,
        seed=dm_cfg.seed,
        ratio=dm_cfg.ratio,
        collate_fn=_simple_collate
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
        save_dir=cfg.get("log_dir", "lightning_logs"),
        name=f"classifier_{run_name}"
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

@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)

    cls_cfg = cfg.classifier

    # Single encoder evaluation from classifier.checkpoint_path
    encoder_path = cls_cfg.checkpoint_path
    noise_sigma = cls_cfg.get("noise_sigma", 0.0)

    print("=" * 60)
    print("  9009 Binary Classification (ab vs wt)")
    print(f"  Encoder: {encoder_path}")
    print(f"  Noise sigma: {noise_sigma}")
    print("=" * 60)

    run_classification(
        cfg=cfg,
        encoder_path=encoder_path,
        noise_sigma=noise_sigma,
        run_name=f"9009_sigma_{noise_sigma}"
    )


if __name__ == "__main__":
    main()
