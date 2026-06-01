"""
Pipeline: extract embeddings from a pre-trained encoder and a prepared dataset,
then train a classifier using Stratified K-Fold Cross Validation.
"""

import gc
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as L
import torch
from omegaconf import DictConfig
from torch import nn
from torch_geometric.nn import global_add_pool
from src.cli.embedding_pipeline import EmbeddingExtractor, EmbeddingSet, EmbeddingsLightModule
from src.data_utils.datamodule import GraphDataSet
from src.data_utils.transforms import GenNormalize
from src.models.encoder import GraphLatent
from src.models.loader_model import load_encoder_from_folder
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset
torch.set_float32_matmul_precision("high")

class LinearClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(in_channels, in_channels),
                                  nn.ReLU(),
                                  nn.Linear(in_features=in_channels,out_features=num_classes)
                                  )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

def train_cv(cfg: DictConfig, x_all: torch.Tensor, y_all: torch.Tensor , class_names = None):
    cls_cfg = cfg.classifier
    num_classes = cls_cfg.get("num_classes", 2)
    n_splits    = cls_cfg["n_splits"]
    batch_size  = cls_cfg.batch_size
    max_epochs  = cls_cfg["max_epochs"]

    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
    x_np = x_all.cpu().numpy()
    y_np = y_all.cpu().numpy()

    fold_metrics = []
    print(f"\nStarting {n_splits}-Fold Cross Validation...")

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(x_np, y_np)):
        print(f"\n{'='*40}\n Fold {fold + 1}/{n_splits}\n{'='*40}")

        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.1, random_state=cfg.seed, stratify=y_np[train_val_idx]
        )

        def make_loader(idx, shuffle):
            ds = TensorDataset(x_all[idx], y_all[idx])
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, persistent_workers=True)

        train_loader = make_loader(train_idx, shuffle=True)
        val_loader   = make_loader(val_idx,   shuffle=False)
        test_loader  = make_loader(test_idx,  shuffle=False)

        module = EmbeddingsLightModule(
            classifier=LinearClassifier(in_channels=x_all.shape[1], num_classes=num_classes),
            lr=cls_cfg["learning_rate"],
            wd=cls_cfg["weight_decay"],
            max_epochs=max_epochs,
            num_classes=num_classes,
            class_names=class_names,
        )

        checkpoint_cb = L.callbacks.ModelCheckpoint(
            monitor="val_acc", mode="max", save_top_k=1,
            filename=f"cv_fold_{fold}-{{epoch:02d}}-{{val_acc:.4f}}",
        )

        trainer = L.Trainer(
            max_epochs=max_epochs,
            accelerator=cfg.trainer.get("accelerator", "gpu"),
            devices=cfg.trainer.get("devices", 1),
            logger=L.loggers.TensorBoardLogger(
                save_dir=cfg["log_dir"],
                name="cv_classifier",
                version=f"fold_{fold}",
            ),
            callbacks=[checkpoint_cb],
            deterministic=True,
            enable_progress_bar=True,
        )

        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        print(f"Testing Fold {fold + 1} best model...")
        results = trainer.test(module, dataloaders=test_loader, verbose=False)
        if results:
            fold_acc = results[0].get("test_acc", 0.0)
            fold_f1  = results[0].get("test_f1",  0.0)
            fold_metrics.append((fold_acc, fold_f1))
            print(f"Fold {fold + 1} -> Accuracy: {fold_acc:.4f}, F1: {fold_f1:.4f}")

        del module, trainer, checkpoint_cb
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return fold_metrics

def get_class_humans(file_path) -> torch.Tensor:
    mapping = {"Human_age40_apical_OFF": 0, "Human_age40_basal_OFF": 0, 
               "Human_age85_apical_OFF": 1, "Human_age_85_basal_OFF" : 1 }
    folder_name = Path(file_path).parent.parent.name
    if folder_name not in mapping:
        raise ValueError(
            f"Folder '{folder_name}' not in mapping {mapping}. File: {file_path}"
        )
    return torch.tensor(mapping[folder_name], dtype=torch.long)

def extract_all_embeddings(cfg: DictConfig, device: torch.device):
    cls_cfg = cfg.classifier
    dm_cfg = cfg.datamodule

    print(f"Loading encoder from: {cls_cfg.checkpoint_path}")
    encoder = load_encoder_from_folder(cls_cfg.checkpoint_path)
    encoder.eval().requires_grad_(False).to(device)

    transforms = []
    gen_normalize = GenNormalize(transforms=transforms, mask_transform=None)

    print(f"Loading dataset from: {cls_cfg.path}")
    ds = GraphDataSet(path=cls_cfg.path, get_class=get_class_humans, transform=gen_normalize)

    encoder_graph = GraphLatent(
        encoder=encoder,
        macro_mean=None,
        macro_std=None,
        pooling=global_add_pool,
    ).to(device)

    extractor = EmbeddingExtractor(encoder=encoder_graph, device=device)
    emb_set: EmbeddingSet = extractor.extract_from_graph_dataset(
        dataset=ds,
        batch_size=cls_cfg.get("batch_size", 128),
        num_workers=dm_cfg.get("num_workers", 4),
        desc="Extracting All",
    )

    x_all, y_all = emb_set.embeddings, emb_set.labels

    del encoder, encoder_graph, extractor, emb_set
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return x_all, y_all





@hydra.main(version_base="1.3", config_path="../../../../configs", config_name="config")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_splits = cfg.get("n_splits", 5)

    print("=" * 60)
    print(" Cross-Validation Evaluation Pipeline human spines")
    print(f" Encoder : {cfg.classifier.checkpoint_path}")
    print(f" Dataset : {cfg.classifier.path}")
    print(f" Folds   : {n_splits}")
    print("=" * 60)

    print("\n[1/2] Extracting Embeddings...")
    x_all, y_all = extract_all_embeddings(cfg, device)
    print(f"Features: {x_all.shape}, Labels: {y_all.shape}")

    print(f"\n[2/2] Training & Evaluating with {n_splits}-Fold CV...")
    fold_metrics = train_cv(cfg, x_all, y_all , class_names=["ab", "wt"])

    if not fold_metrics:
        print("No metrics collected!")
        return

    acc_scores = [m[0] for m in fold_metrics]
    f1_scores  = [m[1] for m in fold_metrics]

    print("\n" + "=" * 60)
    print("  CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    for i, (acc, f1) in enumerate(fold_metrics):
        print(f"  Fold {i+1:>2}:  Accuracy = {acc:.4f},  F1 = {f1:.4f}")
    print("-" * 60)
    print(f"  MEAN  :  Accuracy = {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f},  F1 = {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
