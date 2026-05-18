"""
Pipeline: extract embeddings from a pre-trained encoder and a prepared dataset,
then train a classifier using Stratified K-Fold Cross Validation.
"""

import gc
import re
from pathlib import Path
from typing import Callable

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import global_add_pool
from torchmetrics import Accuracy, F1Score
from src.cli.embedding_pipeline import EmbeddingExtractor, EmbeddingSet
from src.data_utils.datamodule import GraphDataSet
from src.data_utils.stats import compute_macro_stats
from src.data_utils.transforms import GenNormalize
from src.models.encoder import GraphLatent
from src.models.loader_model import load_encoder_from_folder

torch.set_float32_matmul_precision("high")


def make_minnie65_class_getter(csv_path: str) -> Callable:
    df = pd.read_csv(csv_path).dropna(subset=["segment_id", "cell_type"])
    mapping = {str(int(row["segment_id"])): row["cell_type"] for _, row in df.iterrows()}

    class_map = {
        "23P": 0, "4P": 0, "5P-IT": 0, "5P-NP": 0, "5P-PT": 0,
        "6P-CT": 0, "6P-IT": 0, "BC": 1, "BPC": 1, "MC": 1, "NGC": 1,
    }

    def get_class(file_path: Path, out=None, **kwargs) -> torch.Tensor:
        segment_id = None
        if out is not None and hasattr(out, "segment_id") and isinstance(out.segment_id, str):
            match = re.search(r"\d+", out.segment_id)
            if match:
                segment_id = match.group(0)

        if segment_id is None:
            match = re.search(r"\d+", Path(file_path).name)
            if not match:
                raise ValueError(f"Could not find segment_id in filename: {file_path}")
            segment_id = match.group(0)

        cell_type = mapping.get(segment_id)
        if cell_type is None or cell_type not in class_map:
            return torch.tensor(-1, dtype=torch.long)

        return torch.tensor(class_map[cell_type], dtype=torch.long)

    return get_class


def extract_all_embeddings(cfg: DictConfig, encoder_folder: str, dataset_path: str, device: torch.device):
    cls_cfg = cfg.classifier
    dm_cfg = cfg.datamodule

    print(f"Loading encoder from: {encoder_folder}")
    encoder = load_encoder_from_folder(encoder_folder)
    encoder.eval().requires_grad_(False).to(device)
    gen_normalize = GenNormalize(transforms=[], mask_transform=None)

    get_class_fn = make_minnie65_class_getter(dm_cfg.dataset.class_path)

    print(f"Loading dataset from: {dataset_path}")
    ds = GraphDataSet(path=dataset_path, get_class=get_class_fn, transform=gen_normalize)

    macro_mean, macro_std = compute_macro_stats(ds)
    encoder_graph = GraphLatent(
        encoder=encoder,
        macro_mean=macro_mean,
        macro_std=macro_std,
        pooling=global_add_pool,
    ).to(device)


    extractor = EmbeddingExtractor(encoder=encoder_graph, device=device)
    emb_set: EmbeddingSet = extractor.extract_from_graph_dataset(
        dataset=ds,
        batch_size=cls_cfg["batch_size"],
        num_workers=dm_cfg.get("num_workers", 4),
        desc="Extracting All"
    )

    pooling_level = cls_cfg["pooling_level"]
    if pooling_level == "neuron":
        pooling_type = cls_cfg["pooling_type"]
        print(f"Pooling level: {pooling_level}, type: {pooling_type}")
        pooled_set = emb_set.pool_by_segment(pooling_type=pooling_type)
    else:
        pooled_set = emb_set


    x_pooled, y_pooled = pooled_set.embeddings, pooled_set.labels

    del encoder, encoder_graph, extractor, emb_set
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return x_pooled, y_pooled


class LinearClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.head = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class EmbeddingsLightModule(L.LightningModule):
    def __init__(self, classifier, lr, wd, max_epochs, num_classes, class_names=None):
        super().__init__()
        self.classifier = classifier
        self.lr = lr
        self.wd = wd
        self.max_epochs = max_epochs
        self.num_classes = num_classes
        self.class_names = class_names
        self.loss_fn = nn.CrossEntropyLoss()

        metric_kwargs = dict(task="multiclass", num_classes=num_classes)
        self.train_acc = Accuracy(**metric_kwargs, average=None)
        self.val_acc   = Accuracy(**metric_kwargs, average=None)
        self.test_acc  = Accuracy(**metric_kwargs, average=None)
        self.train_f1  = F1Score(**metric_kwargs, average="macro")
        self.val_f1    = F1Score(**metric_kwargs, average="macro")
        self.test_f1   = F1Score(**metric_kwargs, average="macro")

    def forward(self, x):
        return self.classifier(x)

    def _log_class_acc(self, acc_tensor, stage):
        names = self.class_names or [f"class_{i}" for i in range(len(acc_tensor))]
        for name, val in zip(names, acc_tensor):
            self.log(f"{stage}_acc_{name}", val)

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        self.train_acc(preds, y)
        self.train_f1(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_f1", self.train_f1, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        acc = self.train_acc.compute()
        self._log_class_acc(acc, "train")
        self.log("train_acc", acc.mean(), prog_bar=True)
        self.train_acc.reset()

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        preds = logits.argmax(dim=1)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.log("val_loss", self.loss_fn(logits, y), prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self._log_class_acc(acc, "val")
        self.log("val_acc", acc.mean(), prog_bar=True)
        self.val_acc.reset()

    def test_step(self, batch, _):
        x, y = batch
        preds = self(x).argmax(dim=1)
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.log("test_f1", self.test_f1)

    def on_test_epoch_end(self):
        acc = self.test_acc.compute()
        self._log_class_acc(acc, "test")
        self.log("test_acc", acc.mean(), prog_bar=True)
        self.test_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def train_cv(cfg: DictConfig, x_all: torch.Tensor, y_all: torch.Tensor):
    cls_cfg = cfg.classifier
    num_classes = cls_cfg.get("num_classes", 2)
    n_splits    = cfg.get("n_splits", 5)
    batch_size  = cfg.datamodule.batch_size
    max_epochs  = cls_cfg.get("max_epochs", 500)

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
            enable_progress_bar=False,
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


@hydra.main(version_base="1.3", config_path="../../../../configs", config_name="config")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_path = "/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/lightning_logs/jepa_r_1.5_sh_0/version_1"
    dataset_path = "/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/datasets/dataset_sph_minnie65_r=1.5"
    n_splits     = cfg.get("n_splits", 5)

    print("=" * 60)
    print(f" Cross-Validation Evaluation Pipeline")
    print(f" Encoder : {encoder_path}")
    print(f" Dataset : {dataset_path}")
    print(f" Folds   : {n_splits}")
    print("=" * 60)

    print("\n[1/2] Extracting Embeddings...")
    x_pooled, y_pooled = extract_all_embeddings(cfg, encoder_path, dataset_path, device)
    print(f"Features: {x_pooled.shape}, Labels: {y_pooled.shape}")

    print(f"\n[2/2] Training & Evaluating with {n_splits}-Fold CV...")
    fold_metrics = train_cv(cfg, x_pooled, y_pooled)

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