"""
Pipeline: extract embeddings from a specific pre-trained encoder and a prepared dataset,
then train a classifier using Stratified K-Fold Cross Validation.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as L
import torch
import gc
from pathlib import Path
from torch_geometric.nn import global_add_pool
from torch.utils.data import TensorDataset, DataLoader

from src.models.loader_model import load_encoder_from_folder
from src.models.encoder import GraphLatent
from src.data_utils.datamodule import GraphDataSet, make_minnie65_class_getter
from src.data_utils.transforms import GenNormalize
from src.data_utils.stats import compute_macro_stats
from src.cli.train_model import load_stats, build_transforms
from src.cli.extract_embeddings import extract_from_dataset
from src.cli.train_from_embeddings import pool_by_segment

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

torch.set_float32_matmul_precision('high')

# ──────────────────────────────────────────────────────────
#  Step 1: Extract embeddings
# ──────────────────────────────────────────────────────────
def extract_all_embeddings(cfg: DictConfig, encoder_folder: str, dataset_path: str, device):
    """Loads encoder, builds dataset, extracts all embeddings (no splitting)."""
    cls_cfg = cfg.classifier
    dm_cfg = cfg.datamodule

    print(f"Loading encoder from: {encoder_folder}")
    encoder = load_encoder_from_folder(encoder_folder)
    encoder.eval()
    encoder.requires_grad_(False)
    encoder.to(device)

    # Note: Ensure the stats path is correct in your config
    mean_x, std_x, mean_edge, std_edge = load_stats(cls_cfg.stats_path)
    transforms = build_transforms(dm_cfg, mean_x, std_x, mean_edge, std_edge)
    gen_normalize = GenNormalize(transforms=transforms, mask_transform=None)

    csv_path = dm_cfg.dataset.class_path
    get_class_fn = make_minnie65_class_getter(csv_path)

    print(f"Loading dataset from: {dataset_path}")
    ds = GraphDataSet(path=dataset_path, get_class=get_class_fn, transform=gen_normalize)

    macro_mean, macro_std = compute_macro_stats(ds)

    encoder_graph = GraphLatent(
        encoder=encoder,
        macro_mean=macro_mean,
        macro_std=macro_std,
        pooling=global_add_pool,
        sigma=cls_cfg.get("sigma", 1.0),
    ).to(device)

    # Extract embeddings for the entire dataset
    emb_all, y_all, seg_all = extract_from_dataset(ds, encoder_graph, device, "All")

    # Perform segment pooling if necessary
    pooling_level = cls_cfg.get("pooling_level", "graph")
    if pooling_level == "neuron":
        pooling_type = cls_cfg.get("pooling_type", "mean")
        print(f"Pooling level: {pooling_level}, type: {pooling_type}")
        x_pooled, y_pooled = pool_by_segment(emb_all, y_all, seg_all, pooling_type)
    else:
        x_pooled, y_pooled = emb_all, y_all

    # Cleanup encoder from GPU
    del encoder, encoder_graph
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return x_pooled, y_pooled


# ──────────────────────────────────────────────────────────
#  Step 2: Lightning Module for Training
# ──────────────────────────────────────────────────────────
class EmbeddingsLightModule(L.LightningModule):
    """Lightweight classifier on cached embeddings."""
    def __init__(self, classifier, lr, wd, max_epochs, num_classes, class_names=None):
        super().__init__()
        self.classifier = classifier
        self.lr = lr
        self.wd = wd
        self.max_epochs = max_epochs
        self.num_classes = num_classes
        self.class_names = class_names
        self.loss_fn = torch.nn.CrossEntropyLoss()

        from torchmetrics import Accuracy, F1Score
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes, average=None)
        self.val_acc   = Accuracy(task="multiclass", num_classes=num_classes, average=None)
        self.test_acc  = Accuracy(task="multiclass", num_classes=num_classes, average=None)

        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1   = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1  = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x):
        return self.classifier(x)

    def _log_class_acc(self, acc_tensor, stage):
        if self.class_names:
            for i, class_name in enumerate(self.class_names):
                if i < len(acc_tensor):
                    self.log(f"{stage}_acc_{class_name}", acc_tensor[i])
        else:
            for i, val in enumerate(acc_tensor):
                self.log(f"{stage}_acc_class_{i}", val)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y); self.train_f1(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_f1", self.train_f1, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        acc = self.train_acc.compute()
        self._log_class_acc(acc, "train")
        self.log("train_acc", acc.mean(), prog_bar=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y); self.val_f1(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self._log_class_acc(acc, "val")
        self.log("val_acc", acc.mean(), prog_bar=True)
        self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y); self.test_f1(preds, y)
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


# ──────────────────────────────────────────────────────────
#  Step 3: Cross-Validation Loop
# ──────────────────────────────────────────────────────────
from torch import nn
class LinearClassifier(nn.Module):
    """Simple linear probe on top of frozen graph embeddings."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            # nn.LayerNorm(in_channels),
            # nn.Dropout(0.3),
            nn.Linear(in_channels, num_classes),

        )
    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        return self.head(embed)


def train_cv(cfg: DictConfig, x_all: torch.Tensor, y_all: torch.Tensor):

    cls_cfg = cfg.classifier
    num_classes = cls_cfg.get("num_classes", 2)
    n_splits = cfg.get("n_splits", 5)
    batch_size = cfg.datamodule.batch_size
    max_epochs = cls_cfg.get("max_epochs", 500)

    # Use StratifiedKFold for cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
    
    # Move to CPU for splitting if they are on GPU
    x_np = x_all.cpu().numpy()
    y_np = y_all.cpu().numpy()

    fold_metrics = []

    print(f"\nStarting {n_splits}-Fold Cross Validation...")

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(x_np, y_np)):
        print(f"\n{'='*40}")
        print(f" Fold {fold + 1}/{n_splits}")
        print(f"{'='*40}")

        # Split train_val into train and val (e.g., 10% for validation for early stopping)
        train_idx, val_idx = train_test_split(
            train_val_idx, 
            test_size=0.1, 
            random_state=cfg.seed, 
            stratify=y_np[train_val_idx]
        )

        x_train, y_train = x_all[train_idx], y_all[train_idx]
        x_val,   y_val   = x_all[val_idx],   y_all[val_idx]
        x_test,  y_test  = x_all[test_idx],  y_all[test_idx]

        train_ds = TensorDataset(x_train, y_train)
        val_ds   = TensorDataset(x_val, y_val)
        test_ds  = TensorDataset(x_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, persistent_workers=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True)

        in_channels = x_all.shape[1]
        classifier_head = LinearClassifier(in_channels=in_channels, num_classes=num_classes)

        module = EmbeddingsLightModule(
            classifier_head,
            lr=cls_cfg.get("learning_rate", 1e-3),
            wd=cls_cfg.get("weight_decay", 1e-5),
            max_epochs=max_epochs,
            num_classes=num_classes,
        )

        checkpoint_callback = L.callbacks.ModelCheckpoint(
            monitor="val_acc", mode="max", save_top_k=1,
            filename=f"cv_fold_{fold}-{{epoch:02d}}-{{val_acc:.4f}}",
        )

        trainer = L.Trainer(
            max_epochs=max_epochs,
            accelerator=cfg.trainer.get("accelerator", "gpu"),
            devices=cfg.trainer.get("devices", 1),
            logger = L.loggers.TensorBoardLogger(
                save_dir=cfg.get("log_dir", "lightning_logs"),
                name=f"cv_classifier",
                version=f"fold_{fold}",
            ),
            callbacks=[checkpoint_callback],
            deterministic=True,
            enable_progress_bar=False, # Disable progress bar to reduce clutter
        )

        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        print(f"Testing Fold {fold + 1} best model...")
        results = trainer.test(module, dataloaders=test_loader, verbose=False)
        
        if results:
            res = results[0]
            fold_acc = res.get("test_acc", 0.0)
            fold_f1 = res.get("test_f1", 0.0)
            fold_metrics.append((fold_acc, fold_f1))
            print(f"Fold {fold + 1} Results -> Accuracy: {fold_acc:.4f}, F1: {fold_f1:.4f}")

        # Cleanup
        del module, trainer, checkpoint_callback
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return fold_metrics

# ──────────────────────────────────────────────────────────
#  Main Pipeline
# ──────────────────────────────────────────────────────────
@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read specific inputs from config or command line
    encoder_path = "/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/lightning_logs/jepa_r_1.5_sh_0/version_1"
    dataset_path = "/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/datasets/dataset_prepared"
    n_splits = cfg.get("n_splits", 5)

    if not encoder_path or not dataset_path:
        print("Error: You must provide both 'encoder_path' and 'dataset_path'.")
        print("Usage: python src/cli/evaluate_encoder_cv.py encoder_path=/path/to/encoder dataset_path=/path/to/dataset")
        return

    print("=" * 60)
    print(f" Cross-Validation Evaluation Pipeline ")
    print(f" Encoder: {encoder_path}")
    print(f" Dataset: {dataset_path}")
    print(f" Folds:   {n_splits}")
    print("=" * 60)

    # 1. Extract Embeddings
    print("\n[1/2] Extracting Embeddings...")
    x_pooled, y_pooled = extract_all_embeddings(cfg, encoder_path, dataset_path, device)
    print(f"Extracted features shape: {x_pooled.shape}")
    print(f"Labels shape: {y_pooled.shape}")

    # 2. Cross-Validation
    print(f"\n[2/2] Training & Evaluating with {n_splits}-Fold CV...")
    fold_metrics = train_cv(cfg, x_pooled, y_pooled)

    if not fold_metrics:
        print("No metrics collected!")
        return

    # 3. Aggregate Metrics
    acc_scores = [m[0] for m in fold_metrics]
    f1_scores = [m[1] for m in fold_metrics]

    mean_acc, std_acc = np.mean(acc_scores), np.std(acc_scores)
    mean_f1, std_f1 = np.mean(f1_scores), np.std(f1_scores)

    print("\n" + "=" * 60)
    print("  CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    for i, (acc, f1) in enumerate(fold_metrics):
        print(f"  Fold {i+1:>2}:  Accuracy = {acc:.4f},  F1 Score = {f1:.4f}")
    print("-" * 60)
    print(f"  MEAN    :  Accuracy = {mean_acc:.4f} ± {std_acc:.4f},  F1 Score = {mean_f1:.4f} ± {std_f1:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
