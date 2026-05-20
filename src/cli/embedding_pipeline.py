import os
from dataclasses import dataclass
from typing import Literal, Optional
import pytorch_lightning as L
import torch
from torch import nn
from torch_geometric.data import Batch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from src.models.loader_model import load_encoder_from_folder
from src.models.encoder import GraphLatent
from torch_geometric.nn import global_add_pool
from src.data_utils.stats import compute_macro_stats
from src.data_utils.transforms import GenNormalize
from src.data_utils.datamodule import GraphDataSet
from src.cli.inference.minnie65.minnie65_get_class import make_minnie65_class_getter
from torchmetrics import Accuracy, F1Score
import gc
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import DictConfig

class LinearClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.head = nn.Linear(in_channels, num_classes)

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

@dataclass
class EmbeddingSet:
    """Контейнер для хранения эмбеддингов и сопутствующих метаданных."""
    embeddings: torch.Tensor
    labels: torch.Tensor
    segment_ids: Optional[torch.Tensor] = None

    def save(self, path: str | os.PathLike) -> None:
        """Сохраняет эмбеддинги на диск."""
        state = {
            "embeddings": self.embeddings,
            "labels": self.labels,
            "segment_ids": self.segment_ids,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | os.PathLike, device: torch.device = torch.device('cpu')) -> "EmbeddingSet":
        """Загружает эмбеддинги с диска."""
        state = torch.load(path, map_location=device, weights_only=True)
        return cls(
            embeddings=state["embeddings"],
            labels=state["labels"],
            segment_ids=state["segment_ids"]
        )

    def pool_by_segment(self, pooling_type: Literal["mean", "sum", "amin", "amax"] = "mean") -> "EmbeddingSet":
        """
        Векторизованная группировка эмбеддингов по segment_ids.
        Возвращает новый экземпляр EmbeddingSet с агрегированными данными.
        """
        if self.segment_ids is None:
            raise ValueError("segment_ids отсутствуют, пулинг невозможен.")
            
        if self.embeddings.numel() == 0:
            return EmbeddingSet(self.embeddings.clone(), self.labels.clone(), self.segment_ids.clone())
            
        unique_segments, inverse_indices = torch.unique(self.segment_ids, return_inverse=True)
        num_segments = unique_segments.size(0)
        
        pooled_x = torch.zeros(
            (num_segments, self.embeddings.size(1)), 
            dtype=self.embeddings.dtype, 
            device=self.embeddings.device
        )
        pooled_x.scatter_reduce_(
            dim=0, 
            index=inverse_indices.unsqueeze(1).expand_as(self.embeddings), 
            src=self.embeddings, 
            reduce=pooling_type, 
            include_self=False
        )
        
        pooled_y = torch.empty(num_segments, dtype=self.labels.dtype, device=self.labels.device)
        pooled_y.scatter_(dim=0, index=inverse_indices, src=self.labels)
            
        return EmbeddingSet(embeddings=pooled_x, labels=pooled_y, segment_ids=unique_segments)


class EmbeddingExtractor:
    """Класс для извлечения эмбеддингов из датасетов с использованием энкодера."""
    
    def __init__(self, encoder: nn.Module, device: torch.device):
        self.encoder = encoder.to(device)
        self.device = device

    def extract_from_graph_dataset(
        self, 
        dataset: Dataset, 
        batch_size: int = 128,
        num_workers: int = 4,
        ignore_class: int = -1,
        desc: str = "Extracting"
    ) -> EmbeddingSet:
        """
        Извлекает эмбеддинги из графового датасета.
        Поддерживает батчи с наличием или отсутствием segment_id.
        """
        self.encoder.eval()
        
        embeddings_list, labels_list, segment_ids_list = [], [], []
        has_segments = False
        
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            collate_fn=lambda x: Batch.from_data_list(x)
        )
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                
                valid_mask = batch.y != ignore_class
                if not valid_mask.any():
                    continue

                batch = batch.to(self.device)
                valid_mask = valid_mask.to(self.device)
                pooled_emb = self.encoder(batch)

                embeddings_list.append(pooled_emb[valid_mask].cpu())
                labels_list.append(batch.y[valid_mask].cpu())
                
                if hasattr(batch, 'segment_id') and batch.segment_id is not None:
                    has_segments = True
                    segment_ids_list.append(batch.segment_id[valid_mask].cpu())
                
        if len(embeddings_list) == 0:
            out_channels = getattr(self.encoder, 'out_channels', 0)
            return EmbeddingSet(
                embeddings=torch.empty((0, out_channels)), 
                labels=torch.empty(0), 
                segment_ids=torch.empty(0) if has_segments else None
            )
            
        return EmbeddingSet(
            embeddings=torch.cat(embeddings_list),
            labels=torch.cat(labels_list),
            segment_ids=torch.cat(segment_ids_list) if has_segments else None
        )
    

def main():

    encoder_folder = "/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/src/experiment/train_val/checkpoints/ep200"
    dataset_path = "/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/datasets/dataset_sph_minnie65_r=1.5"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = load_encoder_from_folder(encoder_folder)
    encoder.eval().requires_grad_(False)
    gen_normalize = GenNormalize(transforms=[], mask_transform=None)

    print(f"Loading dataset from: {dataset_path}")
    get_class = make_minnie65_class_getter('/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/public_cave_ground_truth_cell_types_with_nucleus.csv')
    dataset = GraphDataSet(path=dataset_path,get_class=get_class, transform=gen_normalize)
    macro_mean, macro_std = compute_macro_stats(dataset)
    encoder = GraphLatent(
        encoder=encoder,
        macro_mean=macro_mean,
        macro_std=macro_std,
        pooling=global_add_pool,
    ).eval().requires_grad_(False).to(device)
    extractor = EmbeddingExtractor(encoder=encoder, device=device)
    emb_set = extractor.extract_from_graph_dataset(
        dataset=dataset,
        batch_size=1024,
        num_workers=2
    )
    pooling_type = 'sum'
    if pooling_type is not None:
        if emb_set.segment_ids is not None:
            print(f"Выполняется пулинг (тип: {pooling_type})...")
            emb_set = emb_set.pool_by_segment(pooling_type=pooling_type)
        else:
            print("Предупреждение: pooling_type указан, но segment_ids отсутствуют в датасете. Пулинг пропущен.")

    save_path = "/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/datasets/embeddings/emb_set.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    emb_set.save(save_path)
    print(f"Embeddings saved to: {save_path}")

if  __name__ == "__main__":
    main()