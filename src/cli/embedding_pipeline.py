import os
from dataclasses import dataclass
from typing import Callable, Literal, Optional
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
from torchmetrics import Accuracy, F1Score, AUROC, AveragePrecision
import gc
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import DictConfig

class LinearClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.head = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

def _bootstrap_ci(values, confidence: float = 0.95, n_boot: int = 10000, seed: int = 0):
    """95% доверительный интервал среднего через bootstrap по значениям прогонов."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    boot_means = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    lo = float(np.quantile(boot_means, (1 - confidence) / 2))
    hi = float(np.quantile(boot_means, 1 - (1 - confidence) / 2))
    return lo, hi


def summarize_cv(fold_metrics, metric_names=("Accuracy", "F1", "ROC-AUC", "PR-AUC")):
    """Печатает все прогоны + mean ± std + bootstrap 95% CI. Возвращает dict со сводкой.

    metric_names обрезается до числа колонок в fold_metrics, поэтому функция
    совместима и со старым форматом (acc, f1), и с новым (acc, f1, roc-auc, pr-auc).
    """
    arr = np.asarray(fold_metrics, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    metric_names = tuple(metric_names)[: arr.shape[1]]
    print("\n" + "=" * 60)
    print("  CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    for i, row in enumerate(arr):
        cells = ",  ".join(f"{name} = {val:.4f}" for name, val in zip(metric_names, row))
        print(f"  Run {i + 1:>2}:  {cells}")
    print("-" * 60)
    summary = {}
    for j, name in enumerate(metric_names):
        col = arr[:, j]
        mean, std = float(col.mean()), float(col.std())
        lo, hi = _bootstrap_ci(col)
        summary[name] = {"mean": mean, "std": std, "ci95_low": lo, "ci95_high": hi, "n": int(col.size)}
        print(f"  {name:>8}: {mean:.4f} ± {std:.4f}   95% CI [{lo:.4f}, {hi:.4f}]   (n={col.size})")
    print("=" * 60)
    return summary
from sklearn.utils.class_weight import compute_class_weight

def train_cv(
    cfg: DictConfig,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    class_names=None,
    save_path: Optional[str] = None,
    classifier_factory: Optional[Callable[[int, int], nn.Module]] = None,
):
    """Repeated Stratified K-Fold CV.

    Возвращает список (acc, f1, roc_auc, pr_auc) по всем n_repeats × n_splits
    прогонам — для оценки доверительного интервала (см. summarize_cv).
    ROC-AUC и PR-AUC (AveragePrecision) считаются по вероятностям, macro по классам.

    Сплиты фиксируются сидом cfg.seed (+ номер повтора), поэтому при равных
    n_repeats/n_splits разные энкодеры/бейзлайны оцениваются на ОДНИХ И ТЕХ ЖЕ
    разбиениях — это позволяет делать честное paired-сравнение.

    classifier_factory(in_channels, num_classes) -> nn.Module позволяет задать
    голову (по умолчанию линейный probe LinearClassifier).
    """
    cls_cfg = cfg.classifier
    num_classes = cls_cfg.get("num_classes", 2)
    n_splits    = cls_cfg["n_splits"]
    n_repeats   = cls_cfg.get("n_repeats", 1)
    batch_size  = cls_cfg.batch_size
    max_epochs  = cls_cfg["max_epochs"]

    if classifier_factory is None:
        classifier_factory = lambda in_channels, n_cls: LinearClassifier(in_channels, n_cls)

    x_np = x_all.cpu().numpy()
    y_np = y_all.cpu().numpy()

    fold_metrics = []
    best_acc = -1.0
    best_state: Optional[dict] = None

    total = n_repeats * n_splits
    # Диагностика сплита (на нейрон-уровне, если данные были запулены по нейронам)
    n = len(x_np)
    test_frac = 1.0 / n_splits
    val_frac = (1.0 - test_frac) * 0.2
    train_frac = (1.0 - test_frac) * 0.8
    classes, counts = np.unique(y_np, return_counts=True)
    print(f"\nStarting {n_repeats}×{n_splits}-Fold Cross Validation ({total} runs)...")
    print(f"  Объектов: {n} | классы (label:count): {dict(zip(classes.tolist(), counts.tolist()))}")
    print(f"  Сплит на прогон: train≈{train_frac:.0%} ({int(round(n*train_frac))}), "
          f"val≈{val_frac:.0%} ({int(round(n*val_frac))}), test≈{test_frac:.0%} ({int(round(n*test_frac))}) "
          f"— непересекающиеся объекты, стратификация по классу")

    for repeat in range(n_repeats):
        seed = int(cfg.seed) + repeat
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        for fold, (train_val_idx, test_idx) in enumerate(skf.split(x_np, y_np)):
            run = repeat * n_splits + fold + 1
            print(f"\n{'='*40}\n Repeat {repeat+1}/{n_repeats}  Fold {fold+1}/{n_splits}  (run {run}/{total})\n{'='*40}")

            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=0.2, random_state=seed, stratify=y_np[train_val_idx]
            )

            # Balancing loss for fold
            train_labels = y_np[train_idx]
            classes_present = np.unique(train_labels)
            
            weights = compute_class_weight(
                class_weight='balanced',
                classes=classes_present,
                y=train_labels
            )
            class_weights_tensor = torch.tensor(weights, dtype=torch.float32)

            def make_loader(idx, shuffle):
                ds = TensorDataset(x_all[idx], y_all[idx])
                return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, persistent_workers=True)

            train_loader = make_loader(train_idx, shuffle=True)
            val_loader   = make_loader(val_idx,   shuffle=False)
            test_loader  = make_loader(test_idx,  shuffle=False)

            module = EmbeddingsLightModule(
                classifier=classifier_factory(x_all.shape[1], num_classes),
                lr=cls_cfg["learning_rate"],
                wd=cls_cfg["weight_decay"],
                max_epochs=max_epochs,
                num_classes=num_classes,
                class_names=class_names,
                class_weights=class_weights_tensor
            )

            # выбор лучшей модели и early stopping — оба по val_loss (гладкий сигнал,
            # надёжнее val_acc на маленьком val)
            checkpoint_cb = L.callbacks.ModelCheckpoint(
                monitor="val_loss", mode="min", save_top_k=1,
                filename=f"cv_r{repeat}_f{fold}-{{epoch:02d}}-{{val_loss:.4f}}",
            )

            callbacks = [checkpoint_cb, L.callbacks.RichProgressBar()]
            patience = cls_cfg.get("early_stop_patience", 0)
            if patience and patience > 0:
                callbacks.append(L.callbacks.EarlyStopping(
                    monitor="val_loss", mode="min", patience=int(patience),
                ))

            trainer = L.Trainer(
                max_epochs=max_epochs,
                accelerator=cfg.trainer.get("accelerator", "gpu"),
                devices=cfg.trainer.get("devices", 1),
                logger=L.loggers.TensorBoardLogger(
                    save_dir=cfg["log_dir"],
                    name="cv_classifier",
                    version=f"r{repeat}_f{fold}",
                ),
                callbacks=callbacks,
                deterministic=True,
            )

            trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

            # ckpt_path="best" -> тестируем лучшую по val_loss модель (а не последнюю
            # эпоху); вместе с EarlyStopping снимает переобучение probe.
            results = trainer.test(module, dataloaders=test_loader, ckpt_path="best", verbose=False)
            if results:
                fold_acc   = results[0].get("test_acc",   0.0)
                fold_f1    = results[0].get("test_f1",    0.0)
                fold_auroc = results[0].get("test_auroc", float("nan"))
                fold_ap    = results[0].get("test_ap",    float("nan"))
                fold_metrics.append((fold_acc, fold_f1, fold_auroc, fold_ap))
                print(f"  run {run}/{total} -> Accuracy: {fold_acc:.4f}, F1: {fold_f1:.4f}, "
                      f"ROC-AUC: {fold_auroc:.4f}, PR-AUC: {fold_ap:.4f}")

                if save_path is not None and fold_acc > best_acc:
                    best_acc = fold_acc
                    best_state = {
                        "in_channels": x_all.shape[1],
                        "num_classes": num_classes,
                        "state_dict": module.classifier.state_dict(),
                    }

            del module, trainer, checkpoint_cb
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if save_path is not None and best_state is not None:
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, "best_classifier.pt")
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(best_state, save_path)
        print(f"Best classifier (fold acc={best_acc:.4f}) saved to {save_path}")

    return fold_metrics



class EmbeddingsLightModule(L.LightningModule):
    def __init__(self, classifier, lr, wd, max_epochs, num_classes, class_names=None, class_weights = None):
        super().__init__()
        self.classifier = classifier
        self.lr = lr
        self.wd = wd
        self.max_epochs = max_epochs
        self.num_classes = num_classes
        self.class_names = class_names
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        metric_kwargs = dict(task="multiclass", num_classes=num_classes)
        self.train_acc = Accuracy(**metric_kwargs, average=None)
        self.val_acc   = Accuracy(**metric_kwargs, average=None)
        self.test_acc  = Accuracy(**metric_kwargs, average=None)
        self.train_f1  = F1Score(**metric_kwargs, average="macro")
        self.val_f1    = F1Score(**metric_kwargs, average="macro")
        self.test_f1   = F1Score(**metric_kwargs, average="macro")
        # AUROC/PR-AUC требуют вероятностей (не argmax); macro по классам.
        # AveragePrecision == площадь под precision-recall (PR-AUC).
        self.val_auroc  = AUROC(**metric_kwargs, average="macro")
        self.test_auroc = AUROC(**metric_kwargs, average="macro")
        self.val_ap     = AveragePrecision(**metric_kwargs, average="macro")
        self.test_ap    = AveragePrecision(**metric_kwargs, average="macro")

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
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.val_auroc(probs, y)
        self.val_ap(probs, y)
        self.log("val_loss", self.loss_fn(logits, y), prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)
        self.log("val_auroc", self.val_auroc, prog_bar=True)
        self.log("val_ap", self.val_ap, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self._log_class_acc(acc, "val")
        self.log("val_acc", acc.mean(), prog_bar=True)
        self.val_acc.reset()

    def test_step(self, batch, _):
        x, y = batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.test_auroc(probs, y)
        self.test_ap(probs, y)
        self.log("test_f1", self.test_f1)
        self.log("test_auroc", self.test_auroc)
        self.log("test_ap", self.test_ap)

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

        if torch.is_floating_point(self.segment_ids):
            raise TypeError(
                f"segment_ids должен быть целочисленным (int64), получено {self.segment_ids.dtype}. "
                "Float-тип схлопнет разные нейроны в один сегмент при группировке."
            )

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