"""
Нелинейная probe-оценка энкодера на h01: kNN или небольшой MLP (вместо линейного).

Всё как в evaluate_encoder_cv: энкодер из cfg.classifier.checkpoint_path, h01
канонизируется на лету (свои статистики), эмбеддинги пулятся по нейрону, затем
repeated Stratified K-fold + 95% CI (summarize_cv). Меняется только голова probe.

Конфиг:
  cfg.classifier.probe : "knn" | "mlp"   (по умолчанию knn)
  cfg.classifier.knn_k : число соседей для kNN (по умолчанию 5)
  cfg.transfer.task    : celltype | layer (как в evaluate_encoder_cv)

Запуск:
  python -m src.cli.inference.h01.evalaute_encoder_nonlin_cv                       # knn
  python -m src.cli.inference.h01.evalaute_encoder_nonlin_cv classifier.probe=mlp
  python -m src.cli.inference.h01.evalaute_encoder_nonlin_cv classifier.knn_k=10 transfer.task=layer
"""
import hydra
import numpy as np
import pytorch_lightning as L
import torch
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from src.cli.embedding_pipeline import train_cv, summarize_cv
from src.cli.inference.h01.evaluate_encoder_cv import extract_all_embeddings, TASKS

torch.set_float32_matmul_precision("high")


class MLPClassifier(nn.Module):
    """Небольшой MLP-probe (один скрытый слой + dropout)."""

    def __init__(self, in_channels: int, num_classes: int, hidden: int = None, dropout: float = 0.3):
        super().__init__()
        hidden = hidden or max(in_channels // 2, num_classes * 4)
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def knn_cv(cfg: DictConfig, x_all: torch.Tensor, y_all: torch.Tensor, k: int):
    """Repeated Stratified K-fold с kNN (sklearn). Сиды совпадают с train_cv
    (cfg.seed + repeat) -> те же разбиения, paired-сравнение с linear/MLP.

    Per-fold StandardScaler фитится ТОЛЬКО на train (kNN чувствителен к масштабу,
    без leakage). val не нужен (у kNN нет обучения) -> фит на всём train-фолде.
    """
    cls_cfg = cfg.classifier
    n_splits = cls_cfg["n_splits"]
    n_repeats = cls_cfg.get("n_repeats", 1)
    x = x_all.cpu().numpy()
    y = y_all.cpu().numpy()

    classes, counts = np.unique(y, return_counts=True)
    total = n_repeats * n_splits
    print(f"\nkNN (k={k}) {n_repeats}×{n_splits}-Fold CV ({total} runs)...")
    print(f"  Объектов: {len(y)} | классы (label:count): {dict(zip(classes.tolist(), counts.tolist()))}")
    print(f"  Сплит на прогон: train≈{1 - 1 / n_splits:.0%}, test≈{1 / n_splits:.0%} — непересекающиеся объекты")

    metrics = []
    for repeat in range(n_repeats):
        seed = int(cfg.seed) + repeat
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr_idx, te_idx in skf.split(x, y):
            scaler = StandardScaler().fit(x[tr_idx])
            x_tr, x_te = scaler.transform(x[tr_idx]), scaler.transform(x[te_idx])
            clf = KNeighborsClassifier(n_neighbors=min(k, len(tr_idx)))
            clf.fit(x_tr, y[tr_idx])
            pred = clf.predict(x_te)
            metrics.append((
                accuracy_score(y[te_idx], pred),
                f1_score(y[te_idx], pred, average="macro"),
            ))
    return metrics


@hydra.main(version_base="1.3", config_path="../../../../configs", config_name="config")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task = cfg.transfer.get("task", "celltype")
    if task not in TASKS:
        raise ValueError(f"transfer.task='{task}' не поддерживается, ожидается одно из {list(TASKS)}")
    task_cfg = TASKS[task]
    cfg.classifier.num_classes = task_cfg["num_classes"]

    probe = cfg.classifier.get("probe", "knn")

    print("=" * 60)
    print(" h01 NON-LINEAR probe evaluation (neuron-level)")
    print(f" Encoder : {cfg.classifier.checkpoint_path}")
    print(f" Task    : {task} ({task_cfg['num_classes']} classes) | probe: {probe}")
    print("=" * 60)

    print("\n[1/2] Extracting Embeddings...")
    x_all, y_all = extract_all_embeddings(cfg, task_cfg, device)
    print(f"Features: {tuple(x_all.shape)}, Labels: {tuple(y_all.shape)}")

    print(f"\n[2/2] Training & Evaluating ({probe})...")
    if probe == "knn":
        k = int(cfg.classifier.get("knn_k", 5))
        fold_metrics = knn_cv(cfg, x_all, y_all, k)
    elif probe == "mlp":
        fold_metrics = train_cv(
            cfg, x_all, y_all,
            class_names=task_cfg["class_names"],
            classifier_factory=lambda in_ch, n_cls: MLPClassifier(in_ch, n_cls),
        )
    else:
        raise ValueError(f"classifier.probe='{probe}' не поддерживается (knn|mlp)")

    if not fold_metrics:
        print("No metrics collected!")
        return

    summarize_cv(fold_metrics)


if __name__ == "__main__":
    main()
