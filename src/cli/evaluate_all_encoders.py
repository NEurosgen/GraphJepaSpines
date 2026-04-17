"""
Pipeline: iterate over all pretrained JEPA encoders (jepa_r_*),
prepare dataset with the matching r, extract embeddings, train classifier.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as L
import torch
import shutil
import gc
import re
from pathlib import Path
from tqdm import tqdm
from torch_geometric.nn import global_add_pool
from torch.utils.data import TensorDataset, DataLoader

from src.models.loader_model import load_encoder_from_folder
from src.models.encoder import GraphLatent
from src.data_utils.datamodule import GraphDataSet, make_minnie65_class_getter
from src.data_utils.transforms import (
    GraphPruning, LaplacianPE, CentralityEncoding, RandomWalkPE, LocalPos,
    GenNormalize, FeatureShuffling
)
from src.data_utils.stats import compute_macro_stats
from src.cli.train_model import load_stats, build_transforms
from src.cli.extract_embeddings import extract_from_dataset
from src.cli.train_from_embeddings import pool_by_segment

torch.set_float32_matmul_precision('high')

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix # <-- добавлено
import numpy as np
# ──────────────────────────────────────────────────────────
#  Helper: find all pretrained encoder folders
# ──────────────────────────────────────────────────────────
def discover_encoder_folders(log_dir: str):
    """
    Scans log_dir for folders matching 'jepa_r_*'.
    For each folder picks the latest version_* subfolder.
    Returns list of (r_value, folder_path) tuples.
    """
    log_path = Path(log_dir)
    results = []

    for folder in sorted(log_path.iterdir()):
        if not folder.is_dir() or not folder.name.startswith("jepa_r_"):
            continue

        # Extract r and sh from the folder name (jepa_r_1.5_sh_1.0 or jepa_r_1.5)
        name = folder.name
        r_val = -1.0
        sh_val = 0.0
        
        # Try to find r_ value
        r_match = re.search(r'jepa_r_(-?\d+\.?\d*)', name)
        if r_match:
            r_val = float(r_match.group(1))
            
        # Try to find sh_ value
        sh_match = re.search(r'_sh_(\d+\.?\d*)', name)
        if sh_match:
            sh_val = float(sh_match.group(1))

        # Find latest version subfolder
        versions = sorted(
            [v for v in folder.iterdir() if v.is_dir() and v.name.startswith("version_")],
            key=lambda p: int(p.name.split("_")[1]),
        )
        if not versions:
            print(f"⚠ Skipping {folder.name}: no version_* subfolders")
            continue

        latest = versions[-1]
        results.append((r_val, sh_val, str(latest)))

    return results


# ──────────────────────────────────────────────────────────
#  Step 1: Prepare dataset for a given r and shuffle_ratio
# ──────────────────────────────────────────────────────────
def prepare_dataset_for_r(cfg: DictConfig, r: float, shuffle_ratio: float = 0.0):
    """Same logic as train_densities.py dataset preparation."""
    OmegaConf.set_struct(cfg, False)
    cfg.datamodule.r = r
    cfg.datamodule.shuffle_ratio = shuffle_ratio
    OmegaConf.set_struct(cfg, True)

    out_dir = Path(cfg.datamodule.dataset.path)
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)

    dataset_path = Path(cfg.datamodule.dataset.raw_path)
    file_paths = sorted(dataset_path.rglob('*.pt'))

    knn_k = cfg.datamodule.get('knn', -1)
    radius_r = cfg.datamodule.get('r', -1.0)
    mutual = cfg.datamodule.get('mutual_knn', False)
    pruning = GraphPruning(k=knn_k, r=radius_r, mutual=mutual)
    shuffling = FeatureShuffling(ratio=cfg.datamodule.get('shuffle_ratio', 0.0))

    se_cfg = cfg.datamodule.get('structural_encoding', {})
    lap_k = se_cfg.get('laplacian_k', 0)
    centrality = se_cfg.get('centrality', False)
    rw_steps = se_cfg.get('random_walk_steps', 0)

    for file_path in tqdm(file_paths, leave=False, desc=f"Preparing dataset r={r}, sh={shuffle_ratio}"):
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        data = pruning(LocalPos()(data))
        data = shuffling(data)
        x_dim_original = data.x.size(1)

        if lap_k > 0:
            data = LaplacianPE(k=lap_k)(data)
            data.laplacian_pe = data.x[:, x_dim_original:]
            data.x = data.x[:, :x_dim_original]

        if centrality:
            data = CentralityEncoding()(data)
            data.centrality_pe = data.x[:, x_dim_original:]
            data.x = data.x[:, :x_dim_original]

        if rw_steps > 0:
            data = RandomWalkPE(walk_length=rw_steps)(data)
            data.random_walk_pe = data.x[:, x_dim_original:]
            data.x = data.x[:, :x_dim_original]

        rel_path = file_path.relative_to(dataset_path)
        out_file = out_dir / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, out_file)


# ──────────────────────────────────────────────────────────
#  Step 2: Extract embeddings
# ──────────────────────────────────────────────────────────
def extract_embeddings_for_model(cfg: DictConfig, encoder_folder: str, device):
    """Loads encoder, builds dataset, extracts train/val/test embeddings after pooling."""
    cls_cfg = cfg.classifier
    dm_cfg = cfg.datamodule

    encoder = load_encoder_from_folder(encoder_folder)
    encoder.eval()
    encoder.requires_grad_(False)
    encoder.to(device)

    mean_x, std_x, mean_edge, std_edge = load_stats(cls_cfg.stats_path)
    transforms = build_transforms(dm_cfg, mean_x, std_x, mean_edge, std_edge)
    gen_normalize = GenNormalize(transforms=transforms, mask_transform=None)

    csv_path = dm_cfg.dataset.class_path
    get_class_fn = make_minnie65_class_getter(csv_path)

    ds_path = cls_cfg.path if Path(cls_cfg.path).exists() else dm_cfg.dataset.path
    ds = GraphDataSet(path=ds_path, get_class=get_class_fn, transform=gen_normalize)

    macro_mean, macro_std = compute_macro_stats(ds)

    encoder_graph = GraphLatent(
        encoder=encoder,
        macro_mean=macro_mean,
        macro_std=macro_std,
        pooling=global_add_pool,
        sigma=cls_cfg.get("sigma", 1.0),
    ).to(device)

    # 1. Извлекаем эмбеддинги для ВСЕГО датасета сразу
    emb_all, y_all, seg_all = extract_from_dataset(ds, encoder_graph, device, "All")

    # 2. Выполняем пулинг (агрегацию) по сегментам ДО разбиения
    pooling_level = cls_cfg.get("pooling_level", "graph")
    if pooling_level == "neuron":
        pooling_type = cls_cfg.get("pooling_type", "mean")
        x_pooled, y_pooled = pool_by_segment(emb_all, y_all, seg_all, pooling_type)
    else:
        x_pooled, y_pooled = emb_all, y_all

    # 3. Разбиваем уже сагрегированные данные на train/val/test
    generator = torch.Generator().manual_seed(dm_cfg.seed)
    perm = torch.randperm(len(x_pooled), generator=generator)

    ratio = dm_cfg.get("ratio", [0.7, 0.2, 0.1])
    train_size = int(len(x_pooled) * ratio[0])
    val_size = int(len(x_pooled) * ratio[1])

    train_indices = perm[:train_size]
    val_indices = perm[train_size:train_size + val_size]
    test_indices = perm[train_size + val_size:]

    output = {
        'train': {'x': x_pooled[train_indices], 'y': y_pooled[train_indices]},
        'val':   {'x': x_pooled[val_indices],   'y': y_pooled[val_indices]},
        'test':  {'x': x_pooled[test_indices],  'y': y_pooled[test_indices]},
    }

    # Cleanup encoder from GPU
    del encoder, encoder_graph
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output


# ──────────────────────────────────────────────────────────
#  Step 3: Train classifier (from train_from_embeddings.py)
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
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([
     0.4313,  0.3684,  0.8037, 11.7879,  7.0727,  0.7367,  0.5441,  4.4205,
         1.6840,  3.9293,  4.4205
], dtype=torch.float))

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


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as L
import gc
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
def train_classifier(cfg, emb_data, r_val, sh_val=0.0):
    """Trains a linear classifier using scikit-learn analytical solvers with data scaling."""
    
    X_train = emb_data['train']['x'].cpu().numpy()
    y_train = emb_data['train']['y'].cpu().numpy()
    
    X_val = emb_data['val']['x'].cpu().numpy()
    y_val = emb_data['val']['y'].cpu().numpy()
    
    X_test = emb_data['test']['x'].cpu().numpy()
    y_test = emb_data['test']['y'].cpu().numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(
        penalty='l2',
        C=1.0,                   
        class_weight='balanced', 
        solver='lbfgs', 
        max_iter=1000,
        random_state=cfg.get("seed", 42)
    )

    clf.fit(X_train, y_train)

    # Предсказания для всех сплитов
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)

    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    print(f"  [Sklearn] Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    
    cls_cfg = cfg.classifier
    class_names = cls_cfg.get("class_names", None)
    if not class_names:
        class_names = ['23P', '4P', '5P-IT', '5P-NP', '5P-PT', '6P-CT', '6P-IT', 'BC', 'BPC', 'MC', 'NGC']
    
    # Расчет матриц ошибок для train, val и test
    labels = np.arange(len(class_names))
    cm_train = confusion_matrix(y_train, y_train_pred, labels=labels)
    cm_val = confusion_matrix(y_val, y_val_pred, labels=labels)
    cm_test = confusion_matrix(y_test, y_test_pred, labels=labels)

    results = {
        "test_acc": float(test_acc),
        "test_f1": float(test_f1),
        "cm_train": cm_train,       # Матрица для Train
        "cm_val": cm_val,           # Матрица для Val
        "cm_test": cm_test,         # Матрица для Test
        "class_names": class_names
    }
    
    classes = np.unique(y_test)
    for cls_np in classes:
        cls = int(cls_np) 
        idx = (y_test == cls_np)
        if np.sum(idx) > 0:
            cls_acc = accuracy_score(y_test[idx], y_test_pred[idx])
            name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
            results[f"test_acc_{name}"] = float(cls_acc)

    return results

# ──────────────────────────────────────────────────────────
#  Main: orchestrate the full pipeline
# ──────────────────────────────────────────────────────────
@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_dir = cfg.get("log_dir", "lightning_logs")
    #encoders = discover_encoder_folders(log_dir)
    encoders = [#(0, 0,"/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/lightning_logs/jepa_r_0_sh_0/version_2"),
                (1, 0,"/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/lightning_logs/jepa_r_1_sh_0/version_1"),
                #(1.5, 0,"/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/lightning_logs/jepa_r_1.5_sh_0/version_1"),
                #(2, 0,"/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/lightning_logs/jepa_r_2_sh_0/version_1"),
                #(3, 0,"/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/lightning_logs/jepa_r_3_sh_0/version_0"),
                ]
                
    if not encoders:
        print("No jepa_r_* encoder folders found!")
        return

    print(f"Found {len(encoders)} pretrained encoders:")
    for r_val, sh_val, folder in encoders:
        print(f"  r={r_val:>6}  →  {folder}")
    print()

    summary = []

# Замените цикл for r_val, sh_val, encoder_folder in encoders: на этот:

    for r_val, sh_val, encoder_folder in encoders:
        print("=" * 60)
        print(f"  r = {r_val}, sh = {sh_val}")
        print("=" * 60)
        
        print(f"[1/3] Preparing dataset with r={r_val}, sh={sh_val} ...")
        #prepare_dataset_for_r(cfg, r_val, sh_val)

        OmegaConf.set_struct(cfg, False)
        cfg.classifier.checkpoint_path = encoder_folder
        OmegaConf.set_struct(cfg, True)

        print(f"[2/3] Extracting embeddings ...")
        #emb_data = extract_embeddings_for_model(cfg, encoder_folder, device)

        emb_dir = Path(cfg.classifier.get("extracted_embeddings_path", "data/embeddings/embeddings.pt")).parent
        emb_dir.mkdir(parents=True, exist_ok=True)
        emb_path = emb_dir / f"r_{r_val}_embeddings.pt"

        print(f"[3/3] Training classifier ...")

        emb_data = torch.load(emb_path)
        metrics = train_classifier(cfg, emb_data, r_val, sh_val)

        test_acc = metrics.get("test_acc", float('nan'))
        test_f1  = metrics.get("test_f1", float('nan'))
        # ... (предыдущий код в цикле внутри main)
        print(f"\n  ✓ r={r_val} sh={sh_val} RESULTS:")
        print(f"  Overall Acc: {test_acc:.4f}, Overall F1: {test_f1:.4f}")
        
        class_accs = {k: v for k, v in metrics.items() if k.startswith("test_acc_")}
        if class_accs:
            print("  Per-class Accuracy:")
            for k, v in sorted(class_accs.items()):
                class_name = k.replace("test_acc_", "")
                print(f"    {class_name:<10}: {v:.4f}")
                
        # ─── Вывод матриц ошибок для всех сплитов ───
        class_names = metrics.get("class_names", [])
        splits_to_print = [
            ("Train", "cm_train"), 
            ("Validation", "cm_val"), 
            ("Test", "cm_test")
        ]
        
        for split_name, cm_key in splits_to_print:
            cm = metrics.get(cm_key)
            if cm is not None:
                print(f"\n  {split_name} Confusion Matrix (True \\ Predicted):")
                header = f"{'':>8} " + " ".join([f"{name:>6}" for name in class_names])
                print("  " + header)
                for i, row in enumerate(cm):
                    row_name = class_names[i]
                    row_str = f"{row_name:>8} " + " ".join([f"{val:>6}" for val in row])
                    print("  " + row_str)
        # ────────────────────────────────────────────
        
        print("-" * 60 + "\n")

        del emb_data
        gc.collect()
    # ── Summary table ──
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'r':>8}  {'sh':>8}  {'test_acc':>10}  {'test_f1':>10}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}")
    for r_val, sh_val, acc, f1 in summary:
        print(f"  {r_val:>8}  {sh_val:>8.2f}  {acc:>10.4f}  {f1:>10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
