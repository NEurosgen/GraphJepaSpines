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
    """Loads encoder, builds dataset, extracts train/val/test embeddings."""
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

    # Split identically to training
    generator = torch.Generator().manual_seed(dm_cfg.seed)
    perm = torch.randperm(len(ds), generator=generator)

    ratio = dm_cfg.get("ratio", [0.7, 0.2, 0.1])
    train_size = int(len(ds) * ratio[0])
    val_size = int(len(ds) * ratio[1])

    train_ds = ds[perm[:train_size]]
    val_ds = ds[perm[train_size:train_size + val_size]]
    test_ds = ds[perm[train_size + val_size:]]

    emb_train, y_train, seg_train = extract_from_dataset(train_ds, encoder_graph, device, "Train")
    emb_val, y_val, seg_val = extract_from_dataset(val_ds, encoder_graph, device, "Val")
    emb_test, y_test, seg_test = extract_from_dataset(test_ds, encoder_graph, device, "Test")

    output = {
        'train': {'x': emb_train, 'y': y_train, 'seg': seg_train},
        'val':   {'x': emb_val,   'y': y_val,   'seg': seg_val},
        'test':  {'x': emb_test,  'y': y_test,  'seg': seg_test},
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
    def __init__(self, classifier, lr, wd, max_epochs, num_classes):
        super().__init__()
        self.classifier = classifier
        self.lr = lr
        self.wd = wd
        self.max_epochs = max_epochs
        self.loss_fn = torch.nn.CrossEntropyLoss()

        from torchmetrics import Accuracy, F1Score
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc   = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc  = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1   = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1  = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y); self.train_f1(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)
        self.log("train_f1", self.train_f1, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y); self.val_f1(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y); self.test_f1(preds, y)
        self.log("test_acc", self.test_acc)
        self.log("test_f1", self.test_f1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def train_classifier(cfg: DictConfig, emb_data: dict, r_val: float, sh_val: float = 0.0):
    """Trains linear classifier on extracted embeddings. Returns test metrics."""
    from src.models.classificator import LinearClassifier

    cls_cfg = cfg.classifier
    pooling_level = cls_cfg.get("pooling_level", "graph")
    pooling_type = cls_cfg.get("pooling_type", "mean")
    num_classes = cls_cfg.get("num_classes", 2)

    def prepare_subset(subset_data):
        x, y, seg = subset_data['x'], subset_data['y'], subset_data['seg']
        if pooling_level == "neuron":
            x, y = pool_by_segment(x, y, seg, pooling_type)
        return TensorDataset(x, y)

    train_ds = prepare_subset(emb_data['train'])
    val_ds   = prepare_subset(emb_data['val'])
    test_ds  = prepare_subset(emb_data['test'])

    batch_size = cfg.datamodule.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True)

    in_channels = emb_data['train']['x'].shape[1]
    classifier_head = LinearClassifier(in_channels=in_channels, num_classes=num_classes)

    max_epochs = cls_cfg.get("max_epochs", 500)
    module = EmbeddingsLightModule(
        classifier_head,
        lr=cls_cfg.get("learning_rate", 1e-3),
        wd=cls_cfg.get("weight_decay", 1e-5),
        max_epochs=max_epochs,
        num_classes=num_classes,
    )

    checkpoint_callback = L.callbacks.ModelCheckpoint(
        monitor="val_acc", mode="max", save_top_k=1,
        filename=f"cls-r_{r_val}-sh_{sh_val}-{{epoch:02d}}-{{val_acc:.4f}}",
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=cfg.trainer.get("accelerator", "gpu"),
        devices=cfg.trainer.get("devices", 1),
        logger = L.loggers.TensorBoardLogger(
            save_dir=cfg.get("log_dir", "lightning_logs"),
            name=f"emb_classifier_r_{r_val}_sh_{sh_val}",
        ),
        callbacks=[checkpoint_callback],
        deterministic=True,
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    results = trainer.test(module, dataloaders=test_loader)

    # Cleanup
    del module, trainer, checkpoint_callback
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results[0] if results else {}


# ──────────────────────────────────────────────────────────
#  Main: orchestrate the full pipeline
# ──────────────────────────────────────────────────────────
@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_dir = cfg.get("log_dir", "lightning_logs")
    encoders = discover_encoder_folders(log_dir)

    if not encoders:
        print("No jepa_r_* encoder folders found!")
        return

    print(f"Found {len(encoders)} pretrained encoders:")
    for r_val, sh_val, folder in encoders:
        print(f"  r={r_val:>6}  →  {folder}")
    print()

    summary = []

    for r_val, sh_val, encoder_folder in encoders:
        print("=" * 60)
        print(f"  r = {r_val}, sh = {sh_val}")
        print("=" * 60)
        
        # 1. Prepare dataset
        print(f"[1/3] Preparing dataset with r={r_val}, sh={sh_val} ...")
        prepare_dataset_for_r(cfg, r_val, sh_val)

        # Update classifier.checkpoint_path for this encoder
        OmegaConf.set_struct(cfg, False)
        cfg.classifier.checkpoint_path = encoder_folder
        OmegaConf.set_struct(cfg, True)

        # 2. Extract embeddings
        print(f"[2/3] Extracting embeddings ...")
        emb_data = extract_embeddings_for_model(cfg, encoder_folder, device)

        # Save embeddings
        emb_dir = Path(cfg.classifier.get(
            "extracted_embeddings_path",
            "data/embeddings/embeddings.pt"
        )).parent
        emb_dir.mkdir(parents=True, exist_ok=True)
        emb_path = emb_dir / f"r_{r_val}_embeddings.pt"
        torch.save(emb_data, emb_path)
        print(f"  Embeddings saved → {emb_path}")

        # 3. Train classifier
        print(f"[3/3] Training classifier ...")
        metrics = train_classifier(cfg, emb_data, r_val, sh_val)

        test_acc = metrics.get("test_acc", float('nan'))
        test_f1  = metrics.get("test_f1", float('nan'))
        summary.append((r_val, sh_val, test_acc, test_f1))

        print(f"  ✓ r={r_val} sh={sh_val} test_acc={test_acc:.4f}  test_f1={test_f1:.4f}\n")

        # Free embeddings
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
