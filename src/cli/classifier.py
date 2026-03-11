import torch
from torch import nn
import pytorch_lightning as L
import numpy as np
from torch_geometric.nn import global_add_pool
from torch_geometric.data import Batch
from sklearn.metrics import f1_score as sklearn_f1_score

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from ..models.jepa import JepaLight
from ..data_utils.datamodule import GraphDataModule, GraphDataSet, make_folder_class_getter
from ..data_utils.transforms import (
    GenNormalize,
    NormNoEps,
    EdgeNorm,
    GraphPruning,
    FeatureChoice,
)
from .train_model import load_stats, build_transforms
from hydra.utils import instantiate
from omegaconf import OmegaConf
torch.set_float32_matmul_precision('high')


# ─── Model ────────────────────────────────────────────────────────────────

class LinearClassifier(nn.Module):
    """Simple linear probe on top of frozen graph embeddings."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Dropout(0.3), # Strong dropout for regularization on 41 training samples
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, num_classes)

        )

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        return self.head(embed)


# ─── Lightning Module ─────────────────────────────────────────────────────

class ClassifierLightModule(L.LightningModule):
    """
    Wraps a frozen JEPA encoder + linear classifier head.
    
    The encoder produces node-level embeddings which are pooled (global_mean_pool)
    to graph-level embeddings, then passed through a linear head.
    """

    def __init__(self, encoder: nn.Module, classifier: LinearClassifier,
                 learning_rate: float = 1e-3, sigma: float = 1.0,
                 class_names: list = None, cfg: DictConfig = None):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder', 'classifier'])

        self.encoder = encoder
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.classifier = classifier
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1., 1.0]),reduction='none')
        self.optimizer_cfg = cfg.optimizer
        self.scheduler_cfg = cfg.get("scheduler", None)
        self.class_names = class_names

        self._test_preds = []
        self._test_labels = []

        self._test_embeddings = []
        self._test_segment_ids = []

    def _encode_graph(self, batch) -> torch.Tensor:
        """Encode graph batch → graph-level embedding."""
        self.encoder.eval() # Prevent Lightning from putting it in train mode
        edge_attr = batch.edge_attr
        if edge_attr is not None:
            edge_attr = torch.exp(-edge_attr ** 2 / self.sigma ** 2)
        node_emb = self.encoder(batch.x, batch.edge_index, edge_attr)
        graph_emb = global_add_pool(node_emb, batch.batch)
        
        # --- Add Graph Macro-Features ---
        num_graphs = batch.batch.max().item() + 1
        num_nodes = torch.bincount(batch.batch, minlength=num_graphs).float()
        
        if batch.edge_index.numel() > 0:
            edge_batch = batch.batch[batch.edge_index[0]]
            num_edges = torch.bincount(edge_batch, minlength=num_graphs).float()
        else:
            num_edges = torch.zeros(num_graphs, device=batch.x.device)
            
        density = num_edges / (num_nodes * (num_nodes - 1)).clamp(min=1)
        
        # Normalize features roughly based on target dataset statistics
        num_nodes_norm = (num_nodes - 11.0) / 3.5
        num_edges_norm = (num_edges - 45.0) / 15.0
        
        basic_macro = torch.stack([num_nodes_norm, num_edges_norm, density], dim=-1).to(graph_emb.dtype)
        
        # Pull the ThesisMacroMetrics pre-computed during Data Loading
        # shape of batch.macro_metrics is [num_nodes, 3], but it's identical for all nodes in the same graph.
        # So we can just pool it with mean or scatter
        if hasattr(batch, 'macro_metrics') and batch.macro_metrics is not None:
            from torch_geometric.utils import scatter
            # macro_metrics is attached at graph-level by the transform, but PyG batching 
            # might have concatenated it to [num_nodes, 3] or [num_graphs, 3] depending on how it was saved.
            # Assuming it was saved as a graph-level attribute:
            if batch.macro_metrics.size(0) == num_graphs:
                thesis_macro = batch.macro_metrics.to(graph_emb.dtype)
            else:
                # Fallback if PyG replicated it per-node or something weird
                thesis_macro = scatter(batch.macro_metrics, batch.batch, dim=0, reduce='mean').to(graph_emb.dtype)
        else:
            thesis_macro = torch.zeros((num_graphs, 4), dtype=graph_emb.dtype, device=graph_emb.device)
        
        macro_features = torch.cat([basic_macro, thesis_macro], dim=-1)
        graph_emb = torch.cat([graph_emb, macro_features], dim=-1)
        # --------------------------------
        
        return graph_emb

    def forward(self, batch):
        with torch.no_grad():
            graph_emb = self._encode_graph(batch)
        logits = self.classifier(graph_emb)
        return logits

    def _compute_f1(self, preds: torch.Tensor, labels: torch.Tensor) -> float:
        """Macro F1 on CPU via sklearn."""
        return sklearn_f1_score(
            labels.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0
        )

    def training_step(self, batch):
        logits = self.forward(batch)
        labels = batch.y.long()
        loss = self.loss_fn(logits, labels).mean()
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        f1 = self._compute_f1(preds, labels)
        self.log("train_loss", loss, prog_bar=True, batch_size=labels.size(0))
        self.log("train_acc", acc, prog_bar=True, batch_size=labels.size(0))
        self.log("train_f1", f1, prog_bar=True, batch_size=labels.size(0))
        return loss

    def validation_step(self, batch):
        logits = self.forward(batch)
        labels = batch.y.long()
        loss = self.loss_fn(logits, labels).mean()
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        f1 = self._compute_f1(preds, labels)
        self.log("val_loss", loss, prog_bar=True, batch_size=labels.size(0))
        self.log("val_acc", acc, prog_bar=True, batch_size=labels.size(0))
        self.log("val_f1", f1, prog_bar=True, batch_size=labels.size(0))
        return loss

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            graph_emb = self._encode_graph(batch)
        logits = self.classifier(graph_emb)
        labels = batch.y.long()
        loss = self.loss_fn(logits, labels).mean()
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        f1 = self._compute_f1(preds, labels)
        self.log("test_loss", loss, prog_bar=True, batch_size=labels.size(0))
        self.log("test_acc", acc, prog_bar=True, batch_size=labels.size(0))
        self.log("test_f1", f1, prog_bar=True, batch_size=labels.size(0))


        self._test_embeddings.append(graph_emb.cpu())
        self._test_labels.append(labels.cpu())
        self._test_segment_ids.append(batch.segment_id.cpu())
        return loss

    def on_test_epoch_end(self):
        """Aggregate embeddings by segment_id, classify, and print report."""
        from sklearn.metrics import classification_report, confusion_matrix
        from collections import defaultdict

        all_embeddings = torch.cat(self._test_embeddings, dim=0)
        all_labels = torch.cat(self._test_labels, dim=0)
        all_seg_ids = torch.cat(self._test_segment_ids, dim=0)

        with torch.no_grad():
            per_graph_logits = self.classifier(all_embeddings.to(self.device))
        per_graph_preds = per_graph_logits.argmax(dim=-1).cpu().numpy()

        # Get num_classes from the Linear layer inside Sequential
        if isinstance(self.classifier.head, nn.Sequential):
            num_classes = self.classifier.head[-1].out_features
        else:
            num_classes = self.classifier.head.out_features
        all_label_ids = list(range(num_classes))
        target_names = self.class_names[:num_classes] if self.class_names else None

        print("\n" + "=" * 60)
        print("Classification Report — Per Sub-graph (Test Set)")
        print("=" * 60)
        print(classification_report(all_labels.numpy(), per_graph_preds,
                                    labels=all_label_ids,
                                    target_names=target_names,
                                    zero_division=0))

        # --- Aggregate by segment_id ---
        seg_embeddings = defaultdict(list)
        seg_labels = {}
        for emb, label, seg_id in zip(all_embeddings, all_labels, all_seg_ids):
            sid = seg_id.item()
            seg_embeddings[sid].append(emb)
            seg_labels[sid] = label.item()

        agg_embs = []
        agg_labels = []
        for sid in sorted(seg_embeddings.keys()):
            stacked = torch.stack(seg_embeddings[sid], dim=0)
            agg_embs.append(stacked.sum(dim=0))
            agg_labels.append(seg_labels[sid])

        agg_embs_t = torch.stack(agg_embs, dim=0).to(self.device)
        agg_labels_np = np.array(agg_labels)

        with torch.no_grad():
            agg_logits = self.classifier(agg_embs_t)
        agg_preds = agg_logits.argmax(dim=-1).cpu().numpy()

        print("=" * 60)
        print(f"Classification Report — Per Neuron (aggregated by segment_id)")
        print(f"  {len(seg_embeddings)} unique neurons from {len(all_labels)} sub-graphs")
        print("=" * 60)
        print(classification_report(agg_labels_np, agg_preds,
                                    labels=all_label_ids,
                                    target_names=target_names,
                                    zero_division=0))
        print("Confusion Matrix (per neuron):")
        print(confusion_matrix(agg_labels_np, agg_preds))
        print("=" * 60 + "\n")

        self._test_embeddings.clear()
        self._test_labels.clear()
        self._test_segment_ids.clear()
        self._test_preds.clear()

    def configure_optimizers(self):
        params = list(self.classifier.parameters())
        opt_cfg = OmegaConf.to_container(self.optimizer_cfg, resolve=True)
        opt_target = opt_cfg.pop('_target_')
        
        import torch.optim as optim
        optimizer_class = getattr(optim, opt_target.split('.')[-1])
        optimizer = optimizer_class(params, lr=self.learning_rate, **opt_cfg)
        
        if self.scheduler_cfg is not None:
            sched_cfg = OmegaConf.to_container(self.scheduler_cfg, resolve=True)
            sched_target = sched_cfg.pop('_target_')
            scheduler_class = getattr(optim.lr_scheduler, sched_target.split('.')[-1])
            scheduler = scheduler_class(optimizer, **sched_cfg)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        
        return optimizer


# ─── Helpers ──────────────────────────────────────────────────────────────

def _simple_collate(data_list):
    """Collate for classification — no masking, just batch graphs."""
    return Batch.from_data_list(data_list)


def _load_encoder_from_checkpoint(ckpt_path: str, cfg: DictConfig) -> nn.Module:
    """
    Load encoder weights from a JepaLight checkpoint.
    Instantiate network from config, load state_dict, return encoder.
    """
    network = instantiate(cfg.network, _recursive_=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    encoder_prefix = "model.encoder."
    encoder_sd = {}
    for k, v in state_dict.items():
        if k.startswith(encoder_prefix):
            encoder_sd[k[len(encoder_prefix):]] = v

    if not encoder_sd:
        encoder_prefix = "model.student_encoder."
        for k, v in state_dict.items():
            if k.startswith(encoder_prefix):
                encoder_sd[k[len(encoder_prefix):]] = v

    if encoder_sd:
        network.encoder.load_state_dict(encoder_sd, strict=False)
        print(f"Loaded encoder from checkpoint: {ckpt_path}")
    else:
        print(f"WARNING: Could not find encoder weights in checkpoint, using randomly initialized encoder!")

    if hasattr(network, 'student_encoder'):
        return network.student_encoder
    return network.encoder


# ─── Main ─────────────────────────────────────────────────────────────────

@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)

    cls_cfg = cfg.classifier


    encoder = _load_encoder_from_checkpoint(cls_cfg.checkpoint_path, cfg)
  


    num_classes = cls_cfg.get("num_classes", 2)
    # The new embed dim integrates the macro graph features (3 items) + thesis metrics (4 items) = 7
    embed_dim = cfg.network.encoder.out_channels + 7
    classifier_head = LinearClassifier(in_channels=embed_dim, num_classes=num_classes)

    # Классы из конфига: class_names и folder_to_label
    class_names = list(cls_cfg.get("class_names", ["ab", "wt"]))
    folder_to_label = dict(cls_cfg.get("folder_to_label", {"ab": 0, "wt": 1}))
    get_class = make_folder_class_getter(folder_to_label)

    module = ClassifierLightModule(
        encoder=encoder,
        classifier=classifier_head,
        learning_rate=cls_cfg.get("learning_rate", 1e-3),
        sigma=cls_cfg.get("sigma", 1.0),
        class_names=class_names[:num_classes],
        cfg=cls_cfg
    )


    dm_cfg = cfg.datamodule
    mean_x, std_x, mean_edge, std_edge = load_stats(cls_cfg.stats_path)
    transforms = build_transforms(dm_cfg, mean_x, std_x, mean_edge, std_edge)
    gen_normalize = GenNormalize(transforms=transforms, mask_transform=None)

    ds = GraphDataSet(
        path=cls_cfg.path,
        get_class=get_class,
        transform=gen_normalize,
    )

    datamodule = GraphDataModule(
        ds,
        batch_size=dm_cfg.batch_size,
        num_workers=dm_cfg.num_workers,
        seed=dm_cfg.seed,
        ratio=dm_cfg.ratio,
        collate_fn=_simple_collate,
    )


    max_epochs = cls_cfg.get("max_epochs", 50)

    checkpoint_callback = L.callbacks.ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="classifier-{epoch:02d}-{val_acc:.4f}",
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=cfg.trainer.get("accelerator", "gpu"),
        devices=cfg.trainer.get("devices", 1),
        log_every_n_steps=cfg.trainer.get("log_every_n_steps", 10),
        callbacks=[checkpoint_callback],
        deterministic=True,
    )

    trainer.fit(module, datamodule=datamodule)


    print("\nRunning evaluation on test set...")
    trainer.test(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
