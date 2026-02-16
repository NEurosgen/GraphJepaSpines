import torch
from torch import nn
import pytorch_lightning as L
import numpy as np
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Batch
from sklearn.metrics import f1_score as sklearn_f1_score

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from ..models.jepa import JepaLight
from ..data_utils.datamodule import GraphDataModule, GraphDataSet
from ..data_utils.transforms import (
    GenNormalize,
    NormNoEps,
    EdgeNorm,
    GraphPruning,
    FeatureChoice,
)
from ..cli.train_model import load_stats, build_transforms

torch.set_float32_matmul_precision('high')


# ─── Model ────────────────────────────────────────────────────────────────

class LinearClassifier(nn.Module):
    """Simple linear probe on top of frozen graph embeddings."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.fd =  nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            )
            
        self.head = nn.Linear(in_channels, num_classes)
        

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        out = self.fd(embed)
        return self.head(out)


# ─── Lightning Module ─────────────────────────────────────────────────────

class ClassifierLightModule(L.LightningModule):
    """
    Wraps a frozen JEPA encoder + linear classifier head.
    
    The encoder produces node-level embeddings which are pooled (global_mean_pool)
    to graph-level embeddings, then passed through a linear head.
    """

    def __init__(self, encoder: nn.Module, classifier: LinearClassifier,
                 learning_rate: float = 1e-3, sigma: float = 1.0,
                 class_names: list = None):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder', 'classifier'])

        self.encoder = encoder
        self.encoder.requires_grad_(False)
        self.encoder.eval()

        self.classifier = classifier
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.15, 1]),ignore_index=2, label_smoothing=0.05)

        self.class_names = class_names

        self._test_preds = []
        self._test_labels = []
        self._test_embeddings = []
        self._test_segment_ids = []

    def _encode_graph(self, batch) -> torch.Tensor:
        """Encode graph batch → graph-level embedding."""
        edge_attr = batch.edge_attr
        if edge_attr is not None:
            edge_attr = torch.exp(-edge_attr ** 2 / self.sigma ** 2)
        node_emb = self.encoder(batch.x, batch.edge_index, edge_attr)
        graph_emb = global_mean_pool(node_emb, batch.batch)
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
        loss = self.loss_fn(logits, labels)
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
        loss = self.loss_fn(logits, labels)
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
        loss = self.loss_fn(logits, labels)
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
        return torch.optim.Adam(self.classifier.parameters(), lr=self.learning_rate)


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
        encoder_prefix = "model.encoder."
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
    encoder.requires_grad_(False)
    encoder.eval()


    num_classes = cls_cfg.get("num_classes", 12)
    embed_dim = cfg.network.encoder.out_channels
    classifier_head = LinearClassifier(in_channels=embed_dim, num_classes=num_classes)


    class_names = ["excitatory", "inhibitory"]

    module = ClassifierLightModule(
        encoder=encoder,
        classifier=classifier_head,
        learning_rate=cls_cfg.get("learning_rate", 1e-3),
        sigma=cls_cfg.get("sigma", 1.0),
        class_names=class_names[:num_classes],
    )


    dm_cfg = cfg.datamodule
    mean_x, std_x, mean_edge, std_edge = load_stats(dm_cfg.dataset.stats_path)
    transforms = build_transforms(dm_cfg, mean_x, std_x, mean_edge, std_edge)
    gen_normalize = GenNormalize(transforms=transforms, mask_transform=None)

    ds = GraphDataSet(
        path=dm_cfg.dataset.path,
        transform=gen_normalize,
        class_path=dm_cfg.dataset.class_path,
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
