import torch
import hydra
from omegaconf import DictConfig
import pytorch_lightning as L
from torch.utils.data import TensorDataset, DataLoader

from src.models.classificator import LinearClassifier, ClassifierLightModule

def pool_by_segment(embeddings, labels, segment_ids, pooling_type="mean"):
    """
    Groups embeddings by segment_ids and applies mean or add pooling.
    Returns (pooled_embeddings, labels).
    """
    if len(embeddings) == 0:
        return embeddings, labels
        
    unique_segments = torch.unique(segment_ids)
    
    pooled_x = []
    pooled_y = []
    
    for seg_id in unique_segments:
        mask = segment_ids == seg_id
        x_seg = embeddings[mask]
        y_seg = labels[mask][0] # All labels should be the same
        
        if pooling_type == "mean":
            x_pool = x_seg.mean(dim=0)
        else: # add
            x_pool = x_seg.sum(dim=0)
            
        pooled_x.append(x_pool)
        pooled_y.append(y_seg)
        
    return torch.stack(pooled_x), torch.stack(pooled_y)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)
    cls_cfg = cfg.classifier
    
    # 1. Load data
    data_path = cls_cfg.get("extracted_embeddings_path", "embeddings.pt")
    print(f"Loading embeddings from {data_path}...")
    try:
        data = torch.load(data_path, weights_only=False)
    except FileNotFoundError:
        print("Embeddings not found! Run python -m src.cli.extract_embeddings first.")
        return
        
    pooling_level = cls_cfg.get("pooling_level", "graph")
    pooling_type = cls_cfg.get("pooling_type", "mean")
    
    print(f"Mode: {pooling_level} level (pooling: {pooling_type})")
    
    def prepare_subset(subset_data):
        x, y, seg = subset_data['x'], subset_data['y'], subset_data['seg']
        if pooling_level == "neuron":
            x, y = pool_by_segment(x, y, seg, pooling_type)
        return TensorDataset(x, y)
        
    train_ds = prepare_subset(data['train'])
    val_ds = prepare_subset(data['val'])
    test_ds = prepare_subset(data['test'])
    
    batch_size = cfg.datamodule.batch_size
    # Since embeddings are small, we can batch them efficiently
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True)
    
    # 2. Setup classifier
    in_channels = data['train']['x'].shape[1]
    num_classes = cls_cfg.get("num_classes", 11) # 11 classes for minnie65
    print(f"Features: {in_channels}, Classes: {num_classes}")
    
    classifier_head = LinearClassifier(in_channels=in_channels, num_classes=num_classes)
    
    # We pass None for encoder_graph since we bypass the graph extraction in LightModule 
    # WAIT! The current ClassifierLightModule expects a graph batch and calls encoder_graph.
    # We need to adapt it, or create a subclass `EmbeddingClassifierLightModule`.
    
    # Let's define a simple LightModule for embeddings directly in this file
    class EmbeddingsLightModule(L.LightningModule):
        def __init__(self, classifier, lr, wd, max_epochs):
            super().__init__()
            self.classifier = classifier
            self.lr = lr
            self.wd = wd
            self.max_epochs = max_epochs
            self.loss_fn = torch.nn.CrossEntropyLoss()
            
            from torchmetrics import Accuracy, F1Score
            self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
            
            self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
            self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
            self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

        def forward(self, x):
            return self.classifier(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = self.loss_fn(logits, y)
            preds = torch.argmax(logits, dim=1)
            self.train_acc(preds, y)
            self.train_f1(preds, y)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)
            self.log("train_f1", self.train_f1, on_epoch=True, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = self.loss_fn(logits, y)
            preds = torch.argmax(logits, dim=1)
            self.val_acc(preds, y)
            self.val_f1(preds, y)
            self.log("val_loss", loss, prog_bar=True)
            self.log("val_acc", self.val_acc, prog_bar=True)
            self.log("val_f1", self.val_f1, prog_bar=True)

        def test_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            preds = torch.argmax(logits, dim=1)
            self.test_acc(preds, y)
            self.test_f1(preds, y)
            self.log("test_acc", self.test_acc)
            self.log("test_f1", self.test_f1)
            
        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

    module = EmbeddingsLightModule(
        classifier_head, 
        lr=cls_cfg.get("learning_rate", 1e-3),
        wd=cls_cfg.get("weight_decay", 1e-5),
        max_epochs=cls_cfg.get("max_epochs", 100)
    )
    
    checkpoint_callback = L.callbacks.ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="emb-class-{epoch:02d}-{val_acc:.4f}",
    )

    trainer = L.Trainer(
        max_epochs=cls_cfg.get("max_epochs", 100),
        accelerator=cfg.trainer.get("accelerator", "gpu"),
        devices=cfg.trainer.get("devices", 1),
        logger=L.loggers.TensorBoardLogger(save_dir=cfg.get("log_dir", "lightning_logs"), name="embeddings_classifier"),
        callbacks=[checkpoint_callback],
        deterministic=True,
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    print("\nRunning evaluation on test set...")
    trainer.test(module, dataloaders=test_loader)

if __name__ == "__main__":
    main()
