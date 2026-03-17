import torch
from torch import nn
import pytorch_lightning as L
import numpy as np
from torch_geometric.nn import global_add_pool
from torch_geometric.data import Batch
from sklearn.metrics import f1_score as sklearn_f1_score
from pathlib import Path
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from ..models.loader_model import load_encoder_from_folder
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
from src.models.classificator import ClassifierLightModule,LinearClassifier
from src.models.loader_model import load_encoder_from_folder
from src.models.encoder import GraphLatent

def compute_macro_stats(dataset, max_samples=2000):
    """Computes mean and std of macro_metrics dynamically over the dataset."""
    import random
    all_macros = []
    indices = list(range(len(dataset)))
    if len(indices) > max_samples:
        indices = random.sample(indices, max_samples)
        
    for i in indices:
        data = dataset[i]
        if hasattr(data, 'macro_metrics') and data.macro_metrics is not None:
            mac = data.macro_metrics
            if mac.dim() == 2:
                mac = mac.mean(dim=0, keepdim=True)
            else:
                mac = mac.view(1, -1) #пачему
            all_macros.append(mac.cpu())
            
    if not all_macros:
        return None, None
        
    all_macros = torch.cat(all_macros, dim=0) # [N, 7]
    macro_mean = all_macros.mean(dim=0, keepdim=True)
    macro_std = all_macros.std(dim=0, keepdim=True)
    return macro_mean, macro_std


def get_class_9009(file_path):

    mapping = {"ab": 0, "wt" :1}

    folder_name = Path(file_path).parent.name.lower()
    if folder_name not in mapping:
        raise ValueError(
            f"Folder '{folder_name}' not in mapping {mapping}. "
            f"File: {file_path}"
        )
    return torch.tensor(mapping[file_path], dtype=torch.long)


def get_class_minnie_65(path):
    pass

# ─── Main ─────────────────────────────────────────────────────────────────

@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)

    cls_cfg = cfg.classifier


    encoder = load_encoder_from_folder(cls_cfg.checkpoint_path)
    encoder.eval()
    encoder.requires_grad_(False)
   
    num_classes = cls_cfg.get("num_classes", 2)
    embed_dim = cfg.network.encoder.out_channels + 7 # сделать более универсальным
    classifier_head = LinearClassifier(in_channels=embed_dim, num_classes=num_classes)


    ds = GraphDataSet(
        path=cls_cfg.path,
        get_class=get_class_9009,
        transform=gen_normalize,
    )
                

    
    print("Computing dynamic macro statistics for dataset...")
    macro_mean, macro_std = compute_macro_stats(ds)
    encoder_graph =  GraphLatent(encdoer=encoder,macro_mean=macro_mean,macro_std=macro_std,pooling=global_add_pool)
    module = ClassifierLightModule(
        cfg=cls_cfg,
        encoder_graph = encoder_graph,
        learning_rate=cls_cfg.get("learning_rate", 1e-3),
        macro_mean=macro_mean,
        macro_std=macro_std
    )


    dm_cfg = cfg.datamodule
    mean_x, std_x, mean_edge, std_edge = load_stats(cls_cfg.stats_path)
    transforms = build_transforms(dm_cfg, mean_x, std_x, mean_edge, std_edge)
    gen_normalize = GenNormalize(transforms=transforms, mask_transform=None)

    datamodule = GraphDataModule(
        ds,
        batch_size=dm_cfg.batch_size,
        num_workers=dm_cfg.num_workers,
        seed=dm_cfg.seed,
        ratio=dm_cfg.ratio,
    )


    max_epochs = cls_cfg.get("max_epochs", 50)

    checkpoint_callback = L.callbacks.ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="classifier-{epoch:02d}-{val_acc:.4f}",
    )

    logger = L.loggers.TensorBoardLogger(save_dir=cfg.get("log_dir", "lightning_logs"), name="classifier")

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=cfg.trainer.get("accelerator", "gpu"),
        devices=cfg.trainer.get("devices", 1),
        log_every_n_steps=cfg.trainer.get("log_every_n_steps", 10),
        logger=logger,
        callbacks=[checkpoint_callback],
        deterministic=True,
    )

    trainer.fit(module, datamodule=datamodule)


    print("\nRunning evaluation on test set...")
    trainer.test(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
