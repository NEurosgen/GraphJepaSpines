# train_model.py
import hydra

from omegaconf import DictConfig
import pytorch_lightning as L
import torch
from hydra.utils import instantiate
from ..data_utils.datamodule import GraphDataModule, GraphDataSet

from ..data_utils.transforms import (
    GenNormalize, 
    create_mask_collate_fn,
    MaskData,
    build_transforms
    
)
from ..data_utils.stats import load_stats
from ..models.jepa import JepaLight


torch.set_float32_matmul_precision('high')


def get_datamodule(cfg):
    mean_x, std_x, mean_edge, std_edge = load_stats(cfg.dataset.stats_path)
    
    transforms = build_transforms(cfg, mean_x, std_x, mean_edge, std_edge)
    
    static_transform = GenNormalize(transforms=transforms, mask_transform=None)
    mask_transform = MaskData(mask_ratio=cfg.mask_ratio)
    dyn_transform = GenNormalize(transforms=[], mask_transform=mask_transform)
    
    collate_fn = create_mask_collate_fn(dyn_transform)
    save_cache = cfg.dataset.get('save_cache', True)
    
    ds = GraphDataSet(path=cfg.dataset.path, transform=static_transform, save_cache=save_cache)
    datamodule = GraphDataModule(
        ds, 
        cfg.batch_size,
        num_workers=cfg.num_workers, 
        seed=cfg.seed,
        ratio=cfg.ratio,
        collate_fn=collate_fn
    )
    return datamodule



@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)
    model = instantiate(cfg.network, _recursive_=True)
    # Для созраняения чек поинтов стоит доабвить все таки созранение только последонего и лучше по val loss а также соранения всех гиперпараметрво по которым создаввалсь модель а также коммит который был на данный момент
    model_module = JepaLight(cfg=cfg, model=model, debug=False)
    checkpoint_callback = L.callbacks.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        filename="jepa-{epoch:02d}-{val_loss:.4f}"
    )

    logger = L.loggers.TensorBoardLogger(save_dir=cfg.get("log_dir", "lightning_logs"), name="jepa")

    trainer = L.Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=[checkpoint_callback],
        deterministic=True
    )

    datamodule = get_datamodule(cfg.datamodule)
    trainer.fit(model_module, datamodule=datamodule)


if __name__ == "__main__":
    main()