# train_model.py
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as L
import torch
from hydra.utils import instantiate
from ...data_utils.datamodule import GraphDataModule, GraphDataSet
from ...data_utils.transforms import (
    GenNormalize,
    create_mask_collate_fn,
    MaskData,
    build_canonical_transform,
    build_augments,
)
from ...data_utils.stats import load_feature_stats
from pathlib import Path
from ...models.jepa import JepaLight


torch.set_float32_matmul_precision('high')


def get_datamodule(cfg):
    """Работает напрямую с исходным датасетом — без отдельного шага записи на диск.

    Канонизация (нормализация + единообразное построение графа из pos) считается
    на лету в GraphDataSet и кэшируется в RAM (save_cache). Маскирование — на
    каждом шаге в collate_fn.
    """
    path = Path(cfg.dataset.path)

    mean_x, std_x = load_feature_stats(cfg.dataset.stats_path)
    canonical_transform = GenNormalize(
        transforms=build_canonical_transform(cfg, mean_x, std_x),
        mask_transform=None,
    )

    mask_transform = MaskData(mask_ratio=cfg.mask_ratio)
    dyn_transform = GenNormalize(transforms=[], mask_transform=mask_transform)
    collate_fn = create_mask_collate_fn(
        dyn_transform,
        num_views=cfg.get('num_views', 1),
        augments=build_augments(cfg),
    )

    ds = GraphDataSet(
        path=path,
        transform=canonical_transform,
        save_cache=cfg.dataset['save_cache']
    )
    datamodule = GraphDataModule(
        ds,
        cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        ratio=cfg.ratio,
        collate_fn=collate_fn
    )
    return datamodule


def run_training(cfg: DictConfig, checkpoint_dir: Path, name: str = None) -> None:
    """Запускает одно полное обучение модели.
    """
    L.seed_everything(cfg.seed, workers=True)

    model = instantiate(cfg.network, _recursive_=True)
    model_module = JepaLight(cfg=cfg, model=model, debug=False)

    checkpoint_dir = Path(checkpoint_dir)

    if name:
        logger = L.loggers.TensorBoardLogger(save_dir=str(checkpoint_dir), name=name)
        target_dir = Path(logger.log_dir)
    else:
        logger = L.loggers.TensorBoardLogger(save_dir=str(checkpoint_dir), name="", version="")
        target_dir = checkpoint_dir

    target_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = L.callbacks.ModelCheckpoint(
        dirpath=str(target_dir / "checkpoints"),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="jepa-{epoch:02d}-{val_loss:.4f}"
    )

    trainer = L.Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=[checkpoint_callback],
        deterministic=True
    )
    
    datamodule = get_datamodule(cfg.datamodule)
    trainer.fit(model_module, datamodule=datamodule)


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    output_dir ="lightning_logs"
    run_training(cfg=cfg, checkpoint_dir=output_dir, name = "h01_train")


if __name__ == "__main__":
    main()