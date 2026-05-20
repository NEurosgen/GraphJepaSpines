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
    preprocess_dataset,
)
from pathlib import Path
from ...models.jepa import JepaLight


torch.set_float32_matmul_precision('high')


def get_datamodule(cfg):
    input_path = Path(cfg.dataset.raw_path)
    output_path = Path(cfg.dataset.path)

    if not output_path.exists() or not any(output_path.rglob("*.pt")):
        print("Preprocessed dataset not found, running preprocessing...")
        preprocess_dataset(cfg, input_path, output_path)

    mask_transform = MaskData(mask_ratio=cfg.mask_ratio)
    dyn_transform = GenNormalize(transforms=[], mask_transform=mask_transform)
    collate_fn = create_mask_collate_fn(dyn_transform)

    ds = GraphDataSet(
        path=output_path,
        transform=None,
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
    run_training(cfg=cfg, checkpoint_dir=output_dir, name = "main_train")


if __name__ == "__main__":
    main()