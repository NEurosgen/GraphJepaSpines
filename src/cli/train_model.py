import hydra
from omegaconf import DictConfig
import pytorch_lightning as L
from hydra.utils import instantiate
from ..models.jepa import JepaLight
@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)
    model = instantiate(cfg.network, _recursive_=True)
    model_module = JepaLight(model=model, cfg=cfg.training)

    checkpoint_callback = L.callbacks.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        filename="jepa-{epoch:02d}-{val_loss:.4f}"
    )

    trainer = L.Trainer(
        **cfg.trainer,
        #callbacks=[checkpoint_callback]
    )

    datamodule = instantiate(cfg.datamodule, _recursive_=True)
    trainer.fit(model_module, datamodule=datamodule)

if __name__ == "__main__":
    main()