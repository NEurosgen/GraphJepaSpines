import hydra
from omegaconf import DictConfig
import pytorch_lightning as L
from hydra.utils import instantiate
from ..models.jepa import JepaLight
@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    model = instantiate(cfg.network)
    model_module = JepaLight(model=model, cfg=cfg.training)

    checkpoint_callback = L.callbacks.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        filename="jepa-{epoch:02d}-{val_loss:.4f}"
    )


    trainer = L.Trainer(
        **cfg.trainer,
       # callbacks=[checkpoint_callback]
    )

    datamodule = instantiate(cfg.datamodule)
    trainer.fit(model_module, datamodule=datamodule)

if __name__ == "__main__":
    main()