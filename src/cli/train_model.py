import hydra
from omegaconf import DictConfig
import pytorch_lightning as L
from hydra.utils import instantiate
from ..models.jepa import JepaLight
from ..data_utils.datamodule import GraphDataModule, GraphDataSet
from ..data_utils.normalization import GenNormalize
import torch

def load_stats(path):
    mean_x = torch.load(path+"means.pt")
    std_x = torch.load(path+"stds.pt")
    mean_edge = torch.load(path+"mean_edge.pt")
    std_edge = torch.load(path+"std_edge.pt")
    return mean_x,std_x, mean_edge,std_edge

def get_datamodule(cfg):
    mean_x,std_x, mean_edge,std_edge = load_stats('/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/data/stats/')
    transfrom = GenNormalize(mean_x,std_x, mean_edge,std_edge)


    ds = GraphDataSet(path = cfg.dataset.path,transform=transfrom)

    datamodule = GraphDataModule(ds, cfg.batch_size,
                                 num_workers=cfg.num_workers, 
                                 seed = cfg.seed,
                                 ratio = cfg.ratio,

                                 )
    return datamodule


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
        #callbacks=[checkpoint_callback],
        deterministic=True
    )

    datamodule = get_datamodule(cfg.datamodule)
    trainer.fit(model_module, datamodule=datamodule)

if __name__ == "__main__":
    main()