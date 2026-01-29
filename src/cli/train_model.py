import hydra
from omegaconf import DictConfig
import pytorch_lightning as L
from hydra.utils import instantiate
from ..models.jepa import JepaLight
from ..data_utils.datamodule import GraphDataModule, GraphDataSet
from ..data_utils.transforms import MaskNorm, GenNormalize, create_mask_collate_fn
import torch
torch.set_float32_matmul_precision('high')
def load_stats(path):
    mean_x = torch.load(path+"means.pt")
    std_x = torch.load(path+"stds.pt")
    mean_edge = torch.load(path+"mean_edge.pt")
    std_edge = torch.load(path+"std_edge.pt")
    return mean_x,std_x, mean_edge,std_edge

def get_datamodule(cfg):
    mean_x, std_x, mean_edge, std_edge = load_stats('/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/data/stats/')
    
    # Create transform for collate_fn (parallel masking in workers)
    mask_norm = MaskNorm(k=cfg.knn,mean_x = mean_x, std_x = std_x, mean_edge=mean_edge, std_edge=std_edge, mask_ratio=cfg.mask_ratio) # mask ratio тоже в конфиг надо вынести
    collate_fn = create_mask_collate_fn(mask_norm)
    
    # Dataset without transform - masking happens in collate_fn
    ds = GraphDataSet(path=cfg.dataset.path, transform=None)

    datamodule = GraphDataModule(ds, cfg.batch_size,
                                 num_workers=cfg.num_workers, 
                                 seed=cfg.seed,
                                 ratio=cfg.ratio,
                                 collate_fn=collate_fn)
    return datamodule


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)
    model = instantiate(cfg.network, _recursive_=True)
    model_module = JepaLight(model=model, cfg=cfg.training, debug = False)

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