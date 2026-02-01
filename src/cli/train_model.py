import hydra
from omegaconf import DictConfig
import pytorch_lightning as L
from hydra.utils import instantiate
from ..models.jepa import JepaLight
from ..data_utils.datamodule import GraphDataModule, GraphDataSet
from ..data_utils.transforms import MaskNorm, GenNormalize, create_mask_collate_fn
import torch
import numpy as np
torch.set_float32_matmul_precision('high')

def load_stats(path):
    mean_x = torch.load(path+"means.pt")
    std_x = torch.load(path+"stds.pt")
    mean_edge = torch.load(path+"mean_edge.pt")
    std_edge = torch.load(path+"std_edge.pt")
    return mean_x,std_x, mean_edge,std_edge

def load_stats_9009(path):
    mean_x = torch.load(path+"means_9009.pt")
    std_x = torch.load(path+"stds_9009.pt")
    mean_edge = torch.load(path+"means_edge_9009.pt")
    std_edge = torch.load(path+"stds_edge_9009.pt")
    return mean_x, std_x, mean_edge, std_edge

def get_datamodule(cfg):
    mean_x, std_x, mean_edge, std_edge = load_stats('/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/data/stats/')
    mask_norm = MaskNorm(k=cfg.knn,mean_x = mean_x, std_x = std_x, mean_edge=mean_edge, std_edge=std_edge, mask_ratio=cfg.mask_ratio) # mask ratio тоже в конфиг надо вынести
    collate_fn = create_mask_collate_fn(mask_norm)
    ds = GraphDataSet(path=cfg.dataset.path, transform=None)

    datamodule = GraphDataModule(ds, cfg.batch_size,
                                 num_workers=cfg.num_workers, 
                                 seed=cfg.seed,
                                 ratio=cfg.ratio,
                                 collate_fn=collate_fn)
    return datamodule


def create_repr_dataloader(repr_cfg):
    """
    Создаёт DataLoader для оценки качества представлений.
    Объединяет несколько датасетов с разными метками классов.
    Returns:
        Tuple[DataLoader, np.ndarray]: DataLoader и массив меток
    """
    from torch.utils.data import DataLoader, ConcatDataset
    from torch_geometric.data import Batch
    
    mean_x, std_x, mean_edge, std_edge = load_stats_9009(repr_cfg.stats_path)
    norm = GenNormalize(mean_x=mean_x, std_x=std_x, mean_edge=mean_edge, std_edge=std_edge)
    
    datasets = []
    labels = []
    
    for ds_cfg in repr_cfg.datasets:
        ds = GraphDataSet(path=ds_cfg.path, transform=norm)
        datasets.append(ds)
        labels.extend([ds_cfg.label] * len(ds))

    combined_dataset = ConcatDataset(datasets)
    labels_array = np.array(labels)
    
    def collate_fn(data_list):
        return Batch.from_data_list(data_list)
    
    dataloader = DataLoader(
        combined_dataset,
        batch_size=repr_cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    return dataloader, labels_array


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)
    model = instantiate(cfg.network, _recursive_=True)

    repr_kwargs = {}
    if cfg.get('representation', {}).get('enabled', False):
        repr_cfg = cfg.representation
        repr_dl, repr_labels = create_repr_dataloader(repr_cfg)
        repr_kwargs = {
            'repr_dl': repr_dl,
            'repr_labels': repr_labels,
            'estimator_cfg': {'estimators': list(repr_cfg.estimators)}
        }
    model_module = JepaLight(model=model, cfg=cfg.training, debug=False, **repr_kwargs)
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