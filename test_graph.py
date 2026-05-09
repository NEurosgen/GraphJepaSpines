import torch
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import hydra
from src.data_utils.datamodule import GraphDataSet, make_folder_class_getter

# We will just load a sample and see its keys
@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg):
    cls_cfg = cfg.classifier
    folder_to_label = dict(cls_cfg.get("folder_to_label", {"ab": 0, "wt": 1}))
    get_class = make_folder_class_getter(folder_to_label)
    ds = GraphDataSet(
        path=cls_cfg.path,
        get_class=get_class,
        transform=None,
    )
    data = ds[0]
    print(data.keys())
    if hasattr(data, 'pos') and data.pos is not None:
        print("pos shape:", data.pos.shape)
    else:
        print("No pos attribute")
        
if __name__ == "__main__":
    main()
