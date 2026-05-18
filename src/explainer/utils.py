import os
import glob
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from omegaconf import DictConfig

from src.data_utils.datamodule import GraphDataSet
from src.models.loader_model import load_encoder_from_folder
from src.models.classificator import ClassifierLightModule
from src.data_utils.transforms import GenNormalize , load_stats, build_transforms
from src.data_utils.stats import compute_macro_stats

def _simple_collate(data_list):
    """Collate for classification - no masking, just batch graphs."""
    return Batch.from_data_list(data_list)

def _find_latest_checkpoint(path):
    """Finds the latest .ckpt file in a directory or returns the path if it's already a file."""
    if os.path.isfile(path):
        return path
    
    ckpt_dir = os.path.join(path, "checkpoints")
    if not os.path.exists(ckpt_dir):
        ckpt_dir = path
        
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpt_files:
        return None
    
    return max(ckpt_files, key=os.path.getmtime)

def load_mesh_vertices(path):
    """
    Loads mesh vertices from a .pt file.
    Returns a numpy array (N, 3).
    """
    if path.endswith('.pt'):
        data = torch.load(path, weights_only=False)
    else:
        raise ValueError(f"Unsupported file format: {path}")
    
    if isinstance(data, dict) and 'pos' in data:
        return data['pos']  # numpy array
    
    if hasattr(data, 'pos'):
        return data.pos.numpy()  # torch tensor -> numpy
    
    raise ValueError(f"Could not find 'pos' in {path}")

def setup_explainer_environment(cfg: DictConfig, device=None):
    """
    Sets up the dataset, models (encoder and classifier), and computes macro stats.
    Returns a dictionary with all the essential objects.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    cls_cfg = cfg.classifier
    dm_cfg = cfg.datamodule
    
    path_to_classifier_dir = cls_cfg.get("classifier_checkpoint_path", None)
    path_to_lejepa_dir = cls_cfg.get("checkpoint_path", None)


    path_to_classifier = _find_latest_checkpoint(path_to_classifier_dir)
    path_to_lejepa = path_to_lejepa_dir

    if not path_to_classifier:
        raise FileNotFoundError(f"No checkpoint found in {path_to_classifier_dir}")

    mean_x, std_x, mean_edge, std_edge = load_stats(cls_cfg.stats_path)
    transforms = build_transforms(dm_cfg, mean_x, std_x, mean_edge, std_edge)
    gen_normalize = GenNormalize(transforms=transforms, mask_transform=None)

    folder_to_label = dict(cls_cfg.get("folder_to_label", {"ab": 0, "wt": 1}))
    


    ds = GraphDataSet(
        path=cls_cfg.path,
        get_class=get_class,
        transform=gen_normalize,
    )

    # Load classifier robustly handling LayerNorm changes
    ckpt = torch.load(path_to_classifier, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    
    num_classes = cls_cfg.get("num_classes")
    embed_dim = cfg.network.encoder.out_channels + 7 
    
    class FlexibleLinearClassifier(nn.Module):
        def __init__(self, state_dict):
            super().__init__()
            layers = []
            
            indices = set()
            for k in state_dict.keys():
                if k.startswith("classifier.head."):
                    parts = k.split(".")
                    if parts[2].isdigit():
                        indices.add(int(parts[2]))
            
            max_idx = max(indices) if indices else 0
            
            for i in range(max_idx + 1):
                w_key = f"classifier.head.{i}.weight"
                if w_key in state_dict:
                    shape = state_dict[w_key].shape
                    if len(shape) == 1:
                        layers.append(nn.LayerNorm(shape[0]))
                    elif len(shape) == 2:
                        layers.append(nn.Linear(shape[1], shape[0]))
                else:
                    layers.append(nn.ReLU())
            
            self.head = nn.Sequential(*layers)
            
        def forward(self, x):
            return self.head(x)
            
    classifier_head = FlexibleLinearClassifier(state_dict)
    
    classifier_module = ClassifierLightModule.load_from_checkpoint(
        path_to_classifier, 
        encoder_graph=nn.Identity(), 
        classifier=classifier_head,
        strict=False,
        weights_only=False
    )
    
    encoder = load_encoder_from_folder(path_to_lejepa)
    classifier_module.to(device)

    print("Computing dynamic macro statistics for dataset...")
    macro_mean, macro_std = compute_macro_stats(ds)
    
    return {
        "dataset": ds,
        "encoder": encoder,
        "classifier_module": classifier_module,
        "macro_mean": macro_mean,
        "macro_std": macro_std,
        "device": device,
        "std_x": std_x,
    }
