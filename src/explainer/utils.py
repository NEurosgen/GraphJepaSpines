import torch
from torch_geometric.data import Batch
from omegaconf import DictConfig

from src.data_utils.datamodule import GraphDataSet
from src.cli.train_model import load_stats, build_transforms, make_folder_class_getter
from src.models.loader_model import load_encoder_from_folder, load_classifier
from src.data_utils.transforms import GenNormalize
from src.data_utils.stats import compute_macro_stats

def _simple_collate(data_list):
    """Collate for classification - no masking, just batch graphs."""
    return Batch.from_data_list(data_list)

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

    if not path_to_classifier_dir or not path_to_lejepa_dir:
        print("Warning: Missing checkpoint paths in config. Falling back to default log locations...")
        path_to_classifier_dir = "/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/lightning_logs/classifier/version_64"
        path_to_lejepa_dir = "lightning_logs/jepa/version_32"

    mean_x, std_x, mean_edge, std_edge = load_stats(cls_cfg.stats_path)
    transforms = build_transforms(dm_cfg, mean_x, std_x, mean_edge, std_edge)
    gen_normalize = GenNormalize(transforms=transforms, mask_transform=None)

    folder_to_label = dict(cls_cfg.get("folder_to_label", {"ab": 0, "wt": 1}))
    
    def get_class(file_path, **kwargs):
        from pathlib import Path
        import torch
        folder_name = Path(file_path).parent.name.lower()
        mapping = {k.lower(): v for k, v in folder_to_label.items()}
        if folder_name in mapping:
            return torch.tensor(mapping[folder_name], dtype=torch.long)
        # fallback for explainer when folder doesn't match
        return torch.tensor(0, dtype=torch.long)

    ds = GraphDataSet(
        path=cls_cfg.path,
        get_class=get_class,
        transform=gen_normalize,
    )

    # Load classifier — all architecture variants handled by loader_model
    classifier_module = load_classifier(path_to_classifier_dir)
    
    encoder = load_encoder_from_folder(path_to_lejepa_dir)
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

