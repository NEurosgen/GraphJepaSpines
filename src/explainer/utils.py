import os
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from omegaconf import DictConfig

from src.data_utils.datamodule import GraphDataSet
from src.models.loader_model import load_encoder_from_folder
from src.data_utils.transforms import GenNormalize
from src.data_utils.stats import compute_macro_stats
from src.cli.embedding_pipeline import LinearClassifier

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

def load_linear_classifier(path: str) -> nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = LinearClassifier(ckpt["in_channels"], ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    return model


def setup_explainer_environment(cfg: DictConfig, get_class, device=None):
    """
    Sets up the dataset, models (encoder and classifier), and computes macro stats.
    Returns a dictionary with all the essential objects.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cls_cfg = cfg.classifier

    path_to_classifier = cls_cfg.get("classifier_checkpoint_path", None)
    if not path_to_classifier or not os.path.isfile(path_to_classifier):
        raise FileNotFoundError(f"Classifier checkpoint not found: {path_to_classifier}")

    path_to_encoder = cls_cfg.get("checkpoint_path", None)

    gen_normalize = GenNormalize(transforms=[], mask_transform=None)
    ds = GraphDataSet(
        path=cls_cfg.path,
        get_class=get_class,
        transform=gen_normalize,
    )

    classifier = load_linear_classifier(path_to_classifier)
    classifier.to(device)

    encoder = load_encoder_from_folder(path_to_encoder)

    print("Computing dynamic macro statistics for dataset...")
    macro_mean, macro_std = compute_macro_stats(ds)

    return {
        "dataset": ds,
        "encoder": encoder,
        "classifier": classifier,
        "macro_mean": macro_mean,
        "macro_std": macro_std,
        "device": device,
    }
