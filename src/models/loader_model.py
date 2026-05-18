
import os
import glob
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from ..models.jepa import JepaLight
import torch.nn as nn

class LinearClassifier(nn.Module):
    """Simple linear probe on top of frozen graph embeddings."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            # nn.LayerNorm(in_channels),
            # nn.Dropout(0.3),
            nn.Linear(in_channels, num_classes),

        )
    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        return self.head(embed)

def _get_latest_checkpoint(folder_path: str) -> str:
    checkpoint_dir = os.path.join(folder_path, "checkpoints")
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"Чекпоинты не найдены в {checkpoint_dir}")

    return max(checkpoint_files, key=os.path.getmtime)
def load_encoder_from_folder(folder_path):
    latest_checkpoint = _get_latest_checkpoint(folder_path=folder_path)
    hparams_path = os.path.join(folder_path, "hparams.yaml")

    cfg = OmegaConf.load(hparams_path)
    if "cfg" in cfg and "network" in cfg.cfg:
        model = instantiate(cfg.cfg.network, _recursive_=True)
    elif "network" in cfg:
        model = instantiate(cfg.network, _recursive_=True)
    else:
        raise ValueError(f"Could not find 'network' config in {hparams_path}")

    jepa_light = JepaLight.load_from_checkpoint(
        checkpoint_path=latest_checkpoint,
        model=model,
        strict=False,
        weights_only=False
    )
    
    jepa_model = jepa_light.model
    if hasattr(jepa_model, 'student_encoder'):
        return jepa_model.student_encoder
    elif hasattr(jepa_model, 'encoder'):
        return jepa_model.encoder
    else:
        return jepa_model
