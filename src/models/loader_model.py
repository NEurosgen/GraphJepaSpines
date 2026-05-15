
import os
import glob
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from ..models.jepa import JepaLight
from ..models.classificator import ClassifierLightModule
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

class FlexibleClassifier(nn.Module):
    """
    Classifier head that dynamically reconstructs its architecture from a
    checkpoint state_dict. Supports both simple Linear-only heads and heads
    containing LayerNorm + Linear + ReLU combinations.
    
    Usage::

        classifier = FlexibleClassifier.from_state_dict(state_dict, prefix="classifier.head.")
    """

    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.head = nn.Sequential(*layers)

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        return self.head(embed)

    @classmethod
    def from_state_dict(cls, state_dict: dict, prefix: str = "classifier.head.") -> "FlexibleClassifier":
        """
        Infer the Sequential architecture from weight keys in *state_dict*.

        Handles three layer types based on weight tensor shape:
          - 2-D  ``(out, in)``  → ``nn.Linear``
          - 1-D  ``(dim,)``     → ``nn.LayerNorm``
          - key gap (no weight) → ``nn.ReLU``
        """
        indices: set[int] = set()
        for key in state_dict:
            if key.startswith(prefix):
                part = key[len(prefix):].split(".")[0]
                if part.isdigit():
                    indices.add(int(part))

        if not indices:
            raise ValueError(
                f"No classifier head layers found with prefix '{prefix}' in state_dict. "
                f"Available keys: {[k for k in state_dict if 'classifier' in k]}"
            )

        layers: list[nn.Module] = []
        for i in range(max(indices) + 1):
            w_key = f"{prefix}{i}.weight"
            if w_key in state_dict:
                shape = state_dict[w_key].shape
                if len(shape) == 1:
                    # LayerNorm stores weight as 1-D tensor of (normalized_shape,)
                    layers.append(nn.LayerNorm(shape[0]))
                elif len(shape) == 2:
                    layers.append(nn.Linear(shape[1], shape[0]))
                else:
                    raise ValueError(f"Unexpected weight shape {shape} for key {w_key}")
            else:
                # No weight at this index → non-parametric layer (ReLU, Dropout, etc.)
                layers.append(nn.ReLU())

        return cls(layers)


def load_classifier(folder_path):
    """
    Load a ``ClassifierLightModule`` checkpoint from *folder_path*.
    
    Automatically detects whether the classifier head is a simple Linear layer
    or a more complex Sequential (e.g. LayerNorm → Linear → ReLU → Linear)
    and reconstructs it accordingly.
    """
    latest_checkpoint = _get_latest_checkpoint(folder_path=folder_path)
    
    ckpt = torch.load(latest_checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    
    # Determine which key prefix the classifier uses
    if any(k.startswith("classifier.head.") for k in state_dict):
        classifier_head = FlexibleClassifier.from_state_dict(state_dict, prefix="classifier.head.")
    elif any(k.startswith("classifier.fd.") for k in state_dict):
        classifier_head = FlexibleClassifier.from_state_dict(state_dict, prefix="classifier.fd.")
    else:
        # Fallback: simple linear
        classifier_head = LinearClassifier(in_channels=64, num_classes=2)
        
    dummy_encoder = nn.Identity()
    
    model = ClassifierLightModule.load_from_checkpoint(
        checkpoint_path=latest_checkpoint,
        encoder_graph=dummy_encoder,
        classifier=classifier_head,
        strict=False,
        weights_only=False
    )
    
    return model

