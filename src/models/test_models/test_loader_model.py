import pytest
import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.models.loader_model import load_encoder_from_folder, load_classifier
from src.models.classificator import ClassifierLightModule


def test_load_encoder_from_folder():
    """
    Test loading the encoder from an existing folder path.
    The task specifically requests to use 'lightning_logs/jepa_r_1.5/version_0' as an example.
    """
    # Assuming tests are run from the project root or similar. 
    # Adjusting path to be absolute or relative to the notebook root.
    # The current root of the workspace is src/models, but tests usually run from notebook root.
    # We will try to find the path dynamically or use the fixed relative path.
    
    # This path is relative to the directory where pytest is invoked.
    # For robust testing, we can check if it exists or use an absolute path based on __file__.
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    folder_path = os.path.join(base_dir, "lightning_logs", "jepa_r_1.5", "version_0")
    
    if not os.path.exists(folder_path):
        pytest.skip(f"Test path {folder_path} not found. Ensure you have the checkpoint available.")
        
    encoder = load_encoder_from_folder(folder_path)
    
    assert encoder is not None
    assert isinstance(encoder, nn.Module)
    # Just a simple check to ensure parameters are loaded
    assert len(list(encoder.parameters())) > 0


def test_save_and_load_classifier(tmp_path):
    """
    Test saving a dummy classifier checkpoint and loading it via load_classifier.
    """
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "test_classifier.ckpt"
    
    in_channels = 32
    num_classes = 3
    
    # Create a state dict matching what load_classifier expects
    state_dict = {
        "classifier.head.0.weight": torch.randn(num_classes, in_channels),
        "classifier.head.0.bias": torch.randn(num_classes),
    }
    
    # We need a dummy cfg because ClassifierLightModule.__init__ expects it
    cfg = OmegaConf.create({
        "optimizer": {"_target_": "torch.optim.Adam"}
    })
    
    ckpt = {
        "state_dict": state_dict,
        "hyper_parameters": {
            "cfg": cfg,
            "learning_rate": 1e-3
        },
        "pytorch-lightning_version": "2.0.0"
    }
    
    torch.save(ckpt, checkpoint_path)
    
    # Load using the function from loader_model
    loaded_model = load_classifier(str(tmp_path))
    
    assert loaded_model is not None
    assert isinstance(loaded_model, ClassifierLightModule)
    assert loaded_model.classifier.head[0].weight.shape == (num_classes, in_channels)
    assert loaded_model.classifier.head[0].bias.shape == (num_classes,)
    
    # Verify weights were correctly loaded
    assert torch.allclose(loaded_model.classifier.head[0].weight.data, state_dict["classifier.head.0.weight"])
    assert torch.allclose(loaded_model.classifier.head[0].bias.data, state_dict["classifier.head.0.bias"])
