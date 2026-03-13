
import os
import glob
import torch
from ..models.jepa import JepaLight
from ..models.classificator import ClassifierLightModule
def load_encoder_from_folder(folder_path):

    checkpoint_dir = os.path.join(folder_path, "checkpoints")
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"Чекпоинты не найдены в {checkpoint_dir}")
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    hparams_path = os.path.join(folder_path, "hparams.yaml")

    model = JepaLight.load_from_checkpoint(
        checkpoint_path=latest_checkpoint,
        hparams_file=hparams_path
    )
    
    return model





def load_classifier(folder_path):
    checkpoint_dir = os.path.join(folder_path, "checkpoints")
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"Чекпоинты не найдены в {checkpoint_dir}")
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    hparams_path = os.path.join(folder_path, "hparams.yaml")

    model = ClassifierLightModule.load_from_checkpoint(
        checkpoint_path=latest_checkpoint,
        hparams_file=hparams_path
    )
    
    return model
