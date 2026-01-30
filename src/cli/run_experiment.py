import os
import json
import copy
import hydra  # <--- ВАЖНО
from pathlib import Path
from datetime import datetime
from typing import Any

import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import Callback
from omegaconf import OmegaConf

# Ваши импорты
from src.models.jepa import JepaLight # Сам GraphJepa создаст Hydra
from src.data_utils.transforms import MaskNorm, create_mask_collate_fn
from src.data_utils.datamodule import GraphDataSet,GraphDataModule
torch.set_float32_matmul_precision('high')



# ==========================================
# 1. УТИЛИТЫ КОНФИГУРАЦИИ
# ==========================================

def load_config(path: str):
    """Загружает YAML в объект OmegaConf."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    conf = OmegaConf.load(path)
    return conf 

def update_config_by_path(config, path: str, value: Any):
    """
    Обновляет OmegaConf объект по пути.
    OmegaConf.update(config, path, value) - более надежный способ
    """
    OmegaConf.update(config, path, value)



def load_stats(path: str):
    return (
        torch.load(path + "means.pt"),
        torch.load(path + "stds.pt"),
        torch.load(path + "mean_edge.pt"),
        torch.load(path + "std_edge.pt")
    )

def get_datamodule(cfg):
    mean_x, std_x, mean_edge, std_edge = load_stats('/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/data/stats/')
    
    mask_norm = MaskNorm(k = cfg.knn ,mean_x = mean_x, std_x = std_x, mean_edge=mean_edge, std_edge=std_edge, mask_ratio=cfg.mask_ratio) # mask ratio тоже в конфиг надо вынести
    collate_fn = create_mask_collate_fn(mask_norm)
    
    ds = GraphDataSet(path=cfg.dataset.path, transform=None)

    datamodule = GraphDataModule(ds, cfg.batch_size,
                                 num_workers=cfg.num_workers, 
                                 seed=cfg.seed,
                                 ratio=cfg.ratio,
                                 collate_fn=collate_fn)
    return datamodule

def create_components(cfg: OmegaConf, seed: int):
    """
    Создает объекты, используя Hydra Instantiation.
    """

    model = hydra.utils.instantiate(cfg.network)
    lightning_module = JepaLight(model=model, cfg=cfg.training, debug=False)
    

    datamodule = get_datamodule(cfg.datamodule)
    
    return lightning_module, datamodule


# ==========================================
# 3. ЭКСПЕРИМЕНТ (Без изменений логики)
# ==========================================

class ExperimentLogger:
    def __init__(self, output_dir: str, experiment_name: str, base_config: OmegaConf, param_name: str):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(output_dir, f"{experiment_name}_{timestamp}.json")
        
        config_dict = OmegaConf.to_container(base_config, resolve=True)
        
        self.data = {
            "meta": { "timestamp": timestamp, "variable_param": param_name, "base_config": config_dict },
            "results": []
        }
        self._initial_save()

    def _initial_save(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def log_result(self, result: dict):
        self.data["results"].append(result)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

class ValLossTracker(Callback):
    """Callback для отслеживания val_loss."""
    def __init__(self):
        super().__init__()
        self.val_losses = []
        self.min_val_loss = float('inf')

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            val = val_loss.item()
            self.val_losses.append(val)
            if val < self.min_val_loss:
                self.min_val_loss = val

def run_single_trial(config, seed, param_info):
    L.seed_everything(seed, workers=True)
    
    model_module, datamodule = create_components(config, seed)
    val_tracker = ValLossTracker()
    trainer = L.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=False,
        callbacks=[val_tracker],
        logger=False
    )

    trainer.fit(model_module, datamodule=datamodule)
    

    return {
        "param_value": param_info["value"],
        "param_name": param_info["name"],
        "seed": seed,
        "min_val_loss": val_tracker.min_val_loss,
        "last_val_loss": val_tracker.val_losses[-1] if val_tracker.val_losses else None,
        # "all_losses": val_tracker.val_losses  # Можно раскомментировать, если нужно
    }



def run_experiment_grid(param_to_vary: str, values: list, seeds: list):
    # 1. Загружаем Базовый Конфиг
    base_cfg = load_config("configs/config.yaml")

    print(f"\n🚀 STARTING HYDRA EXPERIMENT: Varying '{param_to_vary}'")

    logger = ExperimentLogger(
        output_dir=PATHS["output"], 
        experiment_name=f"exp_{param_to_vary.replace('.', '_')}", 
        base_config=base_cfg,
        param_name=param_to_vary
    )

    for val in values:
        for seed in seeds:
            current_cfg = base_cfg.copy()
            
            update_config_by_path(current_cfg, param_to_vary, val)
            
            print(f"👉 Running: {param_to_vary}={val}, seed={seed}")

            try:
                result = run_single_trial(
                    current_cfg, 
                    seed, 
                    param_info={"name": param_to_vary, "value": val}
                )

                # 3. Логирование (дописывает в тот же файл)
                logger.log_result(result)
                print(f"   ✅ Min Loss: {result['min_val_loss']:.6f}")

            except Exception as e:
                print(f"   ❌ Error: {e}")
                logger.log_result({
                    "param_value": val,
                    "seed": seed,
                    "error": str(e)
                })

if __name__ == "__main__":

    
    PARAM_NAME = 'datamodule.batch_size'
    PARAM_VALUES = [i for i in range(16, 32,4)]
    SEEDS = [42,51,113]
    PATHS = {
    "stats": "/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/data/stats/",
    "output": '/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/exp/batch_loss'
    }
    run_experiment_grid(PARAM_NAME, PARAM_VALUES, SEEDS)