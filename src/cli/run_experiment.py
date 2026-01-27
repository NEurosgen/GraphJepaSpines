import os
import json
import copy
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import Callback
from omegaconf import OmegaConf

# Импорты ваших модулей (оставляем как есть)
from ..models.jepa import GraphJepa, JepaLight, CrossAttentionPredictor
from ..models.encoder import GraphGcnEncoder
from ..data_utils.datamodule import GraphDataModule, GraphDataSet
from ..data_utils.transforms import MaskNorm, create_mask_collate_fn

torch.set_float32_matmul_precision('high')

# ==========================================
# 1. КОНФИГУРАЦИЯ И ПУТИ
# ==========================================

# Базовая конфигурация (по умолчанию)
BASE_CONFIG = {
    "model": {
        "in_channels": 21,
        "out_channels": 64,
        "alpha": 0.1,
        "num_layers": 6, 
        "ema": 0.996,
    },
    "predictor": {
        "hidden_dim": 64,
        "pos_dim": 3,
        "num_heads": 2,
        "dropout": 0.1,
    },
    "training": {
        "learning_rate": 1e-2,
        "weight_decay": 1e-5,
        "max_epochs": 10,
    },
    "data": {
        "batch_size": 512,
        "num_workers": 6,
        "ratio": [0.7, 0.2, 0.1],
        "mask_ratio": 0.01,
    }
}

PATHS = {
    "stats": "/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/data/stats/",
    "dataset": "/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/notebooks/graph_dataset",
    "output": "/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/exp/alpha_enc"
}


# ==========================================
# 2. УТИЛИТЫ (Логгер и Хелперы)
# ==========================================

class ExperimentLogger:
    """
    Отвечает за сохранение результатов в ОДИН файл.
    Файл создается при инициализации, а метод save обновляет его.
    """
    def __init__(self, output_dir: str, experiment_name: str, base_config: dict, param_name: str):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Формируем имя файла один раз
        self.filepath = os.path.join(output_dir, f"{experiment_name}_{timestamp}.json")
        
        # Структура для сохранения
        self.data = {
            "meta": {
                "timestamp": timestamp,
                "variable_param": param_name,
                "base_config": base_config
            },
            "results": []
        }
        self._initial_save()

    def _initial_save(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        print(f"📁 Log file created: {self.filepath}")

    def log_result(self, result: dict):
        self.data["results"].append(result)
        # Перезаписываем файл с новыми данными
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        print(f"💾 Results updated in {self.filepath}")


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


def update_config_by_path(config: dict, path: str, value: Any):
    """
    Обновляет значение в словаре по пути через точку.
    Пример: update_config_by_path(cfg, 'model.num_layers', 10)
    """
    keys = path.split('.')
    current = config
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value


# ==========================================
# 3. ФАБРИКИ (Создание объектов)
# ==========================================

def load_stats(path: str):
    return (
        torch.load(path + "means.pt"),
        torch.load(path + "stds.pt"),
        torch.load(path + "mean_edge.pt"),
        torch.load(path + "std_edge.pt")
    )

def create_components(cfg: dict, seed: int):
    """Создает модель и DataModule на основе переданного конфига."""
    
    # 1. Model
    encoder = GraphGcnEncoder(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        alpha=cfg["model"]["alpha"],
        num_layers=cfg["model"]["num_layers"]
    )
    
    predictor = CrossAttentionPredictor(
        hidden_dim=cfg["predictor"]["hidden_dim"],
        pos_dim=cfg["predictor"]["pos_dim"],
        num_heads=cfg["predictor"]["num_heads"],
        dropout=cfg["predictor"]["dropout"]
    )
    
    model = GraphJepa(encoder=encoder, predictor=predictor, ema=cfg["model"]["ema"])
    
    # OmegaConf для Lightning модуля
    train_cfg = OmegaConf.create({
        "learning_rate": cfg["training"]["learning_rate"],
        "weight_decay": cfg["training"]["weight_decay"],
        "optimizer": {"_target_": "torch.optim.AdamW", "weight_decay": cfg["training"]["weight_decay"]}
    })
    
    lightning_module = JepaLight(model=model, cfg=train_cfg, debug=False)

    # 2. DataModule
    mean_x, std_x, mean_edge, std_edge = load_stats(PATHS["stats"])
    mask_norm = MaskNorm(mean_x, std_x, mean_edge, std_edge, mask_ratio=cfg["data"]["mask_ratio"])
    
    datamodule = GraphDataModule(
        GraphDataSet(path=PATHS["dataset"], transform=None),
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        seed=seed,
        ratio=cfg["data"]["ratio"],
        collate_fn=create_mask_collate_fn(mask_norm)
    )
    
    return lightning_module, datamodule


# ==========================================
# 4. ЯДРО ЭКСПЕРИМЕНТА
# ==========================================

def run_single_trial(config: dict, seed: int, param_info: dict) -> dict:
    """Запуск одного прогона (один seed, одно значение параметра)."""
    
    L.seed_everything(seed, workers=True)
    
    model_module, datamodule = create_components(config, seed)
    val_tracker = ValLossTracker()

    trainer = L.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator="gpu",
        devices=1,
        log_every_n_steps=10,
        deterministic=True,
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
    """
    Основной цикл эксперимента.
    param_to_vary: путь к параметру в конфиге (например 'model.num_layers')
    """
    print(f"\n🚀 STARTING EXPERIMENT: Varying '{param_to_vary}'")
    print(f"Values: {values}")
    print(f"Seeds: {seeds}\n")

    # Инициализируем логгер ОДИН раз
    logger = ExperimentLogger(
        output_dir=PATHS["output"], 
        experiment_name=f"exp_{param_to_vary.replace('.', '_')}", 
        base_config=BASE_CONFIG,
        param_name=param_to_vary
    )

    for val in values:
        for seed in seeds:
            try:
                # 1. Подготовка конфига для конкретной итерации
                current_config = copy.deepcopy(BASE_CONFIG)
                update_config_by_path(current_config, param_to_vary, val)
                
                print(f"👉 Running: {param_to_vary}={val}, seed={seed}")

                # 2. Запуск
                result = run_single_trial(
                    current_config, 
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

# ==========================================
# 5. ТОЧКА ВХОДА
# ==========================================

if __name__ == "__main__":

    PARAM_NAME = "model.alpha"
    
    PARAM_VALUES = [i/100 for i in range(0,101,5)]
    

    SEEDS = [42, 123, 456]

    run_experiment_grid(PARAM_NAME, PARAM_VALUES, SEEDS)