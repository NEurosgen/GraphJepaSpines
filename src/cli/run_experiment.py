"""
Эксперимент: зависимость val_loss от числа слоев в энкодере.

Описание:
- Все параметры фиксированы, кроме num_layers
- Число эпох: 10
- mask_ratio: 0.01 (очень маленькое значение для точного восстановления)
- num_layers: от 1 до 30
- Для каждого num_layers проводим 3 эксперимента с разными seed
- Записываем min_val_loss и val_loss на последней эпохе
- Сохраняем результаты в exp/enc_layers/
"""

import os
import json
from pathlib import Path
from datetime import datetime

import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import Callback

from omegaconf import OmegaConf

from ..models.jepa import GraphJepa, JepaLight, CrossAttentionPredictor
from ..models.encoder import GraphGcnEncoder
from ..data_utils.datamodule import GraphDataModule, GraphDataSet
from ..data_utils.transforms import MaskNorm, create_mask_collate_fn

torch.set_float32_matmul_precision('high')

# ===== ФИКСИРОВАННЫЕ ПАРАМЕТРЫ ЭКСПЕРИМЕНТА =====
FIXED_CONFIG = {
    # Encoder params (кроме num_layers)
    "in_channels": 21,
    "out_channels": 64,
    "alpha": 0.1,
    
    # Predictor params
    "predictor_hidden_dim": 64,
    "predictor_pos_dim": 3,
    "predictor_num_heads": 2,
    "predictor_dropout": 0.1,
    
    # Model params
    "ema": 0.996,
    
    # Training params
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "max_epochs": 10,
    
    # DataModule params
    "batch_size": 512,
    "num_workers": 6,
    "ratio": [0.7, 0.2, 0.1],
    
    # Mask ratio - очень маленькое для точного восстановления
    "mask_ratio": 0.01,
}

# Пути
STATS_PATH = "/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/data/stats/"
DATASET_PATH = "/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/notebooks/graph_dataset"
OUTPUT_DIR = "/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/exp/enc_layers"

# Параметры эксперимента
NUM_LAYERS_RANGE = range(1, 31)  # от 1 до 30 включительно
SEEDS = [42, 123, 456]  # 3 разных seed для стабильности


class ValLossTracker(Callback):
    """Callback для отслеживания val_loss на каждой эпохе."""
    
    def __init__(self):
        super().__init__()
        self.val_losses = []
        self.min_val_loss = float('inf')
    
    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            val_loss_value = val_loss.item()
            self.val_losses.append(val_loss_value)
            if val_loss_value < self.min_val_loss:
                self.min_val_loss = val_loss_value


def load_stats(path: str):
    """Загрузка статистик для нормализации."""
    mean_x = torch.load(path + "means.pt")
    std_x = torch.load(path + "stds.pt")
    mean_edge = torch.load(path + "mean_edge.pt")
    std_edge = torch.load(path + "std_edge.pt")
    return mean_x, std_x, mean_edge, std_edge


def create_model(num_layers: int) -> GraphJepa:
    """Создание модели с заданным числом слоев."""
    encoder = GraphGcnEncoder(
        in_channels=FIXED_CONFIG["in_channels"],
        out_channels=FIXED_CONFIG["out_channels"],
        alpha=FIXED_CONFIG["alpha"],
        num_layers=num_layers
    )
    
    predictor = CrossAttentionPredictor(
        hidden_dim=FIXED_CONFIG["predictor_hidden_dim"],
        pos_dim=FIXED_CONFIG["predictor_pos_dim"],
        num_heads=FIXED_CONFIG["predictor_num_heads"],
        dropout=FIXED_CONFIG["predictor_dropout"]
    )
    
    model = GraphJepa(
        encoder=encoder,
        predictor=predictor,
        ema=FIXED_CONFIG["ema"]
    )
    
    return model


def create_datamodule(seed: int) -> GraphDataModule:
    """Создание DataModule с заданным seed."""
    mean_x, std_x, mean_edge, std_edge = load_stats(STATS_PATH)
    
    # Transform с очень маленьким mask_ratio
    mask_norm = MaskNorm(
        mean_x, std_x, mean_edge, std_edge, 
        mask_ratio=FIXED_CONFIG["mask_ratio"]
    )
    collate_fn = create_mask_collate_fn(mask_norm)
    
    ds = GraphDataSet(path=DATASET_PATH, transform=None)
    
    datamodule = GraphDataModule(
        ds,
        batch_size=FIXED_CONFIG["batch_size"],
        num_workers=FIXED_CONFIG["num_workers"],
        seed=seed,
        ratio=FIXED_CONFIG["ratio"],
        collate_fn=collate_fn
    )
    
    return datamodule


def create_training_config():
    """Создание OmegaConf конфига для JepaLight."""
    cfg_dict = {
        "learning_rate": FIXED_CONFIG["learning_rate"],
        "weight_decay": FIXED_CONFIG["weight_decay"],
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "weight_decay": FIXED_CONFIG["weight_decay"]
        }
    }
    return OmegaConf.create(cfg_dict)


def run_single_experiment(num_layers: int, seed: int) -> dict:
    """
    Запуск одного эксперимента с заданными num_layers и seed.
    
    Returns:
        dict с результатами: min_val_loss, last_val_loss, all_val_losses
    """
    print(f"\n{'='*60}")
    print(f"Running: num_layers={num_layers}, seed={seed}")
    print(f"{'='*60}")
    
    # Установка seed для воспроизводимости
    L.seed_everything(seed, workers=True)
    
    # Создание модели
    model = create_model(num_layers)
    cfg = create_training_config()
    model_module = JepaLight(model=model, cfg=cfg , debug = False)
    
    # Callback для отслеживания val_loss
    val_tracker = ValLossTracker()
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=FIXED_CONFIG["max_epochs"],
        accelerator="gpu",
        devices=1,
        log_every_n_steps=10,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=False,
        callbacks=[val_tracker],
        logger=False,  # Отключаем логгер для чистоты вывода
    )
    
    # DataModule
    datamodule = create_datamodule(seed)
    
    # Обучение
    trainer.fit(model_module, datamodule=datamodule)
    
    # Результаты
    result = {
        "num_layers": num_layers,
        "seed": seed,
        "min_val_loss": val_tracker.min_val_loss,
        "last_val_loss": val_tracker.val_losses[-1] if val_tracker.val_losses else None,
        "all_val_losses": val_tracker.val_losses,
    }
    
    print(f"Results: min_val_loss={result['min_val_loss']:.6f}, last_val_loss={result['last_val_loss']:.6f}")
    
    return result


def save_results(results: list, output_dir: str):
    """Сохранение результатов в JSON файл."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_results_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Подготовка данных для сохранения
    output_data = {
        "config": FIXED_CONFIG,
        "experiment_params": {
            "num_layers_range": list(NUM_LAYERS_RANGE),
            "seeds": SEEDS,
        },
        "results": results,
        "timestamp": timestamp,
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {filepath}")
    
    # Также сохраняем сводную таблицу
    summary_filepath = os.path.join(output_dir, f"summary_{timestamp}.txt")
    with open(summary_filepath, "w", encoding="utf-8") as f:
        f.write("num_layers\tseed\tmin_val_loss\tlast_val_loss\n")
        for r in results:
            min_loss = f"{r['min_val_loss']:.6f}" if r['min_val_loss'] is not None else "ERROR"
            last_loss = f"{r['last_val_loss']:.6f}" if r['last_val_loss'] is not None else "ERROR"
            f.write(f"{r['num_layers']}\t{r['seed']}\t{min_loss}\t{last_loss}\n")
    
    print(f"Summary saved to: {summary_filepath}")
    
    return filepath


def main():
    """Основная функция запуска эксперимента."""
    print("="*60)
    print("ЭКСПЕРИМЕНТ: Зависимость val_loss от num_layers")
    print("="*60)
    print(f"\nФиксированные параметры:")
    for key, value in FIXED_CONFIG.items():
        print(f"  {key}: {value}")
    print(f"\nnum_layers: {list(NUM_LAYERS_RANGE)}")
    print(f"seeds: {SEEDS}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("="*60)
    
    all_results = []
    
    for num_layers in NUM_LAYERS_RANGE:
        for seed in SEEDS:
            try:
                result = run_single_experiment(num_layers, seed)
                all_results.append(result)
                
                # Сохраняем промежуточные результаты после каждого эксперимента
                save_results(all_results, OUTPUT_DIR)
                
            except Exception as e:
                print(f"\nERROR: num_layers={num_layers}, seed={seed}")
                print(f"Exception: {e}")
                all_results.append({
                    "num_layers": num_layers,
                    "seed": seed,
                    "error": str(e),
                    "min_val_loss": None,
                    "last_val_loss": None,
                    "all_val_losses": [],
                })
    
    # Финальное сохранение
    final_path = save_results(all_results, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("ЭКСПЕРИМЕНТ ЗАВЕРШЕН")
    print(f"Всего экспериментов: {len(all_results)}")
    print(f"Результаты сохранены в: {final_path}")
    print("="*60)


if __name__ == "__main__":
    main()