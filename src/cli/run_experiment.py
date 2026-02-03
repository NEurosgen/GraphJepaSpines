import os
import json
import copy
import hydra  # <--- ВАЖНО
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import numpy as np
import pytorch_lightning as L
from pytorch_lightning.callbacks import Callback
from omegaconf import OmegaConf

# Ваши импорты
from src.models.jepa import JepaLight
from src.data_utils.transforms import (
    GenNormalize, 
    create_mask_collate_fn,
    NormNoEps,
    EdgeNorm,
    GraphPruning,
    MaskData,
    FeatureChoice
)
from src.data_utils.datamodule import GraphDataSet, GraphDataModule

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


def build_transforms(cfg, mean_x, std_x, mean_edge, std_edge):
    """
    Build transforms by config
    """
    transforms = []
    
    features = cfg.get('features', None)
    if features is not None:
        features = list(features)
        transforms.append(FeatureChoice(feature=features))
        mean_x = mean_x[features]
        std_x = std_x[features]
    
    transforms.append(NormNoEps(mean=mean_x, std=std_x, eps=cfg.get('eps', 1e-6)))
    transforms.append(EdgeNorm(mean=mean_edge, std=std_edge))
    
    knn_k = cfg.get('knn', -1)
    if knn_k > 0:
        transforms.append(GraphPruning(k=knn_k, mutual=cfg.get('mutual_knn', False)))
    
    return transforms


def get_datamodule(cfg):
    mean_x, std_x, mean_edge, std_edge = load_stats(cfg.dataset.stats_path)
    
    transforms = build_transforms(cfg, mean_x, std_x, mean_edge, std_edge)
    mask_transform = MaskData(mask_ratio=cfg.mask_ratio)
    gen_normalize = GenNormalize(transforms=transforms, mask_transform=mask_transform)
    
    collate_fn = create_mask_collate_fn(gen_normalize)
    ds = GraphDataSet(path=cfg.dataset.path, transform=None)

    datamodule = GraphDataModule(
        ds, 
        cfg.batch_size,
        num_workers=cfg.num_workers, 
        seed=cfg.seed,
        ratio=cfg.ratio,
        collate_fn=collate_fn
    )
    return datamodule


def create_repr_dataloader(repr_cfg):
    """
    Создает dataloader для оценки representation quality.
    
    Returns:
        Tuple[DataLoader, np.ndarray]: DataLoader и массив меток
    """
    from torch.utils.data import DataLoader, ConcatDataset
    from torch_geometric.data import Batch
    
    mean_x, std_x, mean_edge, std_edge = load_stats(repr_cfg.stats_path)
    transforms = build_transforms(repr_cfg, mean_x, std_x, mean_edge, std_edge)
    norm = GenNormalize(transforms=transforms, mask_transform=None)
    
    datasets = []
    labels = []
    
    for ds_cfg in repr_cfg.datasets:
        ds = GraphDataSet(path=ds_cfg.path, transform=norm)
        datasets.append(ds)
        labels.extend([ds_cfg.label] * len(ds))

    combined_dataset = ConcatDataset(datasets)
    labels_array = np.array(labels)
    
    def collate_fn(data_list):
        return Batch.from_data_list(data_list)
    
    dataloader = DataLoader(
        combined_dataset,
        batch_size=repr_cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    return dataloader, labels_array


def create_components(cfg: OmegaConf, seed: int):
    """
    Создает объекты, используя Hydra Instantiation.
    """
    model = hydra.utils.instantiate(cfg.network)
    
    # Добавляем поддержку repr_kwargs для эстиматоров
    repr_kwargs = {}
    if cfg.get('representation', {}).get('enabled', False):
        repr_cfg = cfg.representation
        repr_dl, repr_labels = create_repr_dataloader(repr_cfg)
        repr_kwargs = {
            'repr_dl': repr_dl,
            'repr_labels': repr_labels,
            'estimator_cfg': {'estimators': list(repr_cfg.estimators)}
        }
    
    lightning_module = JepaLight(model=model, cfg=cfg.training, debug=False, **repr_kwargs)
    datamodule = get_datamodule(cfg.datamodule)
    
    return lightning_module, datamodule


# ==========================================
# 3. ЭКСПЕРИМЕНТ
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


class MetricsTracker(Callback):
    """Callback для отслеживания произвольных метрик."""
    
    def __init__(self, metrics_to_track: List[str] = None):
        super().__init__()
        # По умолчанию отслеживаем только val_loss
        self.metrics_to_track = metrics_to_track or ['val_loss']
        self.history: Dict[str, List[float]] = {m: [] for m in self.metrics_to_track}
        # Метрики, которые нужно минимизировать (остальные максимизируются)
        self.minimize_metrics = {'val_loss', 'train_loss', 'alignment', 'davies_bouldin', 'intra_class_distance'}
        
    def on_validation_epoch_end(self, trainer, pl_module):
        for metric_name in self.metrics_to_track:
            value = trainer.callback_metrics.get(metric_name)
            # Проверяем с префиксом repr/ (метрики эстиматоров)
            if value is None:
                value = trainer.callback_metrics.get(f"repr/{metric_name}")
            if value is not None:
                val = value.item() if hasattr(value, 'item') else value
                self.history[metric_name].append(val)
    
    def get_best(self, metric_name: str) -> Optional[float]:
        """Возвращает лучшее значение метрики (min или max в зависимости от типа)."""
        values = self.history.get(metric_name, [])
        if not values:
            return None
        if metric_name in self.minimize_metrics:
            return min(values)
        return max(values)
    
    def get_last(self, metric_name: str) -> Optional[float]:
        """Возвращает последнее значение метрики."""
        values = self.history.get(metric_name, [])
        return values[-1] if values else None
    
    def get_all(self, metric_name: str) -> List[float]:
        """Возвращает всю историю метрики."""
        return self.history.get(metric_name, [])


def run_single_trial(config, seed, param_info, metrics_to_track: List[str] = None):
    """
    Запускает один эксперимент с заданными параметрами.
    
    Args:
        config: Конфигурация эксперимента
        seed: Random seed
        param_info: Информация о варьируемом параметре
        metrics_to_track: Список метрик для отслеживания (по умолчанию ['val_loss'])
    
    Returns:
        dict: Результаты эксперимента с best_* и last_* для каждой метрики
    """
    if metrics_to_track is None:
        metrics_to_track = ['val_loss']
    
    L.seed_everything(seed, workers=True)
    
    model_module, datamodule = create_components(config, seed)
    metrics_tracker = MetricsTracker(metrics_to_track=metrics_to_track)
    
    trainer = L.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=False,
        callbacks=[metrics_tracker],
        logger=False
    )

    trainer.fit(model_module, datamodule=datamodule)
    
    # Собираем результаты по всем метрикам
    result = {
        "param_value": param_info["value"],
        "param_name": param_info["name"],
        "seed": seed,
    }
    
    for metric in metrics_to_track:
        result[f"best_{metric}"] = metrics_tracker.get_best(metric)
        result[f"last_{metric}"] = metrics_tracker.get_last(metric)
    
    # Для обратной совместимости дублируем val_loss
    if 'val_loss' in metrics_to_track:
        result["min_val_loss"] = result.get("best_val_loss")
        result["last_val_loss"] = result.get("last_val_loss")
    
    return result


def run_experiment_grid(
    param_to_vary: str, 
    values: list, 
    seeds: list, 
    metrics_to_track: List[str] = None
):
    """
    Запускает grid search эксперимент.
    
    Args:
        param_to_vary: Путь к параметру для варьирования (dot-notation)
        values: Список значений параметра
        seeds: Список random seeds
        metrics_to_track: Список метрик для отслеживания
    """
    if metrics_to_track is None:
        metrics_to_track = ['val_loss']
    
    # 1. Загружаем Базовый Конфиг
    base_cfg = load_config("configs/config.yaml")

    print(f"\n🚀 STARTING HYDRA EXPERIMENT: Varying '{param_to_vary}'")
    print(f"📊 Tracking metrics: {metrics_to_track}")

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
                    param_info={"name": param_to_vary, "value": val},
                    metrics_to_track=metrics_to_track
                )

                # 3. Логирование (дописывает в тот же файл)
                logger.log_result(result)
                
                # Выводим основные результаты
                best_val = result.get('best_val_loss', result.get('min_val_loss'))
                if best_val is not None:
                    print(f"   ✅ Best val_loss: {best_val:.6f}")
                
                # Выводим дополнительные метрики
                for metric in metrics_to_track:
                    if metric != 'val_loss':
                        best = result.get(f'best_{metric}')
                        if best is not None:
                            print(f"   📈 Best {metric}: {best:.6f}")

            except Exception as e:
                print(f"   ❌ Error: {e}")
                logger.log_result({
                    "param_value": val,
                    "seed": seed,
                    "error": str(e)
                })


if __name__ == "__main__":
    
    PARAM_NAME = "network.encoder.num_layers"  # TODO: enter your value, e.g. "network.encoder.out_channels"
    PARAM_VALUES =  [1, 2 ,3 ]  # TODO: enter values, e.g. [32, 64, 128]
    SEEDS = [42, 51]
    
    # Метрики для отслеживания: val_loss по умолчанию  
    # Можно добавить: 'rank_me', 'isotropy', 'uniformity', 'silhouette' и др.
    METRICS_TO_TRACK = ['val_loss']
    
    PATHS = {
        "stats": "/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/data/stats/",
        "output": "/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/exp/results/"  # TODO: enter dir for save, e.g. "./exp_results/"
    }
    
    # Проверка, что все параметры заданы
    assert PARAM_NAME is not None, "Please set PARAM_NAME"
    assert PARAM_VALUES is not None, "Please set PARAM_VALUES"
    assert PATHS["output"] is not None, "Please set PATHS['output']"
    
    run_experiment_grid(PARAM_NAME, PARAM_VALUES, SEEDS, metrics_to_track=METRICS_TO_TRACK)