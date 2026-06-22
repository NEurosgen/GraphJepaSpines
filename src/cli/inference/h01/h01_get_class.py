"""Геттеры меток для h01 (датасет proofread-нейронов человека).

Источник меток — `c3/tables/somas.csv` из релиза H01 (колонка `proofread_104_rep`
= neuron_id из имён файлов h01). Заранее собраны в два CSV:
  data/h01_celltype.csv  (neuron_id, cell_type)
  data/h01_layer.csv     (neuron_id, layer)

Две фабрики:
  make_h01_celltype_getter -> класс ТИПА КЛЕТКИ (бинарно exc/inh, как в minnie65)
  make_h01_layer_getter    -> класс СЛОЯ КОРЫ (Layer 1..6, White matter)

neuron_id берётся как первый числовой токен из имени файла
(`neuron_<neuron_id>_limb_..._branch_...`), что согласовано с извлечением
segment_id в GraphDataSet (пулинг по нейрону).
"""
import re
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch

# Тип клетки -> класс. exc=0 / inh=1 (как в minnie65). Глия (OLIGO/MG_OPC) -> -1 (игнор).
CELLTYPE_CLASS_MAP = {
    "PYRAMIDAL": 0,
    "SPINY_ATYPICAL": 0,
    "SPINY_STELLATE": 0,
    "INTERNEURON": 1,
}

# Слой коры -> класс.
LAYER_CLASS_MAP = {
    "Layer 1": 0,
    "Layer 2": 1,
    "Layer 3": 2,
    "Layer 4": 3,
    "Layer 5": 4,
    "Layer 6": 5,
    "White matter": 6,
}


def _neuron_id_from_path(file_path) -> Optional[str]:
    match = re.search(r"\d+", Path(file_path).name)
    return match.group(0) if match else None


def _make_getter(csv_path: Optional[str], value_col: str, class_map: dict) -> Callable:
    """Общий конструктор геттера: CSV (neuron_id, value_col) + маппинг value -> класс.

    Пустой csv_path -> геттер всегда возвращает -1 (объект игнорируется в probe).
    """
    if not csv_path:
        def get_unlabeled(file_path, out=None, **kwargs) -> torch.Tensor:
            return torch.tensor(-1, dtype=torch.long)
        return get_unlabeled

    df = pd.read_csv(csv_path, dtype={"neuron_id": str}).dropna(subset=["neuron_id", value_col])
    mapping = {str(row["neuron_id"]).strip(): row[value_col] for _, row in df.iterrows()}

    def get_class(file_path, out=None, **kwargs) -> torch.Tensor:
        nid = _neuron_id_from_path(file_path)
        value = mapping.get(nid) if nid is not None else None
        if value is None or value not in class_map:
            return torch.tensor(-1, dtype=torch.long)
        return torch.tensor(class_map[value], dtype=torch.long)

    return get_class


def make_h01_celltype_getter(csv_path: Optional[str], class_map: Optional[dict] = None) -> Callable:
    """neuron_id -> класс типа клетки (по умолчанию бинарно exc=0 / inh=1)."""
    return _make_getter(csv_path, "cell_type", class_map if class_map is not None else CELLTYPE_CLASS_MAP)


def make_h01_layer_getter(csv_path: Optional[str], class_map: Optional[dict] = None) -> Callable:
    """neuron_id -> класс слоя коры (Layer 1..6=0..5, White matter=6)."""
    return _make_getter(csv_path, "layer", class_map if class_map is not None else LAYER_CLASS_MAP)
