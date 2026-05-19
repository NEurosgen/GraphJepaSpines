
import re
from pathlib import Path
from typing import Callable
import pandas as pd
import torch
def make_minnie65_class_getter(csv_path: str) -> Callable:
    df = pd.read_csv(csv_path).dropna(subset=["segment_id", "cell_type"])
    mapping = {str(int(row["segment_id"])): row["cell_type"] for _, row in df.iterrows()}

    class_map = {
        "23P": 0, "4P": 0, "5P-IT": 0, "5P-NP": 0, "5P-PT": 0,
        "6P-CT": 0, "6P-IT": 0, "BC": 1, "BPC": 1, "MC": 1, "NGC": 1,
    }

    def get_class(file_path: Path, out=None, **kwargs) -> torch.Tensor:
        segment_id = None
        if out is not None and hasattr(out, "segment_id") and isinstance(out.segment_id, str):
            match = re.search(r"\d+", out.segment_id)
            if match:
                segment_id = match.group(0)

        if segment_id is None:
            match = re.search(r"\d+", Path(file_path).name)
            if not match:
                raise ValueError(f"Could not find segment_id in filename: {file_path}")
            segment_id = match.group(0)

        cell_type = mapping.get(segment_id)
        if cell_type is None or cell_type not in class_map:
            return torch.tensor(-1, dtype=torch.long)

        return torch.tensor(class_map[cell_type], dtype=torch.long)

    return get_class


