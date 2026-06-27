import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

SPINE_RE = re.compile(r"^(?P<prefix>.+)_spine_(?P<spine_id>\d+)$")


def group_spine_files(src_dir: Path):
    groups = defaultdict(dict)
    for path in src_dir.glob("*.npy"):
        stem = path.stem
        is_centroid = stem.endswith("_centroid")
        if is_centroid:
            stem = stem[: -len("_centroid")]

        match = SPINE_RE.match(stem)
        if match is None:
            continue

        prefix = match.group("prefix")
        spine_id = int(match.group("spine_id"))
        key = "centroid" if is_centroid else "feature"
        groups[prefix].setdefault(spine_id, {})[key] = path

    return groups


def build_graph(spines: dict):
    valid_ids = sorted(
        spine_id
        for spine_id, files in spines.items()
        if "feature" in files and "centroid" in files
    )
    if not valid_ids:
        return None

    x = np.stack([np.load(spines[i]["feature"]) for i in valid_ids])
    pos = np.stack([np.load(spines[i]["centroid"]) for i in valid_ids])

    x_tensor = torch.from_numpy(x).float()
    pos_tensor = torch.from_numpy(pos).float()

    num_nodes = pos_tensor.size(0)

    # Генерация индексов для полносвязного графа
    row, col = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing="ij")
    row = row.flatten()
    col = col.flatten()

    # Удаление self-loops (петель)
    mask = row != col
    row = row[mask]
    col = col[mask]
    
    edge_index = torch.stack([row, col], dim=0)

    # Вычисление евклидова расстояния между узлами для edge_attr
    if edge_index.numel() > 0:
        distances = torch.norm(pos_tensor[row] - pos_tensor[col], p=2, dim=1)
        edge_attr = distances.view(-1, 1)  # Формат [num_edges, 1] стандартен для PyG
    else:
        # Случай, если в графе только 1 узел (ребер нет)
        edge_attr = torch.empty((0, 1), dtype=torch.float)

    return Data(
        x=x_tensor,
        pos=pos_tensor,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )


def main():
    src_dir = Path(__file__).resolve().parents[2] / "datasets" / "minnie65_output_oriented_descriptors_subset"
    dst_dir = Path(__file__).resolve().parents[2] / "datasets" / "minnie65_oriented_dendrite_graph"
    dst_dir.mkdir(parents=True, exist_ok=True)

    groups = group_spine_files(src_dir)

    skipped = 0
    for prefix, spines in tqdm(groups.items(), desc="Building branch graphs"):
        graph = build_graph(spines)
        if graph is None:
            skipped += 1
            continue
        torch.save(graph, dst_dir / f"{prefix}.pt")

    print(f"Built {len(groups) - skipped} graphs, skipped {skipped} empty branches.")


if __name__ == "__main__":
    main()
