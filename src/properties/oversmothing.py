import torch
from ..models.loader_model import load_encoder_from_folder
from  ..cli.train_model  import load_stats, build_transforms
from ..data_utils.stats import compute_macro_stats
from ..data_utils.datamodule import GraphDataModule, GraphDataSet, make_folder_class_getter, make_minnie65_class_getter
from ..data_utils.transforms import (
    GenNormalize,
    NormNoEps,
    EdgeNorm,
    GraphPruning,
    FeatureChoice,
)


import torch
import torch.nn.functional as F

def compute_dirichlet_energy(x, edge_index):
    """Вычисляет энергию Дирихле для графа."""
    src, dst = edge_index
    # Квадрат евклидова расстояния между связанными узлами
    energy = torch.norm(x[src] - x[dst], dim=1).pow(2).sum() / 2.0
    return energy.item()

def compute_mad(x):
    """Вычисляет Mean Average Distance (MAD).
    Внимание: требует O(N^2) памяти. Для графов >20k узлов может вызвать OOM.
    """
    N = x.size(0)
    # L2-нормализация векторов для косинусного сходства
    x_norm = F.normalize(x, p=2, dim=1)
    
    # Матрица косинусного сходства N x N
    cos_sim = torch.mm(x_norm, x_norm.t())
    
    # Расстояние = 1 - косинусное сходство
    dist = 1 - cos_sim
    dist.fill_diagonal_(0) # Исключаем самопетли (расстояние узла до самого себя)
    
    mad = dist.sum() / (N * (N - 1))
    return mad.item()

def compute_feature_variance(x):
    """Вычисляет среднюю дисперсию признаков по графу."""
    mean_x = x.mean(dim=0)
    variance = torch.norm(x - mean_x, dim=1).pow(2).mean()
    return variance.item()


encoder = load_encoder_from_folder('/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/lightning_logs/jepa_r_6/version_0')

encoder.eval()
encoder.requires_grad_(False)

num_classes = 2

dm_cfg = {}
mean_x, std_x, mean_edge, std_edge = load_stats('/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/data/stats_sph/')  # для minnie поменять путь в конфиге на обучный stat path
transforms = build_transforms(dm_cfg, mean_x, std_x, mean_edge, std_edge)
gen_normalize = GenNormalize(transforms=transforms, mask_transform=None)

ds = GraphDataSet(
    path="/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/datasets/dataset_prepared",
    transform=gen_normalize,
)

encoder.to('cpu')
print(encoder)

out = encoder(ds[0].x, ds[0].edge_index, ds[0].edge_attrs)


hidden_representations = []

def hook_fn(module, input, output):
    """Хук для сохранения тензора на выходе слоя."""

    out_tensor = output[0] if isinstance(output, tuple) else output
    hidden_representations.append(out_tensor.clone().detach())


for layer in encoder.layers:
    layer.register_forward_hook(hook_fn)


data = ds[0]
data.to('cpu')


hidden_representations.clear()

with torch.no_grad():
    x_proj = encoder.proj(data.x)
    hidden_representations.append(x_proj.clone())

out = encoder(data.x, data.edge_index, data.edge_attrs)


print(f"{'Слой':<8} | {'Dirichlet Energy':<18} | {'MAD':<10} | {'Variance':<10}")
print("-" * 54)

for layer_idx, h_x in enumerate(hidden_representations):

    
    energy = compute_dirichlet_energy(h_x, data.edge_index)
    mad = compute_mad(h_x)
    var = compute_feature_variance(h_x)
    
    layer_name = f"Proj (0)" if layer_idx == 0 else f"GIN ({layer_idx})"
    print(f"{layer_name:<8} | {energy:<18.4f} | {mad:<10.4f} | {var:<10.4f}")