import torch.nn as nn
from torch_geometric.nn import radius_graph
from src.data_utils.structural_stats import ThesisMacroMetrics
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

def fast_normalization_by_features(data, eps=1e-6):
    """
    Считает среднее и std для каждого из n признака, 
    игнорируя значения |x| <= eps.
    """
    mask = data.abs() > eps
    means = torch.zeros(data.size(1))
    stds = torch.ones(data.size(1))
    
    for i in range(data.size(1)):
        col = data[:, i]
        col_mask = mask[:, i]
        
        if torch.any(col_mask):
            valid_data = col[col_mask]
            means[i] = valid_data.mean()
            stds[i] = valid_data.std() if valid_data.numel() > 1 else 1.0
        else:
            means[i] = 0.0
            stds[i] = 1.0
            
    return means, stds



def create_mask_collate_fn(transform: 'GenNormalize' = None, num_views: int = 1, augments=None):
    """Собирает батч JEPA-пар (context, target).

    На каждый базовый граф генерируется (len(augments) × num_views) представлений:
    для каждой аугментации из augments (например поворота на фиксированный угол) —
    отдельный объект (по КОПИИ графа), и маскирование (transform) считается
    НЕЗАВИСИМО для каждого представления.

    augments: список callable (aug(data) -> data) | None. None трактуется как одна
    «пустая» аугментация (идентичность). num_views=1 и augments=None воспроизводят
    прежнее поведение.
    """
    from torch_geometric.data import Batch

    aug_list = list(augments) if augments else [None]

    def mask_collate_fn(batch):
        if transform is None:
            return Batch.from_data_list(batch)

        contexts = []
        targets = []

        for data in batch:
            for aug in aug_list:
                for _ in range(num_views):
                    view = data if aug is None else aug(data.clone())
                    ctx, tgt = transform(view)
                    if ctx.num_nodes > 0 and tgt.num_nodes > 0:
                        contexts.append(ctx)
                        targets.append(tgt)

        if len(contexts) == 0:
            return None
        context_batch = Batch.from_data_list(contexts)
        target_batch = Batch.from_data_list(targets)

        return context_batch, target_batch

    return mask_collate_fn



        

class NormNoEps(torch.nn.Module):
    def __init__(self, mean : torch.Tensor , std : torch.Tensor , eps: float = 0.0):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        if torch.any(std.abs() < 1e-8):
            raise ValueError("Your std is too small. It's dangerous for division!")
        self.eps = eps
    def forward(self,data) -> torch.Tensor:
        mask = (data.x.abs() > self.eps)
        normalized_x  = (data.x - (self.mean))/(self.std)
        data.x = torch.where(mask,normalized_x,data.x)
        return data
    

class LocalPos(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,data):
        pos = data.pos
        pos_mean = torch.mean(pos,dim=0)
        pos_std = torch.std(pos,dim=0).clamp(min=1e-6)
        pos = (pos - pos_mean)/pos_std
        data.pos = pos
        return data





class BuildGeometricGraph(torch.nn.Module):
    '''
    Единообразное построение графа из позиций.

    Датасеты неоднородны: h01 приходит без рёбер (только x, pos), minnie65 —
    полносвязным, где-то есть предпосчитанные дистанции, где-то нет. Поэтому
    структура ВСЕГДА восстанавливается из data.pos: рёбра по радиусу r, а
    edge_attr = евклидовы дистанции между соединёнными узлами. Любые исходные
    edge_index / edge_attr перетираются.
    '''
    def __init__(self, r: float, loop: bool = False):
        super().__init__()
        self.r = r
        self.loop = loop

    def forward(self, data):
        edge_index = radius_graph(
            data.pos,
            r=self.r,
            batch=getattr(data, "batch", None),
            loop=self.loop,
            max_num_neighbors=data.num_nodes,
        )
        row, col = edge_index
        edge_attr = (data.pos[row] - data.pos[col]).norm(dim=-1, keepdim=True)
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        return data





class MaskData(torch.nn.Module):
    '''
    Waring! This class return TWO graphs , contex and target (In JEPA notations)

    '''
    def __init__(self, mask_ratio: float):
        super().__init__()
        self.mask_ratio = mask_ratio

    def _get_random_node_mask(self, data: Data) -> torch.Tensor:
        """Uniform random node masking without topological constraints."""
        num_nodes = data.num_nodes
        num_mask_goal = max(1, int(num_nodes * self.mask_ratio))
        
        device = data.x.device if hasattr(data, 'x') and data.x is not None else 'cpu'
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        
        perm = torch.randperm(num_nodes, device=device)
        selected = perm[:num_mask_goal]
        
        mask[selected] = True
        
        return mask

    def _split_data_by_mask(self, data, mask):
        num_nodes = data.num_nodes
        if mask.sum() == 0:
            mask[torch.randint(0, num_nodes, (1,)).item()] = True
        if (~mask).sum() == 0:
            true_idx = mask.nonzero(as_tuple=True)[0][0].item()
            mask[true_idx] = False
        
        subset_ctx = ~mask
        subset_tgt = mask
        def build_subgraph(subset):
            edge_index, edge_attr = subgraph(
                subset, data.edge_index, edge_attr=data.edge_attr, 
                relabel_nodes=True, num_nodes=data.num_nodes
            )
            
            return Data(
                x=data.x[subset],
                pos=data.pos[subset] if data.pos is not None else None,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=data.y if hasattr(data, 'y') and data.y is not None else None,
                segment_id=data.segment_id if hasattr(data, 'segment_id') else None,
            )

        return build_subgraph(subset_ctx), build_subgraph(subset_tgt)

    def forward(self, data):
        mask = self._get_random_node_mask(data)
        return self._split_data_by_mask(data, mask)
class FeatureChoice(nn.Module):
    '''
    Input list of index of choiced feature for training
    '''
    def __init__(self, feature = None):
        super().__init__()
        self.feature = feature
    def forward(self, data):
        if self.feature is not None:
            data.x = data.x[:, self.feature]
        return data



class GenNormalize(torch.nn.Module):
    def __init__(self, transforms, mask_transform = None):
        super().__init__()
        self.transforms = transforms
        self.mask_transform = mask_transform
    def forward(self, data):
        out = data
        for transform in self.transforms:
            out = transform(out)
        if self.mask_transform is not None:
            context, target = self.mask_transform(out)
            return context, target
        return out


class FeatureShuffling(torch.nn.Module):
    """
    Shuffles node features (data.x) with a given ratio.
    ratio = 0: No shuffling (original graph)
    ratio = 1: Absolute random shuffling (no attention to structure)
    0 < ratio < 1: Partial shuffling (spectrum of randomness)
    """
    def __init__(self, ratio: float = 0.0):
        super().__init__()
        self.ratio = ratio

    def forward(self, data):
        if self.ratio <= 0:
            return data
        
        num_nodes = data.x.size(0)
        if num_nodes <= 1:
            return data
            
        if self.ratio >= 1.0:
            # Full random permutation
            perm = torch.randperm(num_nodes, device=data.x.device)
            data.x = data.x[perm]
        else:
            num_to_shuffle = int(num_nodes * self.ratio)
            if num_to_shuffle > 1:
                indices = torch.randperm(num_nodes, device=data.x.device)[:num_to_shuffle]
                shuffled_indices = indices[torch.randperm(num_to_shuffle, device=data.x.device)]
                data.x[indices] = data.x[shuffled_indices].clone()
        
        return data


class GaussianNoiseAugmentation(torch.nn.Module):
    """
    Adds Gaussian noise N(0, sigma) to node features.
    sigma = 0: no noise (original features preserved)
    sigma > 0: data.x = data.x + N(0, sigma)
    """
    def __init__(self, sigma: float = 0.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, data):
        if self.sigma > 0:
            noise = torch.randn_like(data.x) * self.sigma
            data.x = data.x + noise
        return data


class GaussianPositionNoise(torch.nn.Module):
    """
    Adds Gaussian noise N(0, sigma) to node positions.
    sigma = 0: no noise (original positions preserved)
    sigma > 0: data.pos = data.pos + N(0, sigma)
    """
    def __init__(self, sigma: float = 0.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, data):
        if self.sigma > 0 and data.pos is not None:
            noise = torch.randn_like(data.pos) * self.sigma
            data.pos = data.pos + noise
        return data


class Rotate(torch.nn.Module):
    """Аугментация: поворот позиций data.pos на фиксированный угол (в градусах)
    вокруг оси axis ('x' | 'y' | 'z'; по умолчанию 'z' = поворот в плоскости XY).

    Энкодер инвариантен к повороту (работает на дистанциях/рёбрах, которые
    сохраняются жёстким поворотом), поэтому поворот меняет вход только для
    предиктора, потребляющего pos. Вместе с независимым маскированием на каждое
    представление это даёт разнообразие обучающих пар. Дискретные 90° повороты —
    «чистая» перестановка/смена знака осей, корректны и после LocalPos.
    """
    _AXES = {'x': 0, 'y': 1, 'z': 2}

    def __init__(self, angle_deg: float, axis: str = 'z'):
        super().__init__()
        self.angle_deg = float(angle_deg)
        self.axis = axis
        theta = torch.deg2rad(torch.tensor(self.angle_deg))
        c, s = torch.cos(theta), torch.sin(theta)
        a = self._AXES[axis]
        i, j = [k for k in range(3) if k != a]  # плоскость вращения
        R = torch.eye(3)
        R[i, i], R[i, j] = c, -s
        R[j, i], R[j, j] = s, c
        self.register_buffer("R", R)

    def forward(self, data):
        if data.pos is None or self.angle_deg % 360 == 0:
            return data
        data.pos = data.pos @ self.R.T.to(data.pos.dtype)
        return data


def build_augments(cfg):
    """Список аугментаций представлений для collate_fn (или None, если выключены).

    Сейчас поддержаны повороты: cfg.rotation_angles = [0, 90, 180, 270] даёт по
    одному представлению на угол; пустой список / None — без аугментаций.
    """
    angles = cfg.get('rotation_angles', None)
    if not angles:
        return None
    axis = cfg.get('rotation_axis', 'z')
    return [Rotate(angle_deg=a, axis=axis) for a in angles]


def build_canonical_transform(cfg, mean_x, std_x):
    """Детерминированная канонизация графа (применяется один раз, кэшируется в RAM):
    нормализация признаков → нормализация позиций → единообразное построение графа
    из pos. Макро-метрики топологии — опционально по флагу cfg.use_macro.
    Структурные PE убраны.

    Маскирование (MaskData) сюда НЕ входит — оно стохастическое и считается на
    каждом шаге в collate_fn (на каждое представление независимо).
    """
    transforms = []

    features = cfg.get('features', None)
    if features is not None:
        features = list(features)
        transforms.append(FeatureChoice(feature=features))
        mean_x = mean_x[features]
        std_x = std_x[features]

    transforms.append(NormNoEps(mean=mean_x, std=std_x, eps=cfg.get('eps', 1e-6)))
    transforms.append(LocalPos())
    transforms.append(BuildGeometricGraph(r=cfg['r']))

    if cfg.get('use_macro', False):
        transforms.append(ThesisMacroMetrics())

    return transforms






