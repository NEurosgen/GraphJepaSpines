import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import scatter

import hydra
from omegaconf import DictConfig

from src.data_utils.datamodule import GraphDataModule, GraphDataSet, make_folder_class_getter
from src.cli.train_model import load_stats, build_transforms
from src.models.loader_model import load_encoder_from_folder
from src.models.classificator import ClassifierLightModule, LinearClassifier
from src.data_utils.transforms import GenNormalize
from src.data_utils.stats import compute_macro_stats, extract_macro_features
def get_best_projection(pos_np):
    """Находит лучшую 2D проекцию через PCA"""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    xy = pca.fit_transform(pos_np)
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    return xy

def feature_name(idx, num_node_features=0):
    local_names = [
        'head_area', 'head_bbox_max', 'head_bbox_middle', 'head_bbox_min',
        'head_skeletal_length', 'head_volume', 'head_width_ray', 'head_width_ray_80_perc',
        'neck_area', 'neck_bbox_max', 'neck_bbox_middle', 'neck_bbox_min',
        'neck_skeletal_length', 'neck_volume', 'neck_width_ray', 'neck_width_ray_80_perc',
        'spine_bbox_volume', 'spine_n_faces', 'spine_sdf_mean', 'spine_skeletal_length',
        'spine_volume'
    ]
    if idx < num_node_features:
        if idx < len(local_names):
            return local_names[idx]
        return f"Node Feature {idx}"
    
    macro_idx = idx - num_node_features
    return f"Macro Feature {macro_idx}"


def set_neurips_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--"
    })


def plot_custom_graph(data, node_mask=None, edge_mask=None, 
                      save_path="graph.png", title="Graph",
                      mesh_data=None):  # добавили mesh_data
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patches as mpatches
    
    set_neurips_style()
    data = data.cpu()
    
    G = nx.Graph()
    for i in range(data.num_nodes):
        G.add_node(i)
        
    edge_index = data.edge_index.cpu().numpy()
    
    if edge_mask is not None:
        edge_mask_np = edge_mask.cpu().numpy()
        if len(edge_mask_np) > 0 and edge_mask_np.max() > 0:
            edge_mask_np = edge_mask_np / edge_mask_np.max()
    else:
        edge_mask_np = np.ones(data.num_edges)
        
    for j in range(data.num_edges):
        u, v = edge_index[0, j], edge_index[1, j]
        weight = edge_mask_np[j]
        if G.has_edge(u, v):
            G[u][v]['weight'] = max(G[u][v]['weight'], weight)
        else:
            G.add_edge(u, v, weight=weight)
    
    # --- Позиции узлов из data.pos ---
    if hasattr(data, 'pos') and data.pos is not None:
        pos_np = data.pos.cpu().numpy()
        
        # PCA проекция
        xy = get_best_projection(pos_np)
        
        # Нормализация
        xy_min = xy.min(axis=0)
        xy_max = xy.max(axis=0)
        xy_range = xy_max - xy_min
        xy_range[xy_range == 0] = 1
        xy_norm = (xy - xy_min) / xy_range
        
        pos = {i: xy_norm[i] for i in range(data.num_nodes)}
    else:
        pos = nx.spring_layout(G, seed=42)
        xy_norm = None
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(title)
    
    # --- Рисуем точки меша как фон ---
    if mesh_data is not None:
        # mesh_data — это просто массив xyz точек меша
        mesh_np = mesh_data if isinstance(mesh_data, np.ndarray) else mesh_data.numpy()
        mesh_xy = mesh_np[:, :2]
        
        # Нормализуем в ту же систему что и узлы
        mesh_xy_norm = (mesh_xy - xy_min) / xy_range
        
        ax.scatter(
            mesh_xy_norm[:, 0], mesh_xy_norm[:, 1],
            c='lightblue', s=0.5, alpha=0.3, zorder=1,
            label='Mesh vertices'
        )
    elif hasattr(data, 'pos') and data.pos is not None:
        # Если меша нет — рисуем сами spine positions как фоновые точки
        ax.scatter(
            xy_norm[:, 0], xy_norm[:, 1],
            c='lightgray', s=80, alpha=0.4, zorder=1,
            marker='*', label='Spine centers'
        )
    
    # --- Рисуем граф поверх ---
    if node_mask is not None:
        node_importance = node_mask.sum(dim=-1).cpu().numpy()
        if node_importance.max() > 0:
            node_importance = node_importance / node_importance.max()
    else:
        node_importance = np.ones(data.num_nodes)
        
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()] if G.edges() else []
    
    if node_mask is None and edge_mask is None:
        nx.draw(
            G, pos, ax=ax,
            node_color='#1f77b4',
            edge_color='gray',
            node_size=200,
            with_labels=False,
            zorder=2
        )
    else:
        nodes = nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_importance,
            cmap=plt.cm.Reds,
            node_size=200,
            edgecolors='black',
            linewidths=0.5,
            zorder=3
        )
        if edge_weights:
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                edge_color=edge_weights,
                edge_cmap=plt.cm.Blues,
                width=[1.0 + 3.0 * w for w in edge_weights],
                zorder=2
            )
        fig.colorbar(nodes, ax=ax, label="Node Importance", 
                    fraction=0.046, pad=0.04)
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.Blues, 
            norm=plt.Normalize(vmin=0, vmax=1)
        )
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Edge Importance", 
                    fraction=0.046, pad=0.04)
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

class GraphExplainerWrapper(nn.Module):
    def __init__(self, jepa_model, classifier, num_node_features, sigma=1.0):
        super().__init__()
        self.graph_jepa = jepa_model
        for param in self.graph_jepa.parameters():
            param.requires_grad = False
        self.graph_jepa.eval()
        
        self.classifier = classifier
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.classifier.eval()
        
        self.sigma = sigma
        self.num_node_features = num_node_features

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        x_real = x[:, :self.num_node_features]
        global_feats = x[0, self.num_node_features:].unsqueeze(0)
        
        # ─── Edge Attribute Transformation ───
        if edge_attr is not None and edge_attr.numel() > 0:
            if batch is None:
                edge_batch = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
            else:
                edge_batch = batch[edge_index[0]]
            
            min_vals = scatter(edge_attr, edge_batch, dim=0, reduce='min')
            edge_attr_processed = edge_attr - min_vals[edge_batch]
            edge_attr_exp = torch.exp(-edge_attr_processed ** 2 / (self.sigma ** 2 + 1e-6))
        else:
            edge_attr_exp = torch.ones(edge_index.size(1), 1, device=x_real.device, dtype=torch.float32)
            
        # Пропускаем через GNN только реальные признаки узлов
        graph_emb = self.graph_jepa(x_real, edge_index, edge_attr_exp)
        
        if batch is None:
            batch = torch.zeros(x_real.size(0), dtype=torch.long, device=x_real.device)
            
        graph_emb_pooled = global_add_pool(graph_emb, batch)
        
        combined_features = torch.cat([graph_emb_pooled, global_feats], dim=-1)
        
        return self.classifier(combined_features)


def _simple_collate(data_list):
    """Collate for classification — no masking, just batch graphs."""
    return Batch.from_data_list(data_list)


def _find_latest_checkpoint(path):
    """Finds the latest .ckpt file in a directory or returns the path if it's already a file."""
    import os
    import glob
    if os.path.isfile(path):
        return path
    
    ckpt_dir = os.path.join(path, "checkpoints")
    if not os.path.exists(ckpt_dir):
        ckpt_dir = path
        
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpt_files:
        return None
    
    return max(ckpt_files, key=os.path.getmtime)



def load_mesh_vertices(path):
    """
    Загружает координаты узлов из файла .pt или .pt
    Возвращает numpy array (N, 3)
    """
    import torch
    if path.endswith('.pt'):
        data = torch.load(path, weights_only=False)
    elif path.endswith('.pt'):
        data = torch.load(path, weights_only=False)
    else:
        raise ValueError(f"Unsupported file format: {path}")
    
    if isinstance(data, dict) and 'pos' in data:
        return data['pos']  # numpy array
    
    if hasattr(data, 'pos'):
        return data.pos.numpy()  # torch tensor → numpy
    
    raise ValueError(f"Could not find 'pos' in {path}")

@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    cls_cfg = cfg.classifier
    dm_cfg = cfg.datamodule
    
    # Use paths from config instead of hardcoded strings
    path_to_classifier_dir = cls_cfg.get("classifier_checkpoint_path", None)
    path_to_lejepa_dir = cls_cfg.get("checkpoint_path", None)

    # if not path_to_classifier_dir or not path_to_lejepa_dir:
    print("Warning: Missing checkpoint paths in config. Falling back to default log locations...")
    path_to_classifier_dir = "/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/lightning_logs/classifier/version_64"
    path_to_lejepa_dir = "lightning_logs/jepa/version_32"

    path_to_classifier = _find_latest_checkpoint(path_to_classifier_dir)
    path_to_lejepa = path_to_lejepa_dir # load_encoder_from_folder handles folder

    if not path_to_classifier:
        raise FileNotFoundError(f"No checkpoint found in {path_to_classifier_dir}")

    # Load statistics and transforms
    mean_x, std_x, mean_edge, std_edge = load_stats(cls_cfg.stats_path)
    transforms = build_transforms(dm_cfg, mean_x, std_x, mean_edge, std_edge)
    gen_normalize = GenNormalize(transforms=transforms, mask_transform=None)

    folder_to_label = dict(cls_cfg.get("folder_to_label", {"ab": 0, "wt": 1}))
    get_class = make_folder_class_getter(folder_to_label)

    # Dataset initialization
    ds = GraphDataSet(
        path=cls_cfg.path,
        get_class=get_class,
        transform=gen_normalize,
    )

    # Initialize models using standardized loader
    encoder = load_encoder_from_folder(path_to_lejepa)
    
    num_classes = cls_cfg.get("num_classes", 2)
    # Dynamically determine dimensions if possible, or use config
    embed_dim = cfg.network.encoder.out_channels + 7 
    classifier_head = LinearClassifier(in_channels=embed_dim, num_classes=num_classes)
    
    classifier_module = ClassifierLightModule.load_from_checkpoint(
        path_to_classifier, 
        encoder_graph=nn.Identity(), 
        classifier=classifier_head,
        strict=False,
        weights_only=False
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier_module.to(device)


    print("Computing dynamic macro statistics for dataset...")
    macro_mean, macro_std = compute_macro_stats(ds)

    # Select a sample graph for explanation
    sample_idx = cls_cfg.get("explain_sample_idx", 0)
    data = ds[sample_idx].to(device)
    print(data)
    print(sample_idx)
    print("pos shape:", data.pos.shape)
    print("pos sample:", data.pos[:3])
    print("pos min/max:", data.pos.min(0).values, data.pos.max(0).values)

    # Extract macro features
    global_feats = extract_macro_features(data, macro_mean, macro_std) # Формат: [1, num_macro_features]

    # Сохраняем исходное количество признаков узла
    num_node_features = data.x.size(1)

    # Дублируем макро-признаки для каждого узла графа
    global_feats_broadcasted = global_feats.expand(data.x.size(0), -1)
    
    # Конкатенируем локальные признаки узлов и макро-признаки
    x_combined = torch.cat([data.x, global_feats_broadcasted], dim=-1)

    model_wrapper = GraphExplainerWrapper(
        jepa_model=encoder, 
        classifier=classifier_module.classifier,
        num_node_features=num_node_features,
        sigma=cls_cfg.get("sigma", 1.0)
    ).to(device)
    
    # Configure GNN explainer
    explainer = Explainer(
        model=model_wrapper,
        algorithm=GNNExplainer(epochs=5000),
        explanation_type='model',
        node_mask_type='attributes',   # Будет вычислять важность для x_combined
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw',
        ),
    )

    print(f"Explaining sample {sample_idx}...")
    # Передаем объединенный тензор x_combined
    explanation = explainer(x=x_combined, edge_index=data.edge_index, edge_attr=data.edge_attr)
    print("Explanation computed successfully!")
    
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patches as mpatches

    os.makedirs("explanations", exist_ok=True)
    set_neurips_style()
    
    # График важности признаков
    feat_importance = explanation.node_mask.mean(dim=0).cpu().numpy()
    top_k = min(20, len(feat_importance))
    indices = np.argsort(feat_importance)[-top_k:]
    
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4' if idx < num_node_features else '#ff7f0e' for idx in indices]
    plt.barh(range(top_k), feat_importance[indices], align='center', color=colors)
    plt.yticks(range(top_k), [feature_name(idx, num_node_features) for idx in indices])
    plt.xlabel('Attribute Importance')
    plt.title('Feature Importance')
    
    # Custom legend
    node_patch = mpatches.Patch(color='#1f77b4', label='Local Node Feature')
    macro_patch = mpatches.Patch(color='#ff7f0e', label='Macro Feature')
    plt.legend(handles=[node_patch, macro_patch], loc='lower right')
    
    plt.tight_layout()
    plt.savefig("explanations/feature_importance_sph.png")
    plt.close()
    print("Feature importance graph saved to: explanations/feature_importance_sph.png")
    mesh_points = load_mesh_vertices("/home/eugen/Desktop/CodeWork/Projects/Diplom/Archiv/processed_dataset/ab/ab24-1_processed.pt")
    # Отрисовка оригинального графа
    plot_custom_graph(
    data=data,
    node_mask=None,
    edge_mask=None,
    save_path="explanations/original_graph_sph.png",
    title="Original Graph Structure",
    mesh_data=mesh_points  # или None если нет меша
)
    print("Original graph saved to: explanations/original_graph_sph.png")

    # Отрисовка графа с подсветкой важности
    plot_custom_graph(
        data=data,
        node_mask=explanation.node_mask,
        edge_mask=explanation.edge_mask,
        save_path="explanations/graph_explanation_sph.png",
        title="Explanation (Important Nodes & Edges)",
        mesh_data=mesh_points  # или None если нет меша
    )
    print("Graph explanation saved to: explanations/graph_explanation_sph.png")

if __name__ == "__main__":
    main()