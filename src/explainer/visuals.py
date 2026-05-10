import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

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

def get_best_projection(pos_np):
    """Находит лучшую 2D проекцию через PCA"""
    pca = PCA(n_components=2)
    xy = pca.fit_transform(pos_np)
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    return xy

def plot_custom_graph(data, node_mask=None, edge_mask=None, 
                      save_path="graph.png", title="Graph",
                      mesh_data=None):
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
        mesh_np = mesh_data if isinstance(mesh_data, np.ndarray) else mesh_data.numpy()
        mesh_xy = mesh_np[:, :2]
        
        mesh_xy_norm = (mesh_xy - xy_min) / xy_range
        
        ax.scatter(
            mesh_xy_norm[:, 0], mesh_xy_norm[:, 1],
            c='lightblue', s=0.5, alpha=0.3, zorder=1,
            label='Mesh vertices'
        )
    elif hasattr(data, 'pos') and data.pos is not None:
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
        )
    else:
        nodes = nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_importance,
            cmap=plt.cm.Reds,
            node_size=200,
            edgecolors='black',
            linewidths=0.5,
        )
        if edge_weights:
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                edge_color=edge_weights,
                edge_cmap=plt.cm.Blues,
                width=[1.0 + 3.0 * w for w in edge_weights],
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
