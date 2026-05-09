import torch
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import hydra
from src.data_utils.datamodule import GraphDataSet, make_folder_class_getter
import os
import numpy as np

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
        "axes.grid": False, # Better to turn off grid for graph plots
    })

def plot_graph_explanation_side_by_side(data, node_mask, edge_mask, save_path):
    set_neurips_style()
    data = data.cpu()
    
    G = nx.Graph()
    for i in range(data.num_nodes):
        G.add_node(i)
        
    edge_index = data.edge_index.cpu().numpy()
    if edge_mask is not None:
        edge_mask_np = edge_mask.cpu().numpy()
        if edge_mask_np.max() > 0:
            edge_mask_np = edge_mask_np / edge_mask_np.max()
    else:
        edge_mask_np = np.ones(data.num_edges)
        
    # Aggregate edge weights
    for j in range(data.num_edges):
        u, v = edge_index[0, j], edge_index[1, j]
        weight = edge_mask_np[j]
        if G.has_edge(u, v):
            G[u][v]['weight'] = max(G[u][v]['weight'], weight)
        else:
            G.add_edge(u, v, weight=weight)
            
    if hasattr(data, 'pos') and data.pos is not None:
        pos = {i: data.pos[i].numpy()[:2] for i in range(data.num_nodes)}
    else:
        pos = nx.spring_layout(G, seed=42)
        
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Original Graph
    ax1 = axes[0]
    ax1.set_title("Original Graph")
    nx.draw(
        G, pos, ax=ax1, 
        node_color='#1f77b4', 
        edge_color='gray', 
        node_size=300, 
        with_labels=False
    )
    
    # 2. Explained Graph
    ax2 = axes[1]
    ax2.set_title("Explanation (Important Nodes & Edges)")
    
    if node_mask is not None:
        node_importance = node_mask.sum(dim=-1).cpu().numpy()
        if node_importance.max() > 0:
            node_importance = node_importance / node_importance.max()
    else:
        node_importance = np.ones(data.num_nodes)
        
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax2, 
        node_color=node_importance, 
        cmap=plt.cm.Reds, 
        node_size=300,
        edgecolors='black',
        linewidths=0.5
    )
    
    edges = nx.draw_networkx_edges(
        G, pos, ax=ax2,
        edge_color=edge_weights,
        edge_cmap=plt.cm.Blues,
        width=[1.0 + 3.0 * w for w in edge_weights]
    )
    
    fig.colorbar(nodes, ax=ax2, label="Node Importance", orientation='vertical', fraction=0.046, pad=0.04)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax2, label="Edge Importance", orientation='vertical', fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# Dummy test
cfg = OmegaConf.create({"classifier": {"path": "/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/notebooks/9009_prepared", "folder_to_label": {"ab": 0, "wt": 1}}})
get_class = make_folder_class_getter(cfg.classifier.folder_to_label)
ds = GraphDataSet(path=cfg.classifier.path, get_class=get_class, transform=None)
data = ds[0]

# mock explanation masks
node_mask = torch.rand((data.num_nodes, 21))
edge_mask = torch.rand(data.num_edges)

plot_graph_explanation_side_by_side(data, node_mask, edge_mask, "test_explanation.png")
print("Done!")
