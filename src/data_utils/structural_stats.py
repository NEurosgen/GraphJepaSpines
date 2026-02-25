import torch
import torch.nn as nn
from torch_geometric.utils import to_networkx
import networkx as nx
import numpy as np

class ThesisMacroMetrics(nn.Module):
    """
    Computes macro graph topology metrics used in the previous thesis baseline:
    1. Average Subgraph Size (Connected Components)
    2. Average Intra-cluster Distance (Average shortest path within components)
    3. Modularity (using Louvain)
    4. Average Clustering Coefficient
    
    Stores the result in data.macro_metrics (Tensor of shape [1, 4])
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, data):
        device = data.x.device if data.x is not None else 'cpu'
        G = to_networkx(data, to_undirected=True)
        
        # 1. Average Subgraph Size
        components = list(nx.connected_components(G))
        if len(components) > 0:
            avg_subgraph_size = np.mean([len(c) for c in components])
        else:
            avg_subgraph_size = 0.0
            
        # 2. Average Intra-cluster Distance
        intra_dists = []
        for c in components:
            if len(c) > 1:
                subg = G.subgraph(c)
                try:
                    intra_dists.append(nx.average_shortest_path_length(subg))
                except:
                    pass
        avg_intra_dist = np.mean(intra_dists) if len(intra_dists) > 0 else 0.0
        
        # 3. Modularity
        try:
            communities = nx.community.louvain_communities(G)
            modularity = nx.community.modularity(G, communities)
        except:
            modularity = 0.0
            
        # 4. Clustering Coefficient
        try:
            clustering_coeff = nx.average_clustering(G)
        except:
            clustering_coeff = 0.0
            
        # Normalization (Basic clipping/scaling to keep within reasonable ranges for NN)
        metrics = [
            min(max(avg_subgraph_size / 20.0, 0.0), 1.0),    # Approx range [1, 50] -> [0, 1]
            min(max(avg_intra_dist / 5.0, 0.0), 1.0),        # Approx range [1, 10] -> [0, 1]
            min(max((modularity + 0.5) / 1.5, 0.0), 1.0),    # Approx bounds [-0.5, 1.0] -> [0, 1]
            min(max(clustering_coeff, 0.0), 1.0)             # [0, 1]
        ]
        
        data.macro_metrics = torch.tensor(metrics, dtype=torch.float32, device=device).unsqueeze(0)
        return data
