# Idea from Residual Connections and Normalization Can Provably Prevent Oversmoothing in GNNs (2025)

import torch_geometric
from torch import nn
from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn



import torch
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import numpy as np
def compute_v_kp_for_pyg(data, k):
    N = data.num_nodes
    edge_index = data.edge_index
    

    edge_index, _ = add_self_loops(edge_index, num_nodes=N)

    adj = to_scipy_sparse_matrix(edge_index, num_nodes=N).tocsr()
    
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ adj @ D_inv_sqrt
    
    def matvec(x):
        ax = A_norm @ x
        return ax - np.mean(ax) * np.ones_like(ax)

    A_hat_op = sp.linalg.LinearOperator((N, N), matvec=matvec)
    
    eigenvalues, V_k = eigsh(A_hat_op, k=k, which='LM')
    V_k = torch.from_numpy(V_k).float()
    ones = torch.ones((N, 1))
    r = ones - V_k @ (V_k.t() @ ones)
    r_norm = torch.norm(r, p=2)
    if r_norm > 1e-6:
        r = r / r_norm
        V_kp = torch.cat([V_k, r], dim=1) # 
    else:
        V_kp = V_k
        
    return V_kp



class BatchRmsNorm(nn.Module):
    def __init__(self, in_channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(in_channels))
        self.beta = nn.Parameter(torch.zeros(in_channels))
    
    def forward(self, x, *args, **kwargs):
        # FATAL BUG FIX: Compute RMS per-node (dim=-1) instead of per-batch (dim=0)
        mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_sq + self.eps)
        x_norm = x / rms
        return self.gamma * x_norm + self.beta
class GraphNormv2(nn.Module):               # Слишком тяжело для вычисления для нестатических графов :(
    def __init__(self, in_channels, k):
        super().__init__()
 

        self.tau = nn.Parameter(torch.Tensor(in_channels, k + 1))
        self.gamma = nn.Parameter(torch.ones(in_channels))
        self.beta = nn.Parameter(torch.zeros(in_channels))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.tau, a=0, b=0.1)

    def forward(self, x, V_kp):

        N, C = x.shape
        vt_x = torch.matmul(self.V_kp.t(), x) 


        inner = torch.sum(self.tau.t() * vt_x, dim=0)
        

        weighted_tau = self.tau * inner.unsqueeze(1)
        projection = torch.matmul(self.V_kp, weighted_tau.t())
        
        x_centered = x - projection
        sigma = torch.norm(x_centered, p=2, dim=0) / (N**0.5)
        x_scaled = x_centered / (sigma + 1e-6)
        
        return self.gamma * x_scaled + self.beta
    


import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GraphGCNResNorm(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.model = GCNConv(in_channels=in_channels, out_channels=in_channels, add_self_loops=True, normalize=True)
        self.norm = BatchRmsNorm(in_channels=in_channels)
    
    def forward(self, x, edge_index, edge_weight=None):
        out = self.model(x, edge_index, edge_weight)
        out = out + x
        out = self.norm(out)
        return out




class GraphGcnEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.proj = nn.Linear(in_features=in_channels, out_features=out_channels, bias=False)
        self.proj.requires_grad_(False)
        self.layers = nn.ModuleList([
            GraphGCNResNorm(in_channels=out_channels)
            for _ in range(num_layers)
        ])
        self.act = nn.ReLU()
    
    def forward(self, x, edge_index, edge_weight=None):
        x = self.proj(x)
        for layer in self.layers[:-1]:
            x = layer(x, edge_index, edge_weight)
            x = self.act(x)
        x = self.layers[-1](x, edge_index, edge_weight)
        return x
    
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv  # Импортируем GINConv
# Импортируем необходимые компоненты для MLP
from torch.nn import Sequential, Linear, ReLU 

# --- Вспомогательный код из вашего примера (оставлен без изменений) ---
# (Функция compute_v_kp_for_pyg и классы BatchRmsNorm, GraphNormv2)

class BatchRmsNorm(nn.Module):
    def __init__(self, in_channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(in_channels))
        self.beta = nn.Parameter(torch.zeros(in_channels))
    
    def forward(self, x, *args, **kwargs):
        # FATAL BUG FIX: Compute RMS per-node (dim=-1) instead of per-batch (dim=0)
        mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_sq + self.eps)
        x_norm = x / rms
        return self.gamma * x_norm + self.beta



class GraphGINResNorm(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
    
        mlp = Sequential(
            Linear(in_channels, in_channels),
            ReLU(),
            Linear(in_channels, in_channels)
        )

        self.model = GINConv(nn=mlp, train_eps=True)

        self.norm = BatchRmsNorm(in_channels=in_channels)
    
    def forward(self, x, edge_index, edge_weight=None):
        out = self.model(x, edge_index)
        out = out + x
        out = self.norm(out)
        return out


class GraphGinEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.proj = nn.Linear(in_features=in_channels, out_features=out_channels, bias=False)
        self.proj.requires_grad_(False)
        self.layers = nn.ModuleList([
            GraphGINResNorm(in_channels=out_channels)
            for _ in range(num_layers)
        ])
        self.act = nn.ReLU()
    
    def forward(self, x, edge_index, edge_weight=None):
        x = self.proj(x)
        for layer in self.layers[:-1]:
            x = layer(x, edge_index, edge_weight)
            x = self.act(x)
        x = self.layers[-1](x, edge_index, edge_weight)
        return x
    
class GraphLatent(nn.Module):
    def __init__(self, encoder, macro_mean, macro_std, pooling, sigma=1):
        super().__init__()
        self.encoder = encoder
        if macro_mean is not None:
            self.register_buffer("macro_mean", macro_mean)
        else:
            self.macro_mean = None
        if macro_std is not None:
            self.register_buffer("macro_std", macro_std)
        else:
            self.macro_std = None
        self.pooling = pooling
        self.sigma = sigma
    def forward(self,batch):
        with torch.no_grad():
            self.encoder.eval()
            edge_attr = batch.edge_attr
            
            # Note: edge_attr may be normalized (centered). RBF expects distance-like values (>=0).
            # We shift by the graph minimum value to ensure weights are in (0, 1] range consistently per graph.
            if edge_attr is not None and edge_attr.numel() > 0:
                from torch_geometric.utils import scatter
                edge_batch = batch.batch[batch.edge_index[0]]
                min_vals = scatter(edge_attr, edge_batch, dim=0, reduce='min')
                edge_attr = edge_attr - min_vals[edge_batch]
                edge_attr = torch.exp(-edge_attr ** 2 / (self.sigma ** 2 + 1e-6))
            
            node_emb = self.encoder(batch.x, batch.edge_index, edge_attr)
            graph_emb = self.pooling(node_emb, batch.batch)
            if self.macro_mean is not None and self.macro_std is not None:
                thesis_macro = batch.macro_metrics
                thesis_macro = (thesis_macro - self.macro_mean.to(thesis_macro.device)) / (self.macro_std.to(thesis_macro.device) + 1e-6)
                graph_emb = torch.cat([graph_emb, thesis_macro], dim=-1)
        return graph_emb