# Idea from Residual Connections and Normalization Can Provably Prevent Oversmoothing in GNNs (2025)

import torch_geometric
from torch import nn
from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn



import torch
from torch_geometric.utils import add_self_loops, separation_plus_low_rank_spectral_decomposition # если доступно, но лучше вручную
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





class GraphNormv2(nn.Module):
    def __init__(self, in_channels, V_kp):
        super().__init__()
 
        self.register_buffer('V_kp', V_kp) 
        k_plus_1 = V_kp.shape[1]
        self.tau = nn.Parameter(torch.Tensor(in_channels, k_plus_1))
        self.gamma = nn.Parameter(torch.ones(in_channels))
        self.beta = nn.Parameter(torch.zeros(in_channels))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.tau, a=0, b=0.1)

    def forward(self, x):
        N, C = x.shape
        vt_x = torch.matmul(self.V_kp.t(), x) 
        

        inner = torch.sum(self.tau.t() * vt_x, dim=0)
        

        weighted_tau = self.tau * inner.unsqueeze(1)
        projection = torch.matmul(self.V_kp, weighted_tau.t())
        
        x_centered = x - projection
        sigma = torch.norm(x_centered, p=2, dim=0) / (N**0.5)
        x_scaled = x_centered / (sigma + 1e-6)
        
        return self.gamma * x_scaled + self.beta
    
class GraphResNorm(nn.Module):
    def __init__(self, in_channels, alpha):
        super().__init__()
        self.alpha = alpha 
        self.w1 = nn.Linear(in_channels,in_channels, bias=False)
        self.w2 = nn.Linear(in_channels,in_channels, bias =False)
    def forward(self, x, x0, edge_index, edge_weight = None):
        ax = GCNConv(x ,edge_index, edge_weight)

        out = (1 - self.alpha)* self.w1(ax) + self.alpha*(self.w2(x0))

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class DeepGCNBlock(nn.Module):
    def __init__(self, channels, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.conv = GCNConv(channels, channels, add_self_loops=True, normalize=True)
        self.res_weight = nn.Linear(channels, channels, bias=False)
        self.norm = GraphNormv2(channels, V_kp)


    def forward(self, x, x_0, edge_index):
        axw1 = self.conv(x, edge_index)
        res = self.res_weight(x_0)
        x = (1 - self.alpha) * axw1 + self.alpha * res
        x = self.norm(x) 
        
        return x, x_0 ,edge_index


class GCNEncoder(nn.Module):
    def __init__(self, n_layers, in_channels, out_channels, alpha = 0.1):
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_gcn = GCNConv(in_channels=in_channels,out_channels=out_channels,add_self_loops=True,normalize=True)
        self.module = nn.Sequential([ DeepGCNBlock(out_channels,alpha=alpha) for i in range(self.num_layers- 1)])
    def forward(self,x0 , edge_index, edge_weight = None):
        x = self.init_gcn(x0,edge_index)
        out = self.module(x,x0,edge_index)
        return out, x0 , edge_index