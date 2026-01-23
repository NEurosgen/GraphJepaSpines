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



class BatchRmsNorm(nn.Module):
    def __init__(self, in_channels,eps =1e-6):
        super().__init__()
        self.eps = eps
        self.gammma = nn.Parameter(torch.ones(in_channels))
        self.betta = nn.Parameter(torch.zeros(in_channels))
    def forward(self, x , *args, **kwargs):
        mean_sq = x.pow(2).mean(dim = 0,keepdim =True)
        rms = torch.sqrt(mean_sq + self.eps)
        x_norm = x /rms
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
    
class GraphRes(nn.Module):
    def __init__(self, in_channels, alpha , model): 
        '''
        Docstring for __init__
        
        :param self: Description
        :param in_channels: Description
        :param alpha: Description
        :param model: model return x , x0 , edge index
        '''
        super().__init__()
        self.model = model
        self.alpha = alpha 
        self.w1 = nn.Linear(in_channels,in_channels, bias=False)
        self.w2 = nn.Linear(in_channels,in_channels, bias =False)
    def forward(self, x, x0, edge_index, edge_weight = None):
        ax, _ ,_ = self.model(x ,edge_index, edge_weight)

        out = (1 - self.alpha)* self.w1(ax) + self.alpha*(self.w2(x0))
        return out, x0 , edge_index

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GraphGCNResNorm(nn.Module):
    def __init__(self, in_channels  ,alpha = 0.1):
        self.in_channels = in_channels
        self.alpha = alpha
        model = GCNConv(in_channels=in_channels,out_channels=in_channels,add_self_loops=True,normalize=True)
        self.res = GraphRes(in_channels=in_channels,alpha=alpha, model = model)
        self.norm = BatchRmsNorm(in_channels=in_channels)
        self.act = nn.ReLU()
    def forward(self, x ,x0, edge_index, edge_weigh = None):
        out,_,_ =self.res(x,x0,edge_index,edge_weigh)
        
        out = self.act(self.norm(out))

        return out, x0, edge_index




class GraphGcnEncoder(nn.Module):
    def __init__(self,in_channels, out_channels , alpha = 0.1 , num_layers = 1):
        self.proj = nn.Linear(in_channels=in_channels,out_channels=out_channels,bias=False)
        self.layers = nn.ModuleList([
            GraphGCNResNorm(in_channels=out_channels, alpha=alpha)  
            for _ in range(num_layers )
        ])
    def forward(self, x, edge_index, edge_weight = None):
        x = self.proj(x)
        x0 = x.clone()
        for layer in self.layers:
            x = layer(x, x0, edge_index, edge_weight)

        return x