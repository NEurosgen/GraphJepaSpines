# Idea from Residual Connections and Normalization Can Provably Prevent Oversmoothing in GNNs (2025)
from torch import nn
from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU 
from torch_geometric.utils import scatter
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

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
    



class BatchRmsNorm(nn.Module):
    def __init__(self, in_channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(in_channels))
        self.beta = nn.Parameter(torch.zeros(in_channels))
    
    def forward(self, x, *args, **kwargs):
        mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_sq + self.eps)
        x_norm = x / rms
        return self.gamma * x_norm + self.beta
    

class WeightedGINConv(MessagePassing):
    """Графовая свертка GIN с поддержкой весовых коэффициентов ребер."""
    def __init__(self, nn_mlp, eps=0.0, train_eps=True):
        super().__init__(aggr='add')
        self.nn = nn_mlp
        self.initial_eps = eps
        if train_eps:
            self.eps = nn.Parameter(torch.tensor([eps]))
        else:
            self.register_buffer('eps', torch.tensor([eps]))

    def forward(self, x, edge_index, edge_weight):
        if edge_index is None or (hasattr(edge_index, 'numel') and edge_index.numel() == 0):
            return self.nn((1 + self.eps) * x)
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = out + (1 + self.eps) * x
        return self.nn(out)

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.view(-1, 1)

class GraphGINResNorm(nn.Module):
    def __init__(self, in_channels ,init_alpha = 1e-3):
        super().__init__()
        self.in_channels = in_channels
    
        mlp = Sequential(
            Linear(in_channels, in_channels),
            ReLU(),
            Linear(in_channels, in_channels)
        )

        self.model = WeightedGINConv(nn_mlp=mlp, train_eps=True)
        self.alpha = nn.Parameter(init_alpha * torch.ones(in_channels))
        self.norm = BatchRmsNorm(in_channels=in_channels)
    
    def forward(self, x, edge_index, edge_weight=None):
        out = self.model(x, edge_index, edge_weight)
        out = F.relu(out)
        out = self.alpha*out + x
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
        
    
    def forward(self, x, edge_index, edge_weight=None):
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
        return x

def linear_edge_weighting(batch):
        edge_batch = batch.batch[batch.edge_index[0]]
        min_vals = scatter(batch.edge_attr, edge_batch, dim=0, reduce='min')
        max_vals = scatter(batch.edge_attr, edge_batch, dim=0, reduce='max')
        denom = (max_vals[edge_batch] - min_vals[edge_batch]).clamp_(min=1e-6)
        batch.edge_attr = (batch.edge_attr - min_vals[edge_batch]) / denom
            
        return batch

class GraphLatent(nn.Module):
    def __init__(self, encoder, macro_mean, macro_std, pooling):
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
    def forward(self,batch):
        with torch.no_grad():
            self.encoder
        
            if batch.edge_attr is not None and batch.edge_attr.numel() > 0:
                batch = linear_edge_weighting(batch=batch)
            
            node_emb = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
            graph_emb = self.pooling(node_emb, batch.batch)
            if self.macro_mean is not None and self.macro_std is not None:
                thesis_macro = batch.macro_metrics
                thesis_macro = (thesis_macro - self.macro_mean.to(thesis_macro.device)) / (self.macro_std.to(thesis_macro.device) + 1e-6)
                graph_emb = torch.cat([graph_emb, thesis_macro], dim=-1)
        return graph_emb
    
