import torch_geometric
import torch


def fast_normalization_by_features(data, eps=1e-6):
    """
    Считает среднее и std для каждого из 21 признака, 
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
            stds[i] = valid_data.std() 
        else:
            means[i] = 0.0
            stds[i] = 1.0
            
    return means, stds




        

class NormNoEps(torch.nn.Module):
    def __init__(self, mean : torch.Tensor , std : torch.Tensor , eps: float = 0.0):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        if torch.any(std.abs() < 1e-8):
            raise ValueError("Your std is too small. It's dangerous for division!")
        self.eps = eps
    def forward(self,x) -> torch.Tensor:
        mask = (x.abs() > self.eps)
        normalized_x  = (x - (self.mean))/(self.std)
        return torch.where(mask,normalized_x,x)
    

class EdgeNorm(torch.nn.Module):
    def __init__(self, mean : torch.Tensor , std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        if torch.any(std.abs() < 1e-8):
            raise ValueError("Your std is too small. It's dangerous for division!")
    def forward(self, edge_attr):
        return (edge_attr - self.mean)/self.std
        

class GenNormalize(torch.nn.Module):
    def __init__(self, mean_x, std_x , mean_edge, std_edge, eps = 0):
        super().__init__()
        self.norm_x = NormNoEps(mean_x,std_x , eps)
        self.norm_edge = EdgeNorm(mean_edge,std_edge)
    def forward(self, data):
        data.x = self.norm_x(data.x)
        data.edge_attr = self.norm_edge(data.edge_attr)
        return data

