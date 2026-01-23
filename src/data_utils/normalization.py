import torch_geometric
import torch


def save_from_eps( data , eps = 0):
    mask = (data.abs() > eps)
    if not torch.any(mask):
        return 0 , 1
    mean = data[mask].mean()
    std = data[mask].std()
    return mean , std


def normalization(data, eps = 0):
    for row in data:
        mean, std = save_from_eps(row, eps)
        row.sub_(-mean)._div(std)

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
        