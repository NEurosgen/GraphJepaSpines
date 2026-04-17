import torch_geometric
import torch
import torch.nn as nn
from torch_geometric.nn import knn_graph, radius_graph


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



def create_mask_collate_fn(transform: 'GenNormalize' = None):
    from torch_geometric.data import Batch
    
    def mask_collate_fn(batch):
        if transform is None:
            return Batch.from_data_list(batch)
        
        contexts = []
        targets = []
        
        for data in batch:
            
            ctx, tgt = transform(data)
            if ctx.num_nodes > 0 and tgt.num_nodes > 0:
                contexts.append(ctx)
                targets.append(tgt)

        
        if len(contexts) == 0:
            return None
        
        context_batch = Batch.from_data_list(contexts)
        target_batch = Batch.from_data_list(targets)
        
        return context_batch, target_batch
    
    return mask_collate_fn



        

class NormNoEps(torch.nn.Module):
    def __init__(self, mean : torch.Tensor , std : torch.Tensor , eps: float = 0.0):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        if torch.any(std.abs() < 1e-8):
            raise ValueError("Your std is too small. It's dangerous for division!")
        self.eps = eps
    def forward(self,data) -> torch.Tensor:
        mask = (data.x.abs() > self.eps)
        normalized_x  = (data.x - (self.mean))/(self.std)
        data.x = torch.where(mask,normalized_x,data.x)
        return data
    

class EdgeNorm(torch.nn.Module):
    def __init__(self, mean : torch.Tensor , std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        if torch.any(std.abs() < 1e-8):
            raise ValueError("Your std is too small. It's dangerous for division!")
    def forward(self, data):
        data.edge_attr = (data.edge_attr - self.mean)/self.std
        return data
        
class LocalPos(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,data):
        pos = data.pos
        pos_mean = torch.mean(pos,dim=0)
        pos_std = torch.std(pos,dim=0).clamp(min=1e-6)
        pos = (pos - pos_mean)/pos_std
        data.pos = pos
        return data



import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, k_hop_subgraph, to_undirected

class GraphPruning(torch.nn.Module):
    '''
    Return pruned graph with knn or radius
    '''
    def __init__(self, k = -1, r = -1.0, mutual = False):
        super().__init__()
        self.k = k
        self.r = r
        self.mutual = mutual
    def forward(self, data):
        if self.k < 0 and self.r < 0.0:
            return data
            
        batch = data.batch
        num_nodes = data.num_nodes
        
        if self.r >= 0.0:
            # Generate edges based on spatial distance (like DBSCAN)
            # max_num_neighbors is set to num_nodes because default is 32, which limits dense graphs
            new_edge_index = radius_graph(data.pos, r=self.r, batch=batch, loop=False, max_num_neighbors=num_nodes)
            row, col = new_edge_index
        else:
            # Fallback to older KNN logic
            new_edge_index = knn_graph(data.pos, self.k, batch, loop=False)
            row, col = new_edge_index
            
        if self.mutual and self.r < 0.0: # mutual has no meaning for radius, it's always symmetric
            knn_hashes = row*num_nodes + col
            knn_hashes_rev = col*num_nodes + row
            is_mutual = torch.isin(knn_hashes,knn_hashes_rev)
            valid_hashes = knn_hashes[is_mutual]
        else:
            valid_hashes = row*num_nodes + col
            
        curr_row, curr_col = data.edge_index
        curr_hashes = curr_row * num_nodes + curr_col
        mask = torch.isin(curr_hashes, valid_hashes)
        
        # Also need to make sure we don't accidentally lose edges that radius_graph found
        # but the original mesh didn't have? 
        # Actually, the user's code previously filtered existing mesh edges to only keep those 
        # present in KNN. We will do the same for Radius Graph.
        
        pruned_edge_index = data.edge_index[:, mask]
        pruned_edge_attr = None
        if data.edge_attr is not None:
            pruned_edge_attr = data.edge_attr[mask]
            
        return Data(
            x=data.x,
            edge_index=pruned_edge_index,
            edge_attr=pruned_edge_attr,
            pos=data.pos,
            y=data.y if hasattr(data, 'y') and data.y is not None else None,
            segment_id=data.segment_id if hasattr(data, 'segment_id') else None,
            batch=batch
        )





class MaskData(torch.nn.Module):
    '''
    Waring! This class return TWO graphs , contex and target (In JEPA notations)

    '''
    def __init__(self, mask_ratio: float):
        super().__init__()
        self.mask_ratio = mask_ratio

    def _get_random_node_mask(self, data: Data) -> torch.Tensor:
        """Uniform random node masking without topological constraints."""
        num_nodes = data.num_nodes
        num_mask_goal = max(1, int(num_nodes * self.mask_ratio))
        
        device = data.x.device if hasattr(data, 'x') and data.x is not None else 'cpu'
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        
        perm = torch.randperm(num_nodes, device=device)
        selected = perm[:num_mask_goal]
        
        mask[selected] = True
        
        return mask

    def _split_data_by_mask(self, data, mask):
        num_nodes = data.num_nodes
        if mask.sum() == 0:
            mask[torch.randint(0, num_nodes, (1,)).item()] = True
        if (~mask).sum() == 0:
            true_idx = mask.nonzero(as_tuple=True)[0][0].item()
            mask[true_idx] = False
        
        subset_ctx = ~mask
        subset_tgt = mask
        def build_subgraph(subset):
            edge_index, edge_attr = subgraph(
                subset, data.edge_index, edge_attr=data.edge_attr, 
                relabel_nodes=True, num_nodes=data.num_nodes
            )
            
            return Data(
                x=data.x[subset],
                pos=data.pos[subset] if data.pos is not None else None,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=data.y if hasattr(data, 'y') and data.y is not None else None,
                segment_id=data.segment_id if hasattr(data, 'segment_id') else None,
            )

        return build_subgraph(subset_ctx), build_subgraph(subset_tgt)

    def forward(self, data):
        mask = self._get_random_node_mask(data)
        return self._split_data_by_mask(data, mask)
class FeatureChoice(nn.Module):
    '''
    Input list of index of choiced feature for training
    '''
    def __init__(self, feature = None):
        super().__init__()
        self.feature = feature
    def forward(self, data):
        if self.feature is not None:
            data.x = data.x[:, self.feature]
        return data


class LaplacianPE(torch.nn.Module):
    """
    Laplacian Positional Encoding.
    Вычисляет k наименьших нетривиальных собственных векторов 
    нормализованного лапласиана L = I - D^{-1/2}AD^{-1/2}
    и конкатенирует их к data.x.
    """
    def __init__(self, k: int = 4):
        super().__init__()
        self.k = k

    def forward(self, data):
        from torch_geometric.utils import to_scipy_sparse_matrix, get_laplacian
        import numpy as np
        from scipy.sparse.linalg import eigsh

        num_nodes = data.num_nodes
        device = data.x.device

        # Normalized Laplacian
        edge_index_lap, edge_weight_lap = get_laplacian(
            data.edge_index, normalization='sym', num_nodes=num_nodes
        )
        L = to_scipy_sparse_matrix(edge_index_lap, edge_weight_lap, num_nodes=num_nodes)

        # Compute k smallest non-trivial eigenvectors
        # We request k+1 eigenvectors and skip the first (trivial, eigenvalue ≈ 0)
        num_to_compute = min(self.k + 1, num_nodes - 1)
        if num_to_compute < 2:
            # Graph too small — pad with zeros
            pe = torch.zeros(num_nodes, self.k, device=device)
        else:
            try:
                eigenvalues, eigenvectors = eigsh(
                    L.tocsc(), k=num_to_compute, which='SM', tol=1e-4
                )
                # Sort by eigenvalue ascending, skip the first (trivial)
                idx = np.argsort(eigenvalues)
                eigenvectors = eigenvectors[:, idx[1:]]  # skip trivial

                pe = torch.from_numpy(eigenvectors).float().to(device)
                # Pad if fewer eigenvectors than k
                if pe.shape[1] < self.k:
                    padding = torch.zeros(num_nodes, self.k - pe.shape[1], device=device)
                    pe = torch.cat([pe, padding], dim=1)
                else:
                    pe = pe[:, :self.k]
            except Exception:
                pe = torch.zeros(num_nodes, self.k, device=device)

        # Sign ambiguity: make the largest absolute value in each column positive
        sign = torch.sign(pe[pe.abs().argmax(dim=0), torch.arange(self.k, device=device)])
        sign[sign == 0] = 1
        pe = pe * sign.unsqueeze(0)

        data.x = torch.cat([data.x, pe], dim=1)
        return data


class CentralityEncoding(torch.nn.Module):
    """
    Centrality Encoding.
    Конкатенирует степень узла (нормированную на max degree) к data.x.
    """
    def __init__(self):
        super().__init__()

    def forward(self, data):
        from torch_geometric.utils import degree

        device = data.x.device
        num_nodes = data.num_nodes
        row = data.edge_index[0]
        deg = degree(row, num_nodes=num_nodes).float().to(device)

        max_deg = deg.max()
        if max_deg > 0:
            deg = deg / max_deg

        data.x = torch.cat([data.x, deg.unsqueeze(1)], dim=1)
        return data


class RandomWalkPE(torch.nn.Module):
    """
    Random Walk Positional Encoding.
    Для каждого шага 1..walk_length вычисляет вероятность возврата
    (диагональный элемент M^step), где M = D^{-1}A — матрица переходов.
    Конкатенирует walk_length столбцов к data.x.
    """
    def __init__(self, walk_length: int = 8):
        super().__init__()
        self.walk_length = walk_length

    def forward(self, data):
        from torch_geometric.utils import to_scipy_sparse_matrix
        import numpy as np

        num_nodes = data.num_nodes
        device = data.x.device

        # Build adjacency and transition matrix M = D^{-1}A
        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=num_nodes).tocsr()
        deg = np.array(adj.sum(axis=1)).flatten()
        deg_inv = np.zeros_like(deg)
        nonzero = deg > 0
        deg_inv[nonzero] = 1.0 / deg[nonzero]

        import scipy.sparse as sp
        D_inv = sp.diags(deg_inv)
        M = D_inv @ adj  # transition matrix

        # Compute diag(M^step) for step = 1..walk_length
        pe_list = []
        M_power = M.copy()
        for _ in range(self.walk_length):
            diag = torch.from_numpy(M_power.diagonal().copy()).float().to(device)
            pe_list.append(diag.unsqueeze(1))
            M_power = M_power @ M

        pe = torch.cat(pe_list, dim=1)  # (num_nodes, walk_length)
        data.x = torch.cat([data.x, pe], dim=1)
        return data


class GenNormalize(torch.nn.Module):
    def __init__(self, transforms, mask_transform = None):
        super().__init__()
        self.transforms = transforms
        self.mask_transform = mask_transform
    def forward(self, data):
        out = data
        for transform in self.transforms:
            out = transform(out)
        if self.mask_transform is not None:
            context, target = self.mask_transform(out)
            return context, target
        return out


class ConcatStructuralPE(torch.nn.Module):
    """
    Конкатенирует предварительно рассчитанные структурные фичи
    (Laplacian, Centrality, RandomWalk) к data.x
    без их перевычисления.
    """
    def __init__(self):
        super().__init__()

    def forward(self, data):
        pe_list = []
        if hasattr(data , 'laplacian_pe') and data.laplacian_pe is not None:
            pe_list.append(data.laplacian_pe.to(data.x.device))
        if hasattr(data , 'centrality_pe') and data.centrality_pe is not None:
            pe_list.append(data.centrality_pe.to(data.x.device))
        if hasattr(data , 'random_walk_pe') and data.random_walk_pe is not None:  
            pe_list.append(data.random_walk_pe.to(data.x.device))
            
        if len(pe_list) > 0:
            pe_tensor = torch.cat(pe_list, dim=1)
            data.x = torch.cat([data.x, pe_tensor], dim=1)
            
        return data


class FeatureShuffling(torch.nn.Module):
    """
    Shuffles node features (data.x) with a given ratio.
    ratio = 0: No shuffling (original graph)
    ratio = 1: Absolute random shuffling (no attention to structure)
    0 < ratio < 1: Partial shuffling (spectrum of randomness)
    """
    def __init__(self, ratio: float = 0.0):
        super().__init__()
        self.ratio = ratio

    def forward(self, data):
        if self.ratio <= 0:
            return data
        
        num_nodes = data.x.size(0)
        if num_nodes <= 1:
            return data
            
        if self.ratio >= 1.0:
            # Full random permutation
            perm = torch.randperm(num_nodes, device=data.x.device)
            data.x = data.x[perm]
        else:
            # Partial shuffling: select a fraction of nodes and shuffle them
            num_to_shuffle = int(num_nodes * self.ratio)
            if num_to_shuffle > 1:
                # Get random indices to shuffle
                indices = torch.randperm(num_nodes, device=data.x.device)[:num_to_shuffle]
                # Create a permutation for these indices
                shuffled_indices = indices[torch.randperm(num_to_shuffle, device=data.x.device)]
                # Apply shuffling
                data.x[indices] = data.x[shuffled_indices].clone()
        
        return data

