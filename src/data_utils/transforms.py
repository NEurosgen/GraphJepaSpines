import torch_geometric
import torch
import torch.nn as nn
from torch_geometric.nn import knn_graph


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
        



import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, k_hop_subgraph, to_undirected

class GraphPruning(torch.nn.Module):
    '''
    Return pruned graph with knn

    '''
    def __init__(self, k = -1, mutual = False):
        super().__init__()
        self.k = k
        self.mutual = mutual
    def forward(self, data):
        if self.k < 0:
            return data
        batch = data.batch
        knn_edge_index = knn_graph(data.pos,self.k,batch,loop=False)
        row,col = knn_edge_index
        num_nodes = data.num_nodes
        if self.mutual:
            knn_hashes = row*num_nodes + col
            knn_hashes_rev = col*num_nodes + row
            is_mutual = torch.isin(knn_hashes,knn_hashes_rev)
            valid_hashes = knn_hashes[is_mutual]
        else:
            valid_hashes = row*num_nodes + col
        curr_row, curr_col = data.edge_index
        curr_hashes = curr_row * num_nodes + curr_col
        mask = torch.isin(curr_hashes, valid_hashes)
        new_edge_index = data.edge_index[:, mask]
        new_edge_attr = None
        if data.edge_attr is not None:
            new_edge_attr = data.edge_attr[mask]
        return Data(
            x=data.x,
            edge_index=new_edge_index,
            edge_attr=new_edge_attr,
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

    def _get_random_patch_mask(self, data: Data) -> torch.Tensor:
        """Optimized mask generation using estimated hop count."""
        num_nodes = data.num_nodes
        num_mask_goal = max(1, int(num_nodes * self.mask_ratio))
        
        device = data.x.device if data.x is not None else 'cpu'
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        

        num_edges = data.edge_index.size(1)
        avg_degree = num_edges / (num_nodes + 1e-6)
        

        import math
        if avg_degree > 1:
            estimated_hops = max(1, int(math.log(num_mask_goal + 1) / math.log(avg_degree + 1e-6)))
        else:
            estimated_hops = min(num_mask_goal, 4)
        estimated_hops = min(estimated_hops, 6)  
        
        start_node = torch.randint(0, num_nodes, (1,)).item()
        
        subset, _, _, _ = k_hop_subgraph(
            node_idx=start_node,
            num_hops=estimated_hops,
            edge_index=data.edge_index,
            relabel_nodes=False,
            num_nodes=num_nodes
        )
        
        if len(subset) < num_mask_goal and estimated_hops < 6:
            subset, _, _, _ = k_hop_subgraph(
                node_idx=start_node,
                num_hops=estimated_hops + 1,
                edge_index=data.edge_index,
                relabel_nodes=False,
                num_nodes=num_nodes
            )
        
        selected = subset[:num_mask_goal] if len(subset) > num_mask_goal else subset
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
        mask = self._get_random_patch_mask(data)
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

