import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import scatter

class GraphExplainerWrapper(nn.Module):
    """
    Wrapper for GNNExplainer and other attribution methods.
    Assumes that x contains BOTH node features and global macro features concatenated,
    and relies on num_node_features to split them internally.
    
    Note for users of integrated_grad.py / contr_fact.py:
    This unified version splits x internally. If your script passed global_features
    separately before, you must concatenate them into x before calling the explainer,
    or this code will fail/need adaptation.
    """
    def __init__(self, jepa_model, classifier, num_node_features, sigma=1.0):
        super().__init__()
        self.graph_jepa = jepa_model
        for param in self.graph_jepa.parameters():
            param.requires_grad = False
        self.graph_jepa.eval()
        
        self.classifier = classifier
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.classifier.eval()
        
        self.sigma = sigma
        self.num_node_features = num_node_features

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        # We assume x is concatenated: [node_features, global_features]
        x_real = x[:, :self.num_node_features]
        global_feats = x[0, self.num_node_features:].unsqueeze(0)
        
        if edge_attr is not None and edge_attr.numel() > 0:
            if batch is None:
                edge_batch = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
            else:
                edge_batch = batch[edge_index[0]]
            
            min_vals = scatter(edge_attr, edge_batch, dim=0, reduce='min')
            edge_attr_processed = edge_attr - min_vals[edge_batch]
            edge_attr_exp = torch.exp(-edge_attr_processed ** 2 / (self.sigma ** 2 + 1e-6))
        else:
            edge_attr_exp = torch.ones(edge_index.size(1), 1, device=x_real.device, dtype=torch.float32)
            
        graph_emb = self.graph_jepa(x_real, edge_index, edge_attr_exp)
        
        if batch is None:
            batch = torch.zeros(x_real.size(0), dtype=torch.long, device=x_real.device)
            
        graph_emb_pooled = global_add_pool(graph_emb, batch)
        
        combined_features = torch.cat([graph_emb_pooled, global_feats], dim=-1)
        
        return self.classifier(combined_features)
