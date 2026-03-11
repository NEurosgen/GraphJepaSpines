import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool
from torch_geometric.explain import Explainer, GNNExplainer

import hydra
from omegaconf import DictConfig

from src.data_utils.datamodule import GraphDataModule, GraphDataSet, make_folder_class_getter
from src.cli.train_model import load_stats, build_transforms
from src.cli.classifier import _load_encoder_from_checkpoint, LinearClassifier, ClassifierLightModule
from src.data_utils.transforms import GenNormalize


class GraphExplainerWrapper(nn.Module):
    def __init__(self, global_features, jepa_model, classifier, sigma=1.0):
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
        self.register_buffer('global_features', global_features.view(1, -1))

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        if edge_attr is not None:
            edge_attr_exp = torch.exp(-edge_attr**2 / self.sigma**2)
        else:
            edge_attr_exp = torch.ones(edge_index.size(1), 1, device=x.device, dtype=torch.float32)
            
        graph_emb = self.graph_jepa(x, edge_index, edge_attr_exp)
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        # Add pooling for JEPA node embeddings
        graph_emb_pooled = global_add_pool(graph_emb, batch)
        
        global_feats = self.global_features.expand(graph_emb_pooled.size(0), -1)
        combined_features = torch.cat([graph_emb_pooled, global_feats], dim=-1)
        
        return self.classifier(combined_features)


def extract_macro_features(data, macro_mean, macro_std):
    """Computes and normalizes macro parameters for the dataset graph"""
    if hasattr(data, 'macro_metrics') and data.macro_metrics is not None:
        if data.macro_metrics.dim() == 2:
            macro_features = data.macro_metrics.mean(dim=0, keepdim=True)
        else:
            macro_features = data.macro_metrics.view(1, -1)
            
        if macro_mean is not None and macro_std is not None:
            macro_mean = macro_mean.to(macro_features.device)
            macro_std = macro_std.to(macro_features.device)
            macro_features = (macro_features - macro_mean) / (macro_std + 1e-6)
            
    else:
        macro_features = torch.zeros((1, 7), dtype=torch.float32, device=data.x.device)
        
    return macro_features


def compute_macro_stats(dataset, max_samples=2000):
    """Computes mean and std of macro_metrics dynamically over the dataset."""
    import random
    all_macros = []
    indices = list(range(len(dataset)))
    if len(indices) > max_samples:
        indices = random.sample(indices, max_samples)
        
    # We do a quick pass without tqdm to avoid extra dependencies being printed
    for i in indices:
        data = dataset[i]
        if hasattr(data, 'macro_metrics') and data.macro_metrics is not None:
            mac = data.macro_metrics
            if mac.dim() == 2:
                mac = mac.mean(dim=0, keepdim=True)
            else:
                mac = mac.view(1, -1)
            all_macros.append(mac.cpu())
            
    if not all_macros:
        return None, None
        
    all_macros = torch.cat(all_macros, dim=0) # [N, 7]
    macro_mean = all_macros.mean(dim=0, keepdim=True)
    macro_std = all_macros.std(dim=0, keepdim=True)
    return macro_mean, macro_std


def _simple_collate(data_list):
    """Collate for classification — no masking, just batch graphs."""
    return Batch.from_data_list(data_list)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    cls_cfg = cfg.classifier
    dm_cfg = cfg.datamodule
    
    path_to_classifier = "/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/lightning_logs/version_540/checkpoints/classifier-epoch=50-val_acc=0.9231.ckpt"
    path_to_lejepa = "/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/lightning_logs/version_533/checkpoints/epoch=199-step=18000.ckpt"

    # Load statistics and transforms
    mean_x, std_x, mean_edge, std_edge = load_stats(cls_cfg.stats_path)
    transforms = build_transforms(dm_cfg, mean_x, std_x, mean_edge, std_edge)
    gen_normalize = GenNormalize(transforms=transforms, mask_transform=None)

    class_names = list(cls_cfg.get("class_names", ["ab", "wt"]))
    folder_to_label = dict(cls_cfg.get("folder_to_label", {"ab": 0, "wt": 1}))
    get_class = make_folder_class_getter(folder_to_label)

    # Dataset initialization
    ds = GraphDataSet(
        path=cls_cfg.path,
        get_class=get_class,
        transform=gen_normalize,
    )

    # Initialize models
    encoder = _load_encoder_from_checkpoint(path_to_lejepa, cfg)
    
    num_classes = cls_cfg.get("num_classes", 2)
    embed_dim = cfg.network.encoder.out_channels + 7
    classifier_head = LinearClassifier(in_channels=embed_dim, num_classes=num_classes)
    
    classifier_module = ClassifierLightModule.load_from_checkpoint(
        path_to_classifier, 
        encoder=encoder, 
        classifier=classifier_head,
        weights_only=False
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier_module.to(device)

    # Calculate dataset macro stats for dynamic normalization
    print("Computing dynamic macro statistics for dataset...")
    macro_mean, macro_std = compute_macro_stats(ds)

    # Select the first element in dataset
    data = ds[5].to(device)
    
    # Extract macro features (global_features) dynamically normalized
    global_feats = extract_macro_features(data, macro_mean, macro_std)

    model_wrapper = GraphExplainerWrapper(
        global_features=global_feats, 
        jepa_model=classifier_module.encoder, 
        classifier=classifier_module.classifier,
        sigma=cls_cfg.get("sigma", 1.0)
    ).to(device)
    
    # Configure GNN explainer
    explainer = Explainer(
        model=model_wrapper,
        algorithm=GNNExplainer(epochs=100),
        explanation_type='model',
        node_mask_type='attributes',   # compute node attribute importance
        edge_mask_type='object',       # compute edge importance
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw',  # LinearClassifier returns raw logits
        ),
    )

    # Provide data arguments to explain
    explanation = explainer(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
    print("Explanation computed successfully!")
    import os
    os.makedirs("explanations", exist_ok=True)
    
    # 1. Построить график важности топ-10 признаков узлов
    explanation.visualize_feature_importance(
        top_k=20, 
        path="explanations/feature_importance.png"
    )
    print("Feature importance graph saved to: explanations/feature_importance.png")
    
    # 2. Если в графе есть ребра, можно нарисовать сам граф и подсветить важные узлы
    if data.edge_index.shape[1] > 0:
        explanation.visualize_graph(
            path="explanations/graph_explanation.png"
        )
        print("Graph visualization saved to: explanations/graph_explanation.png")


if __name__ == "__main__":
    main()