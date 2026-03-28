import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import scatter

import hydra
from omegaconf import DictConfig

from src.data_utils.datamodule import GraphDataModule, GraphDataSet, make_folder_class_getter
from src.cli.train_model import load_stats, build_transforms
from src.models.loader_model import load_encoder_from_folder
from src.models.classificator import ClassifierLightModule, LinearClassifier
from src.data_utils.transforms import GenNormalize
from src.data_utils.stats import compute_macro_stats, extract_macro_features





class GraphExplainerWrapper(nn.Module):
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
        x_real = x[:, :self.num_node_features]
        global_feats = x[0, self.num_node_features:].unsqueeze(0)
        
        # ─── Edge Attribute Transformation ───
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
            
        # Пропускаем через GNN только реальные признаки узлов
        graph_emb = self.graph_jepa(x_real, edge_index, edge_attr_exp)
        
        if batch is None:
            batch = torch.zeros(x_real.size(0), dtype=torch.long, device=x_real.device)
            
        graph_emb_pooled = global_add_pool(graph_emb, batch)
        
        combined_features = torch.cat([graph_emb_pooled, global_feats], dim=-1)
        
        return self.classifier(combined_features)


def _simple_collate(data_list):
    """Collate for classification — no masking, just batch graphs."""
    return Batch.from_data_list(data_list)


def _find_latest_checkpoint(path):
    """Finds the latest .ckpt file in a directory or returns the path if it's already a file."""
    import os
    import glob
    if os.path.isfile(path):
        return path
    
    ckpt_dir = os.path.join(path, "checkpoints")
    if not os.path.exists(ckpt_dir):
        ckpt_dir = path
        
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpt_files:
        return None
    
    return max(ckpt_files, key=os.path.getmtime)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    cls_cfg = cfg.classifier
    dm_cfg = cfg.datamodule
    
    # Use paths from config instead of hardcoded strings
    path_to_classifier_dir = cls_cfg.get("classifier_checkpoint_path", None)
    path_to_lejepa_dir = cls_cfg.get("checkpoint_path", None)

    if not path_to_classifier_dir or not path_to_lejepa_dir:
        print("Warning: Missing checkpoint paths in config. Falling back to default log locations...")
        path_to_classifier_dir = "lightning_logs/classifier/version_65"
        path_to_lejepa_dir = "lightning_logs/jepa/version_32"

    path_to_classifier = _find_latest_checkpoint(path_to_classifier_dir)
    path_to_lejepa = path_to_lejepa_dir # load_encoder_from_folder handles folder

    if not path_to_classifier:
        raise FileNotFoundError(f"No checkpoint found in {path_to_classifier_dir}")

    # Load statistics and transforms
    mean_x, std_x, mean_edge, std_edge = load_stats(cls_cfg.stats_path)
    transforms = build_transforms(dm_cfg, mean_x, std_x, mean_edge, std_edge)
    gen_normalize = GenNormalize(transforms=transforms, mask_transform=None)

    folder_to_label = dict(cls_cfg.get("folder_to_label", {"ab": 0, "wt": 1}))
    get_class = make_folder_class_getter(folder_to_label)

    # Dataset initialization
    ds = GraphDataSet(
        path=cls_cfg.path,
        get_class=get_class,
        transform=gen_normalize,
    )

    # Initialize models using standardized loader
    encoder = load_encoder_from_folder(path_to_lejepa)
    
    num_classes = cls_cfg.get("num_classes", 2)
    # Dynamically determine dimensions if possible, or use config
    embed_dim = cfg.network.encoder.out_channels + 7 
    classifier_head = LinearClassifier(in_channels=embed_dim, num_classes=num_classes)
    
    classifier_module = ClassifierLightModule.load_from_checkpoint(
        path_to_classifier, 
        encoder_graph=nn.Identity(), 
        classifier=classifier_head,
        strict=False,
        weights_only=False
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier_module.to(device)


    print("Computing dynamic macro statistics for dataset...")
    macro_mean, macro_std = compute_macro_stats(ds)

    # Select a sample graph for explanation
    sample_idx = cls_cfg.get("explain_sample_idx", 0)
    data = ds[sample_idx].to(device)
    
    # Extract macro features
    global_feats = extract_macro_features(data, macro_mean, macro_std) # Формат: [1, num_macro_features]

    # Сохраняем исходное количество признаков узла
    num_node_features = data.x.size(1)

    # Дублируем макро-признаки для каждого узла графа
    global_feats_broadcasted = global_feats.expand(data.x.size(0), -1)
    
    # Конкатенируем локальные признаки узлов и макро-признаки
    x_combined = torch.cat([data.x, global_feats_broadcasted], dim=-1)

    model_wrapper = GraphExplainerWrapper(
        jepa_model=encoder, 
        classifier=classifier_module.classifier,
        num_node_features=num_node_features,
        sigma=cls_cfg.get("sigma", 1.0)
    ).to(device)
    
    # Configure GNN explainer
    explainer = Explainer(
        model=model_wrapper,
        algorithm=GNNExplainer(epochs=3000),
        explanation_type='model',
        node_mask_type='attributes',   # Будет вычислять важность для x_combined
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw',
        ),
    )

    print(f"Explaining sample {sample_idx}...")
    # Передаем объединенный тензор x_combined
    explanation = explainer(x=x_combined, edge_index=data.edge_index, edge_attr=data.edge_attr)
    print("Explanation computed successfully!")
    
    import os
    os.makedirs("explanations", exist_ok=True)
    
    # График важности признаков
    # На графике признаки с индексами от 0 до num_node_features-1 — это признаки узлов.
    # Признаки с индексами от num_node_features и выше — это макро-признаки.
    explanation.visualize_feature_importance(
        top_k=20, 
        path="explanations/feature_importance_sph.png"
    )
    print("Feature importance graph saved to: explanations/feature_importance.png")


if __name__ == "__main__":
    main()