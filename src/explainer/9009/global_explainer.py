import os
from collections import defaultdict

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer
from paper_plots import plot_feature_importance
from src.data_utils.stats import extract_macro_features
from src.explainer.utils import setup_explainer_environment
from src.explainer.models import GraphExplainerWrapper

@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig):
    cls_cfg = cfg.classifier
    
    env = setup_explainer_environment(cfg)
    ds = env["dataset"]
    encoder = env["encoder"]
    classifier_module = env["classifier_module"]
    macro_mean = env["macro_mean"]
    macro_std = env["macro_std"]
    device = env["device"]

    sample_idx = cls_cfg.get("explain_sample_idx", 0)
    data = ds[sample_idx].to(device)
    num_node_features = data.x.size(1)

    model_wrapper = GraphExplainerWrapper(
        jepa_model=encoder, 
        classifier=classifier_module.classifier,
        sigma=cls_cfg.get("sigma", 1.0),
        num_node_features=num_node_features
    ).to(device)
    
    explainer = Explainer(
        model=model_wrapper,
        algorithm=GNNExplainer(epochs=5),
        explanation_type='model',
        node_mask_type='attributes',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw',
        ),
    )

    class_feature_importances = defaultdict(lambda: 0.0)
    class_weights_sum = defaultdict(lambda: 0.0)
    
    num_samples_to_explain = cls_cfg.get("num_samples_to_explain", 50)
    
    print(f"Aggregating explanations over {num_samples_to_explain} samples...")
    for i in range(min(len(ds), num_samples_to_explain)):
        data = ds[i].to(device)
        global_feats = extract_macro_features(data, macro_mean, macro_std)
        
        global_feats_broadcasted = global_feats.expand(data.x.size(0), -1)
        x_combined = torch.cat([data.x, global_feats_broadcasted], dim=-1)
        
        with torch.no_grad():
            logits = model_wrapper(
                x=x_combined, 
                edge_index=data.edge_index, 
                edge_attr=data.edge_attr
            )
            probs = F.softmax(logits, dim=-1)
            predicted_class = probs.argmax(dim=-1).item()
            confidence = probs[0, predicted_class].item()
            
        target = torch.tensor([predicted_class], device=device)
            
        explanation = explainer(
            x=x_combined, 
            edge_index=data.edge_index, 
            target=target,
            edge_attr=data.edge_attr
        )
        
        graph_feat_importance = explanation.node_mask.mean(dim=0).cpu()
        
        class_feature_importances[predicted_class] += graph_feat_importance * confidence
        class_weights_sum[predicted_class] += confidence
        
    os.makedirs("explanations", exist_ok=True)
    set_neurips_style()
    
    for cls_idx in class_feature_importances.keys():
        if class_weights_sum[cls_idx] == 0:
            continue
            
        aggregated_importance = class_feature_importances[cls_idx] / class_weights_sum[cls_idx]
        aggregated_importance = aggregated_importance.numpy()
        save_path_base = f"explanations/global_feature_importance_class_{cls_idx}_sph"
        plot_feature_importance(
            importance_scores=aggregated_importance,
            feature_names=lambda idx: feature_name(idx, num_node_features),
            top_k=20,
            color_threshold=num_node_features,
            color_labels=('Local Node Feature', 'Macro Feature'),
            xlabel='Weighted Attribute Importance',
            title=f'Global Feature Importance for Class {cls_idx}',
            save_path=save_path_base,
            close=True
        )
        print(f"Global feature importance for class {cls_idx} saved to: {save_path_base}.png")


if __name__ == "__main__":
    main()