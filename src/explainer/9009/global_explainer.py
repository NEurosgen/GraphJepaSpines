import os
from collections import defaultdict

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer

from src.data_utils.stats import extract_macro_features
from src.explainer.utils import setup_explainer_environment
from src.explainer.visuals import feature_name, set_neurips_style
from src.explainer.models import GraphExplainerWrapper
import matplotlib.patches as mpatches

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
        algorithm=GNNExplainer(epochs=5000),
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
        #### ТУТ ТОЖЕ НАДО ИЗ ПАКЕТА ЗАМЕНИТЬ
        plt.figure(figsize=(10, 6))
        top_k = min(20, len(aggregated_importance))
        indices = np.argsort(aggregated_importance)[-top_k:]
        colors = ['#1f77b4' if idx < num_node_features else '#ff7f0e' for idx in indices]
        
        plt.barh(range(top_k), aggregated_importance[indices], align='center', color=colors)
        plt.yticks(range(top_k), [feature_name(idx, num_node_features) for idx in indices])
        plt.xlabel('Weighted Attribute Importance')
        plt.title(f'Global Feature Importance for Class {cls_idx}')
        
        node_patch = mpatches.Patch(color='#1f77b4', label='Local Node Feature')
        macro_patch = mpatches.Patch(color='#ff7f0e', label='Macro Feature')
        plt.legend(handles=[node_patch, macro_patch], loc='lower right')
        
        plt.tight_layout()
        save_path = f"explanations/global_feature_importance_class_{cls_idx}_sph.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Global feature importance for class {cls_idx} saved to: {save_path}")


if __name__ == "__main__":
    main()