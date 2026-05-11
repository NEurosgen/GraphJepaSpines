import os
from collections import defaultdict

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import torch

from src.data_utils.stats import extract_macro_features
from src.explainer.utils import setup_explainer_environment
from src.explainer.visuals import feature_name, set_neurips_style
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
    std_x_data = env["std_x"].squeeze().cpu()

    num_node_features = ds[0].x.size(1)

    model_wrapper = GraphExplainerWrapper(
        jepa_model=encoder, 
        classifier=classifier_module.classifier,
        num_node_features=num_node_features,
        sigma=cls_cfg.get("sigma", 1.0)
    ).to(device)

    feature_saliency_sum = defaultdict(lambda: 0.0)
    samples_count = defaultdict(int)
    num_samples_to_explain = cls_cfg.get("num_samples_to_explain", 50)
    
    print(f"Computing Gradient Saliency (Sensitivity Analysis) for {num_samples_to_explain} samples...")
    for i in range(min(len(ds), num_samples_to_explain)):
        data = ds[i].to(device)
        global_feats = extract_macro_features(data, macro_mean, macro_std)
        true_class = data.y.item() if hasattr(data, 'y') else 0 
        target_class = 1 - true_class 
        
        data.x.requires_grad_()
        model_wrapper.zero_grad()
        
        global_feats_broadcasted = global_feats.expand(data.x.size(0), -1)
        x_combined = torch.cat([data.x, global_feats_broadcasted], dim=-1)
        
        logits = model_wrapper(
            x=x_combined, 
            edge_index=data.edge_index, 
            edge_attr=data.edge_attr
        )
        
        target_logit = logits[0, target_class]
        target_logit.backward()
        
        node_attribution = data.x * data.x.grad
        
        graph_saliency = node_attribution.abs().mean(dim=0).cpu()
        
        feature_saliency_sum[true_class] += graph_saliency
        samples_count[true_class] += 1

    os.makedirs("explanations", exist_ok=True)
    set_neurips_style()
    
    for cls_idx in feature_saliency_sum.keys():
        if samples_count[cls_idx] == 0:
            continue
            
        mean_saliency = (feature_saliency_sum[cls_idx] / samples_count[cls_idx]).detach().numpy()
        
        plt.figure(figsize=(10, 6))
        top_k = min(20, len(mean_saliency))
        indices = np.argsort(mean_saliency)[-top_k:] 
        plt.barh(range(top_k), mean_saliency[indices], align='center', color='#1f77b4')
        plt.yticks(range(top_k), [feature_name(idx) for idx in indices])
        
        plt.xlabel('Gradient Sensitivity (Impact of feature change in real units)')
        plt.title(f'Feature Sensitivity for shifting class {cls_idx} -> {1 - cls_idx}')
        plt.tight_layout()
        
        save_path = f"explanations/saliency_real_units_class_{cls_idx}_sph.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Saliency graph for class {cls_idx} saved to: {save_path}")

if __name__ == "__main__":
    main()