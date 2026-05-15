import os
from collections import defaultdict
from paper_plots import plot_feature_importance
import hydra
from omegaconf import DictConfig
import torch
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
    std_x_data = env["std_x"].squeeze().cpu()

    num_node_features = ds[0].x.size(1)

    model_wrapper = GraphExplainerWrapper(
        jepa_model=encoder, 
        classifier=classifier_module.classifier,
        num_node_features=num_node_features,
        sigma=cls_cfg.get("sigma", 1.0)
    ).to(device)

    feature_ig_sum = defaultdict(lambda: 0.0)
    samples_count = defaultdict(int)
    num_samples_to_explain = cls_cfg.get("num_samples_to_explain", 50)
    
    m_steps = 50 
    
    print(f"Computing Integrated Gradients for {num_samples_to_explain} samples...")
    for i in range(min(len(ds), num_samples_to_explain)):
        data = ds[i].to(device)
        global_feats = extract_macro_features(data, macro_mean, macro_std)
        true_class = data.y.item() if hasattr(data, 'y') else 0 
        target_class = 1 - true_class 
        
        baseline_x = torch.zeros_like(data.x, device=device)
        integrated_grads = torch.zeros_like(data.x, device=device)
        
        global_feats_broadcasted = global_feats.expand(data.x.size(0), -1)
        
        for k in range(1, m_steps + 1):
            alpha = k / m_steps
            interpolated_x = baseline_x + alpha * (data.x - baseline_x)
            interpolated_x.requires_grad_()
            
            model_wrapper.zero_grad()
            
            x_combined = torch.cat([interpolated_x, global_feats_broadcasted], dim=-1)
            
            logits = model_wrapper(
                x=x_combined, 
                edge_index=data.edge_index, 
                edge_attr=data.edge_attr
            )
            
            target_logit = logits[0, target_class]
            target_logit.backward()
            
            integrated_grads += interpolated_x.grad
            
        avg_grads = integrated_grads / m_steps
        ig = (data.x - baseline_x) * avg_grads
        
        node_ig_importance = ig.abs().mean(dim=0).cpu()
        
        std_padded = torch.ones_like(node_ig_importance)
        num_stats = min(std_x_data.size(0), node_ig_importance.size(0))
        std_padded[:num_stats] = std_x_data[:num_stats]
        
        node_ig_real = node_ig_importance * std_padded
        
        feature_ig_sum[true_class] += node_ig_real
        samples_count[true_class] += 1
        
    os.makedirs("explanations", exist_ok=True)
    set_neurips_style()
    
    for cls_idx in feature_ig_sum.keys():
        if samples_count[cls_idx] == 0:
            continue
            
        mean_ig = (feature_ig_sum[cls_idx] / samples_count[cls_idx]).numpy()
        ## И ТУТ
        save_path_base = f"explanations/ig_real_units_class_{cls_idx}"
        plot_feature_importance(
            importance_scores=mean_ig,
            feature_names=feature_name,
            top_k=20,
            xlabel='Integrated Gradient Attribution (Real Units Scale)',
            title=f'IG Feature Attribution for shifting class {cls_idx} -> {1 - cls_idx}',
            save_path=save_path_base,
            close=True
        )
        print(f"IG attribution graph for class {cls_idx} saved to: {save_path_base}.png")


if __name__ == "__main__":
    main()