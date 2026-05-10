import os
import torch
import hydra
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from omegaconf import DictConfig
from torch_geometric.explain import Explainer, GNNExplainer

from src.data_utils.stats import extract_macro_features
from src.explainer.utils import setup_explainer_environment, load_mesh_vertices
from src.explainer.visuals import feature_name, set_neurips_style, plot_custom_graph
from src.explainer.models import GraphExplainerWrapper

@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
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
    print(data)
    print(sample_idx)
    print("pos shape:", data.pos.shape)
    print("pos sample:", data.pos[:3])
    print("pos min/max:", data.pos.min(0).values, data.pos.max(0).values)

    global_feats = extract_macro_features(data, macro_mean, macro_std)
    num_node_features = data.x.size(1)
    global_feats_broadcasted = global_feats.expand(data.x.size(0), -1)
    
    x_combined = torch.cat([data.x, global_feats_broadcasted], dim=-1)

    model_wrapper = GraphExplainerWrapper(
        jepa_model=encoder, 
        classifier=classifier_module.classifier,
        num_node_features=num_node_features,
        sigma=cls_cfg.get("sigma", 1.0)
    ).to(device)
    
    explainer = Explainer(
        model=model_wrapper,
        algorithm=GNNExplainer(epochs=5000),
        explanation_type='model',
        node_mask_type='attributes',   
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw',
        ),
    )

    print(f"Explaining sample {sample_idx}...")
    explanation = explainer(x=x_combined, edge_index=data.edge_index, edge_attr=data.edge_attr)
    print("Explanation computed successfully!")
    
    os.makedirs("explanations", exist_ok=True)
    set_neurips_style()
    
    feat_importance = explanation.node_mask.mean(dim=0).cpu().numpy()
    top_k = min(20, len(feat_importance))
    indices = np.argsort(feat_importance)[-top_k:]
    
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4' if idx < num_node_features else '#ff7f0e' for idx in indices]
    plt.barh(range(top_k), feat_importance[indices], align='center', color=colors)
    plt.yticks(range(top_k), [feature_name(idx, num_node_features) for idx in indices])
    plt.xlabel('Attribute Importance')
    plt.title('Feature Importance')
    
    node_patch = mpatches.Patch(color='#1f77b4', label='Local Node Feature')
    macro_patch = mpatches.Patch(color='#ff7f0e', label='Macro Feature')
    plt.legend(handles=[node_patch, macro_patch], loc='lower right')
    
    plt.tight_layout()
    plt.savefig("explanations/feature_importance_sph.png")
    plt.close()
    print("Feature importance graph saved to: explanations/feature_importance_sph.png")
    
    mesh_path = "/home/eugen/Desktop/CodeWork/Projects/Diplom/Archiv/processed_dataset/ab/ab24-1_processed.pt"
    if os.path.exists(mesh_path):
        mesh_points = load_mesh_vertices(mesh_path)
    else:
        mesh_points = None
        
    plot_custom_graph(
        data=data,
        node_mask=None,
        edge_mask=None,
        save_path="explanations/original_graph_sph.png",
        title="Original Graph Structure",
        mesh_data=mesh_points
    )
    print("Original graph saved to: explanations/original_graph_sph.png")

    plot_custom_graph(
        data=data,
        node_mask=explanation.node_mask,
        edge_mask=explanation.edge_mask,
        save_path="explanations/graph_explanation_sph.png",
        title="Explanation (Important Nodes & Edges)",
        mesh_data=mesh_points
    )
    print("Graph explanation saved to: explanations/graph_explanation_sph.png")

if __name__ == "__main__":
    main()