import os
import torch
import hydra
from omegaconf import DictConfig
from torch_geometric.explain import Explainer, GNNExplainer
from src.data_utils.stats import extract_macro_features
from src.explainer.utils import setup_explainer_environment
from src.explainer.models import GraphExplainerWrapper
from src.data_utils.datamodule import GraphDataSet
from src.explainer.visuals import DendriteVisualizer
from src.cli.inference.d9009.evaluate_encoder_cv import get_class_9009

def get_mesh_path(mesh_dataset_path, data_path):
    return os.path.join(mesh_dataset_path, data_path.split("/")[0], "output", data_path.split("/")[1][:-3], "surface_mesh.off")
@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig):
    cls_cfg = cfg.classifier
    
   

    env = setup_explainer_environment(cfg , get_class = get_class_9009)
    ds = env["dataset"]
    encoder = env["encoder"]
    classifier_module = env["classifier"]
    macro_mean = env["macro_mean"]
    macro_std = env["macro_std"]
    device = env["device"]
    
    sample_idx = cls_cfg.get("explain_sample_idx", 0)

    data = ds[sample_idx].to(device)
    
    begin_pos  = GraphDataSet(
        path=cls_cfg.raw_path 
    )[sample_idx].to(device).pos
    print(begin_pos.shape)
    print(data.pos.shape)
    data.pos = begin_pos



    global_feats = extract_macro_features(data, macro_mean, macro_std)
    num_node_features = data.x.size(1)
    global_feats_broadcasted = global_feats.expand(data.x.size(0), -1)
    
    x_combined = torch.cat([data.x, global_feats_broadcasted], dim=-1)
 

    model_wrapper = GraphExplainerWrapper(
        jepa_model=encoder, 
        classifier=classifier_module,
        num_node_features=num_node_features,
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
    

    mesh_geometry_path = os.path.join(cls_cfg.mesh_dataset_path, str(data.path).split("/")[0], "output", str(data.path).split("/")[1][:-3], "surface_mesh.off")
    visualizer = DendriteVisualizer(
        mesh_path=mesh_geometry_path if os.path.exists(mesh_geometry_path) else None, 
        graph_data=data
    )

    html_out_path = "explanations/graph_explanation_sph_3d.html"
    visualizer.plot(
        node_mask=explanation.node_mask,
        edge_mask=explanation.edge_mask,
        show=False,
        html_path=html_out_path
    )
    
    print(f"3D Feature importance graph saved to: {html_out_path}")

if __name__ == "__main__":
    main()