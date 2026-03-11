import torch
import torch.nn as nn

class GraphExplainerWrapper(nn.Model):
    def __init__(self, global_features, jepa_model, classifier):
        super().__init__()
        self.graph_jepa = jepa_model
        self.classifier = classifier # MLPка
        self.register_buffer('global_features', global_features)

    def forward(self, x,edge_index, edge_feature):
        graph_emb = self.graph_jepa(x, edge_index, edge_feature)
        combined_features = torch.cat([graph_emb, self.global_features], dim=-1)
        return self.classifier(combined_features)
    

from torch_geometric.explain import Explainer, GNNExplainer

model_wrapper = GraphExplainerWrapper(
    graph_jepa_model=graph_jepa, 
    final_classifier=classifier, 
    global_features=global_feats
)
model_wrapper.eval()


explainer = Explainer(
    model=model_wrapper,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes', # Вычисление важности признаков 
    edge_mask_type='object',     # Вычисление важности связей
    model_config=dict(
        mode='multiclass_classification',
        task_level='graph',
        return_type='log_probs',
    ),
)

explanation = explainer(x=node_features, edge_index=edge_index)