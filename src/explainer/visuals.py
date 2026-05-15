import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
import trimesh
import plotly.graph_objects as go





class DendriteVisualizer:
    def __init__(self, mesh_path=None, graph_data=None):
        self.mesh = None
        self.graph_data = graph_data
        
        if mesh_path and not mesh_path.endswith('.pt'):
            self.load_mesh(mesh_path)
            
    def load_mesh(self, mesh_path):
        self.mesh = trimesh.load(mesh_path)
        return self.mesh
        
    def create_mesh_trace(self, opacity=0.2, color='lightgrey'):
        if self.mesh is None:
            return None
            
        vertices = self.mesh.vertices
        faces = self.mesh.faces
        
        return go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            opacity=opacity,
            color=color,
            name='Dendrite Mesh',
            hoverinfo='none'
        )
        
    def create_graph_traces(self, node_mask=None, edge_mask=None):
        if self.graph_data is None:
            return []
            
        pos = self.graph_data.pos.cpu().numpy()
        x_nodes, y_nodes, z_nodes = pos[:, 0], pos[:, 1], pos[:, 2]
        
        if node_mask is not None:
            if node_mask.dim() > 1:
                node_importance = node_mask.mean(dim=1).cpu().numpy()
            else:
                node_importance = node_mask.cpu().numpy()
                
            if node_importance.size > 0 and node_importance.max() > 0:
                node_importance = node_importance / node_importance.max()
                
            marker_size = 4 + 10 * node_importance
            marker_color = node_importance
            colorscale = 'Reds'
            showscale = True
            texts = [f'Node {i}<br>Imp: {val:.3f}' for i, val in enumerate(node_importance)]
        else:
            marker_size = 5
            marker_color = 'red'
            colorscale = None
            showscale = False
            texts = [f'Node {i}' for i in range(len(pos))]

        nodes_trace = go.Scatter3d(
            x=x_nodes, y=y_nodes, z=z_nodes,
            mode='markers',
            marker=dict(
                size=marker_size,
                color=marker_color,
                colorscale=colorscale,
                showscale=showscale,
                colorbar=dict(title="Node<br>Importance", x=0.85) if showscale else None,
                symbol='circle',
                line=dict(width=0)
            ),
            name='Spine Nodes',
            text=texts,
            hoverinfo='text'
        )
        
        edges = self.graph_data.edge_index.cpu().numpy()
        edge_traces = []
        
        if edge_mask is not None:
            edge_importance = edge_mask.cpu().numpy()
            if edge_importance.size > 0 and edge_importance.max() > 0:
                edge_importance = edge_importance / edge_importance.max()
        else:
            edge_importance = np.ones(edges.shape[1])

        for i in range(edges.shape[1]):
            src, dst = edges[:, i]
            w = edge_importance[i]
            
            if edge_mask is not None and w < 0.05:
                continue
                
            line_width = 1 + 5 * w if edge_mask is not None else 2
            alpha = max(0.2, w) if edge_mask is not None else 1.0
            line_color = f'rgba(255, 0, 0, {alpha})' if edge_mask is not None else 'black'
            
            edge_traces.append(go.Scatter3d(
                x=[pos[src, 0], pos[dst, 0], None],
                y=[pos[src, 1], pos[dst, 1], None],
                z=[pos[src, 2], pos[dst, 2], None],
                mode='lines',
                line=dict(color=line_color, width=line_width),
                hoverinfo='none',
                showlegend=False
            ))
            
        return edge_traces + [nodes_trace]
        
    def plot(self, node_mask=None, edge_mask=None, show=True, html_path=None):
        traces = []
        mesh_trace = self.create_mesh_trace()
        if mesh_trace:
            traces.append(mesh_trace)
            
        graph_traces = self.create_graph_traces(node_mask=node_mask, edge_mask=edge_mask)
        traces.extend(graph_traces)
        
        fig = go.Figure(data=traces)
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data'
            ),
            title="3D Explanation (Important Nodes & Edges)",
            margin=dict(l=0, r=0, b=0, t=30),
            showlegend=False
        )
        
        if html_path:
            fig.write_html(html_path)
        if show:
            fig.show()
        return fig