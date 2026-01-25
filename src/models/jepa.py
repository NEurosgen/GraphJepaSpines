import torch
from torch import nn



from torch_geometric.data import Data
from collections import deque
import random
from torch_geometric.utils import subgraph


class MaskData(nn.Module):
    def __init__(self, mask_ratio):
        super().__init__()
        self.mask_ratio = mask_ratio
    def _get_bfs_mask(self, data: Data, mask_ratio: float = 0.15) -> torch.Tensor:
        """
        Generate a BFS-based mask for the graph.
        Masked nodes form a contiguous region, simulating realistic occlusion.
        """
        num_nodes = data.num_nodes
        device = data.x.device if data.x is not None else data.edge_index.device
        
        num_mask = max(1, int(num_nodes * mask_ratio))
        adj_list = [[] for _ in range(num_nodes)]
        row, col = data.edge_index
        for i in range(row.size(0)):
            u, v = row[i].item(), col[i].item()
            adj_list[u].append(v)
        
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        visited = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        current_masked_count = 0
        q = deque()

        while current_masked_count < num_mask:
            if not q:
                unmasked_indices = (~mask).nonzero(as_tuple=False).view(-1)
                if len(unmasked_indices) == 0:
                    break
                start_node = unmasked_indices[random.randint(0, len(unmasked_indices) - 1)].item()
                q.append(start_node)
                visited[start_node] = True
                mask[start_node] = True
                current_masked_count += 1
            
            if current_masked_count >= num_mask:
                break
            
            u = q.popleft()
            neighbors = adj_list[u]
            random.shuffle(neighbors)
            
            for v in neighbors:
                if not visited[v]:
                    visited[v] = True
                    mask[v] = True
                    current_masked_count += 1
                    q.append(v)
                    if current_masked_count >= num_mask:
                        break
        
        return mask

    def _split_data_by_mask(self, data, mask):
        # Handle pos if available
        pos_ctx = data.pos[~mask] if data.pos is not None else None
        pos_tgt = data.pos[mask] if data.pos is not None else None
        
        # Context (Visible)
        subset_ctx = ~mask
        edge_index_ctx, edge_attr_ctx = subgraph(
            subset_ctx, data.edge_index, edge_attr=data.edge_attr, relabel_nodes=True, num_nodes=data.num_nodes
        )
        batch_ctx = data.batch[subset_ctx] if hasattr(data, 'batch') and data.batch is not None else None
        context_data = Data(
            x=data.x[subset_ctx], pos=pos_ctx,
            edge_index=edge_index_ctx, edge_attr=edge_attr_ctx, batch=batch_ctx
        )

        # Target (Masked)
        subset_tgt = mask
        batch_tgt = data.batch[subset_tgt] if hasattr(data, 'batch') and data.batch is not None else None

        edge_index_tgt, edge_attr_tgt = subgraph(
            subset_tgt, data.edge_index, edge_attr=data.edge_attr, relabel_nodes=True, num_nodes=data.num_nodes
        )
        target_data = Data(
            x=data.x[subset_tgt], pos=pos_tgt, edge_index=edge_index_tgt, edge_attr=edge_attr_tgt, 
            batch=batch_tgt
        )
        return context_data, target_data
        
    def forward(self, data):
        mask = self._get_bfs_mask(data , self.mask_ratio)
        context_data, target_data  = self._split_data_by_mask(data , mask)
        return context_data, target_data


class CrossAttentionPredictor(nn.Module):
    """
    Predictor that uses cross-attention to predict target node embeddings
    based on context node embeddings and positions.
    
    Query: target positions
    Key/Value: context embeddings + positions
    """
    def __init__(self, hidden_dim: int, pos_dim: int = 3, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Project positions to embedding space
        self.pos_embed = nn.Linear(pos_dim, hidden_dim)
        
        # Cross-attention: query = target positions, key/value = context
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # MLP head for final prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, context_emb: torch.Tensor, context_pos: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_emb: [num_context, hidden_dim] - context node embeddings
            context_pos: [num_context, pos_dim] - context node positions
            target_pos: [num_target, pos_dim] - target node positions
        Returns:
            pred: [num_target, hidden_dim] - predicted embeddings for target nodes
        """
        # Normalize positions (zero-mean, unit-variance) to handle varying scales
        all_pos = torch.cat([context_pos, target_pos], dim=0)
        pos_mean = all_pos.mean(dim=0, keepdim=True)
        pos_std = all_pos.std(dim=0, keepdim=True).clamp(min=1e-6)
        
        context_pos_norm = (context_pos - pos_mean) / pos_std
        target_pos_norm = (target_pos - pos_mean) / pos_std
        
        # Combine context embeddings with positional information
        context_kv = context_emb + self.pos_embed(context_pos_norm)
        
        # Query from target positions
        target_query = self.pos_embed(target_pos_norm)
        
        # Cross-attention (add batch dimension for nn.MultiheadAttention)
        target_query = target_query.unsqueeze(0)
        context_kv = context_kv.unsqueeze(0)
        
        attn_out, _ = self.cross_attn(
            query=target_query,
            key=context_kv,
            value=context_kv
        )
        attn_out = attn_out.squeeze(0)
        
        # Residual + LayerNorm + MLP
        x = self.norm1(attn_out)
        x = x + self.mlp(x)
        x = self.norm2(x)
        
        return x






def sigreg(x: torch.Tensor, global_step: int, seed: int = 42, num_slices: int = 256) -> torch.Tensor:
    assert x.dim() == 2, f"sigreg expects [N, K], got {x.shape}"
    N, K = x.shape
    dev = dict(device=x.device, dtype=x.dtype)
    g = torch.Generator(device=x.device)
    g.manual_seed(seed)
    A = torch.randn(K, num_slices, generator=g, device=x.device, dtype=x.dtype)
    A /= A.norm(p=2, dim=0, keepdim=True)
    t = torch.linspace(-5.0, 5.0, 17, **dev)
    phi_gauss = torch.exp(-0.5 * t**2) 
    z = x @ A
    z_t = z.unsqueeze(-1) * t 
    z_t_complex = z_t.to(torch.complex64)
    ecf = (1j * z_t_complex).exp().mean(dim=0)

    N_eff = N

 
    err = (ecf - phi_gauss.unsqueeze(0)).abs().square() * phi_gauss.unsqueeze(0)

    T_vals = torch.trapz(err, t, dim=-1) * N_eff
    T_mean = T_vals.mean()

    return T_mean


class LeJEPA(nn.Module): # DEFECTED MODULE NEED TO REFACTOR
    def __init__(self, encoder: nn.Module, lambd: float, num_slices: int = 256): 

        super().__init__()
        raise "Review code, it's not correct"
        self.encoder = encoder
        self.lambd = float(lambd)
        self.num_slices = num_slices

    def forward(self, global_views, all_views, global_step: int):

        batch_size = global_views[0].shape[0]

        g_emb = self.encoder(torch.cat(global_views, dim=0))
        a_emb = self.encoder(torch.cat(all_views, dim=0))

        assert g_emb.dim() == 2 and a_emb.dim() == 2, \
            "Encoder is expected to output [N, K] embeddings."

        K = g_emb.shape[-1]
        num_global = len(global_views)
        num_all = len(all_views)

        g_emb = g_emb.view(num_global, batch_size, K)
        a_emb = a_emb.view(num_all, batch_size, K)
        centers = g_emb.mean(dim=0)

        sim_loss = (centers.unsqueeze(0) - a_emb).pow(2).mean() 


        sigreg_loss = sigreg(
            a_emb.view(-1, K),
            global_step=global_step,
            num_slices=self.num_slices,
        )

        loss = (1.0 - self.lambd) * sim_loss + self.lambd * sigreg_loss
        return loss
from typing import Callable, Optional
import copy





class GraphJepa(nn.Module):
    def __init__(
        self, 
        encoder: nn.Module, 
        predictor: nn.Module,
        mask_function = None,
        ema: float = 0.996,
        mask_ratio: float = 0.3
    ):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        self.ema = ema
        
        self.mask_function = mask_function if mask_function is not None else MaskData(mask_ratio=mask_ratio)
        
        if hasattr(encoder, 'proj'):
            in_channels = encoder.proj.in_features
        elif hasattr(encoder, 'in_channels'):
            in_channels = encoder.in_channels
        else:
            in_channels = 128
        
        self.mask_token = nn.Parameter(torch.zeros(1, in_channels))
        nn.init.normal_(self.mask_token, std=0.02)
        

        self.student_encoder = encoder
        self.teach_encoder = copy.deepcopy(encoder)
        for p in self.teach_encoder.parameters():
            p.requires_grad = False
        
        self.predictor = predictor
        self.loss_fn = nn.MSELoss()
    
    @torch.no_grad()
    def _ema(self):
        """Exponential moving average update for teacher encoder."""
        for params_s, params_t in zip(
            self.student_encoder.parameters(), 
            self.teach_encoder.parameters()
        ):
            params_t.data.mul_(self.ema).add_((1 - self.ema) * params_s.data)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, pos: torch.Tensor, edge_weight: Optional[torch.Tensor] = None):
        context, target = self.mask_function(Data(x=x, edge_index=edge_index, edge_attr=edge_weight, pos=pos))
        
        # Encode context with student encoder
        context_enc = self.student_encoder(context.x, context.edge_index, context.edge_attr)
        
        # Encode target with teacher encoder (no gradients)
        with torch.no_grad():
            teacher_enc = self.teach_encoder(target.x, target.edge_index, target.edge_attr)
        
        # Predictor uses context embeddings + positions to predict target embeddings
        pred = self.predictor(
            context_emb=context_enc,
            context_pos=context.pos,
            target_pos=target.pos
        )
        
        loss = self.loss_fn(pred, teacher_enc.detach())
        
        return loss
import pytorch_lightning as L

class JepaLight(L.LightningModule):
    def __init__(self, model: GraphJepa, cfg, debug: bool = False):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.debug = debug
        self.model = model
        
        # Store cfg parameters for optimizer configuration
        self.learning_rate = cfg.learning_rate
        self.optimizer_cfg = cfg.optimizer
        self.scheduler_cfg = cfg.get('scheduler', None)
        self.sigma = 1
    
    def _debug_log(self, batch):
        with torch.no_grad():
            z = self.model.teach_encoder(batch.x, batch.edge_index)
            std = z.std(dim=0).mean()
            norm = z.norm(dim=-1).mean()
        
        self.log("debug_z_std", std, prog_bar=True)
        self.log("debug_z_norm", norm, prog_bar=True)
    
    def training_step(self, batch):
        edge_weight = torch.exp(-batch.edge_attr**2 / self.sigma**2)
        loss = self.model(batch.x, batch.edge_index, batch.pos, edge_weight)
        self.log("train_loss", loss, prog_bar=True)
        if self.debug:
            self._debug_log(batch)
        return loss
    
    def validation_step(self, batch):
        edge_weight = torch.exp(-batch.edge_attr**2 / self.sigma**2)
        loss = self.model(batch.x, batch.edge_index, batch.pos, edge_weight)
        if self.debug:
            self._debug_log(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.model._ema()
    
    def configure_optimizers(self):
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        
        params = list(self.model.student_encoder.parameters()) + list(self.model.predictor.parameters())
        

        opt_cfg = OmegaConf.to_container(self.optimizer_cfg, resolve=True)
        opt_target = opt_cfg.pop('_target_')
        
        import torch.optim as optim
        optimizer_class = getattr(optim, opt_target.split('.')[-1])
        optimizer = optimizer_class(params, lr=self.learning_rate, **opt_cfg)
        
        if self.scheduler_cfg is not None:
            sched_cfg = OmegaConf.to_container(self.scheduler_cfg, resolve=True)
            sched_target = sched_cfg.pop('_target_')
            scheduler_class = getattr(optim.lr_scheduler, sched_target.split('.')[-1])
            scheduler = scheduler_class(optimizer, **sched_cfg)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        
        return optimizer