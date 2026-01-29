import torch
from torch import nn



from torch_geometric.data import Data


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
        self.pos_embed = nn.Linear(pos_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        

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
        all_pos = torch.cat([context_pos, target_pos], dim=0)
        pos_mean = all_pos.mean(dim=0, keepdim=True)
        pos_std = all_pos.std(dim=0, keepdim=True).clamp(min=1e-6)
        
        context_pos_norm = (context_pos - pos_mean) / pos_std
        target_pos_norm = (target_pos - pos_mean) / pos_std
        

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






def sigreg(x: torch.Tensor ,num_slices: int = 256) -> torch.Tensor:
    device = x.device
    proj_shape = (x.size(1),num_slices)
    A = torch.randn(proj_shape,device=device)
    A /= A.norm(p=2,dim = 0)
    t = torch.linspace(-5,5,17,device=device)
    exp_f = torch.exp(-0.5 * t**2)
    x_t = (x@A).unsqueeze(2)*t
    ecf = torch.exp(1j*x_t).mean(0)

    err = (ecf - exp_f).abs().square().mul(exp_f)
    N = x.size(0)
    T = torch.trapz(err,t,dim=1)*N
    return T


class LeJEPA(nn.Module):
    def __init__(self, encoder: nn.Module, 
                 predictor: nn.Module, 
                 lambd: float, 
                 num_slices: int = 256,
                  **kwargs ): 

        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.lambd = lambd
        self.num_slices = num_slices
        self.loss_fn = nn.MSELoss()
    def _ema(self):
        return
    def forward(self, context, target):
        context_enc = self.encoder(context.x, context.edge_index, context.edge_attr)
        target_enc =  self.encoder(target.x, target.edge_index, target.edge_attr)
        
        pred = self.predictor(
            context_emb=context_enc,
            context_pos=context.pos,
            target_pos=target.pos
        )
        loss_fn = self.loss_fn(pred,target_enc)
        loss_reg = (torch.mean(sigreg(context_enc, self.num_slices)) + torch.mean(sigreg(target_enc, self.num_slices))) / 2
        loss = (1 - self.lambd) * loss_fn + self.lambd * loss_reg
        
        return loss
from typing import Callable, Optional
import copy





class GraphJepa(nn.Module):
    def __init__(
        self, 
        encoder: nn.Module, 
        predictor: nn.Module,

        ema: float = 0.996,
        **kwargs
    ):
        super().__init__()
        

        self.ema = ema
        
        
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
    
    def forward(self, context, target):

        

        context_enc = self.student_encoder(context.x, context.edge_index, context.edge_attr)
        
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
        context_x, target_x = batch
        with torch.no_grad():
            z = self.model.student_encoder(context_x.x, context_x.edge_index)
            
            # --- Существующие метрики ---
            std_z = torch.sqrt(z.var(dim=0) + 1e-4)
            std_loss = torch.mean(torch.nn.functional.relu(2 - std_z)) 

            
            #std = z.std(dim=0).mean()
            norm = z.norm(dim=-1).mean()
            self.log("debug_z_std", std_z.mean(), prog_bar=True)
            self.log("debug_z_norm", norm, prog_bar=True)


            z_centered = z - z.mean(dim=0, keepdim=True)
            _, S, _ = torch.linalg.svd(z_centered, full_matrices=False)
            

            self.log("debug_svd_max", S[0], prog_bar=False)
            
            self.log("debug_svd_2nd", S[1], prog_bar=False)
            self.log("debug_svd_3rd", S[2], prog_bar=False)
            

            self.log("debug_svd_min", S[-1], prog_bar=False)
            p = S / (S.sum() + 1e-9)
            entropy = -torch.sum(p * torch.log(p + 1e-9))
            rank_me = torch.exp(entropy)
            
            self.log("debug_rank_me", rank_me, prog_bar=True)
            cond_number = S[0] / (S[-1] + 1e-9)
            self.log("debug_cond_number", cond_number, prog_bar=False)
            return std_loss
    
    def training_step(self, batch):
        context_batch, target_batch = batch
        context_batch.edge_attr = torch.exp(-context_batch.edge_attr**2 / self.sigma**2)
        target_batch.edge_attr = torch.exp(-target_batch.edge_attr**2 / self.sigma**2)
        loss = self.model(context_batch,target_batch)

 
        if self.debug:
            std_loss=self._debug_log(batch)
        total_loss = loss 
        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss
    
    def validation_step(self, batch):
        context_batch, target_batch = batch
        context_batch.edge_attr = torch.exp(-context_batch.edge_attr**2 / self.sigma**2)
        target_batch.edge_attr = torch.exp(-target_batch.edge_attr**2 / self.sigma**2)
        loss = self.model(context_batch,target_batch)
        self.log("val_loss", loss, prog_bar=True)
        if self.debug:
            self._debug_log(batch)
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.model._ema()
    
    def configure_optimizers(self):
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        
        params = list(self.model.parameters())
        

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