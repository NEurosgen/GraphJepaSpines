import torch
from torch import nn
import pytorch_lightning as L
from omegaconf import OmegaConf
import torch.optim as optim
from src.models.encoder import linear_edge_weighting

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
        self.pos_embed = nn.Linear(pos_dim, hidden_dim) # Вроде норм
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.mask_token = nn.Parameter(torch.zeros(hidden_dim))

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

    
        

        context_key = context_emb + self.pos_embed(context_pos)
        context_val = context_emb
        # Query from target positions
        target_query_orig = self.mask_token + self.pos_embed(target_pos)
        
        # Cross-attention (add batch dimension for nn.MultiheadAttention)
        target_query = target_query_orig.unsqueeze(0)
        context_key = context_key.unsqueeze(0)
        context_val = context_val.unsqueeze(0)
        
        attn_out, _ = self.cross_attn(
            query=target_query,
            key=context_key,
            value=context_val
        )
        attn_out = attn_out.squeeze(0)
        
        # Residual + LayerNorm + MLP
        x = target_query_orig + attn_out
        x = x + self.mlp(self.norm1(x))
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
    T = torch.trapezoid(err,t,dim=1)*N
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
    def encode(self, x, edge_index, edge_attr):
        return self.encoder(x, edge_index, edge_attr)
    def forward(self, context, target):
        context_enc = self.encoder(context.x, context.edge_index, context.edge_attr)
        target_enc =  self.encoder(target.x, target.edge_index, target.edge_attr)
        
        pred = self.predictor(
            context_emb=context_enc,
            context_pos=context.pos,
            target_pos=target.pos
        )
        loss = self.loss_fn(pred,target_enc)
        loss_reg = (torch.mean(sigreg(context_enc, self.num_slices)) + torch.mean(sigreg(target_enc, self.num_slices))) / 2
        loss = (1 - self.lambd) * loss + self.lambd * loss_reg
        
        return loss








class JepaLight(L.LightningModule):
    def __init__(self, cfg, model=None, debug: bool = False, **kwargs):
        super().__init__()
        self.save_hyperparameters("cfg")
        self.debug = debug
        self.cfg = cfg.training
        self.model = model

        self.learning_rate = self.cfg.learning_rate
        self.optimizer_cfg = self.cfg.optimizer
        self.scheduler_cfg = self.cfg['scheduler']

   
    
    def encode(self, x, edge_index, edge_attr):
        return self.model.encode(x, edge_index, edge_attr)
    
    def _apply_linear_weights(self, batch):
        """Linear transformation  (Min-Max нормализация)."""
        if batch.edge_attr is not None and batch.edge_attr.numel() > 0:
            batch = linear_edge_weighting(batch)
            
        return batch
    
    def _shared_step(self, batch, step_name: str):
        """General logic training_step и validation_step."""
        context_batch, target_batch = batch
        
        context_batch = self._apply_linear_weights(context_batch)
        target_batch = self._apply_linear_weights(target_batch)
        
        loss = self.model(context_batch, target_batch)
        self.log(f"{step_name}_loss", loss, prog_bar=True)
        
            
        return loss

    def training_step(self, batch):
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        opt_cfg = OmegaConf.to_container(self.optimizer_cfg, resolve=True)
        optimizer_class = getattr(optim, opt_cfg.pop('_target_').split('.')[-1])
        optimizer = optimizer_class(self.model.parameters(), lr=self.learning_rate, **opt_cfg)
        
        if self.scheduler_cfg is not None:
            sched_cfg = OmegaConf.to_container(self.scheduler_cfg, resolve=True)
            scheduler_class = getattr(optim.lr_scheduler, sched_cfg.pop('_target_').split('.')[-1])
            scheduler = scheduler_class(optimizer, **sched_cfg)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        
        return optimizer