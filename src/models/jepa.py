import torch
from torch import nn


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


def default_random_mask(x: torch.Tensor, edge_index: torch.Tensor, mask_ratio: float = 0.3):
    """
    Default random node masking function.
    Returns context (masked nodes) and target (original nodes for masked positions).
    """
    num_nodes = x.size(0)
    num_mask = int(num_nodes * mask_ratio)
    perm = torch.randperm(num_nodes, device=x.device)
    mask_indices = perm[:num_mask]
    
    mask = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)
    mask[mask_indices] = True
    
    return x, mask


class GraphJepa(nn.Module):
    def __init__(
        self, 
        encoder: nn.Module, 
        predictor: nn.Module,
        mask_function: Optional[Callable] = None,
        ema: float = 0.996,
        mask_ratio: float = 0.3
    ):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        self.ema = ema
        
        self.mask_function = mask_function if mask_function is not None else default_random_mask
        
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
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None):
        _, mask = self.mask_function(x, edge_index, self.mask_ratio)
        
        x_masked = x.clone()
        x_masked[mask] = self.mask_token.expand(mask.sum(), -1)
        
        context_enc = self.student_encoder(x_masked, edge_index, edge_weight)
        

        with torch.no_grad():
            teacher_enc = self.teach_encoder(x, edge_index, edge_weight)
        
        pred = self.predictor(context_enc)
        loss = self.loss_fn(pred[mask], teacher_enc[mask].detach())
        
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
    
    def _debug_log(self, batch):
        with torch.no_grad():
            z = self.model.teach_encoder(batch.x, batch.edge_index)
            std = z.std(dim=0).mean()
            norm = z.norm(dim=-1).mean()
        
        self.log("debug_z_std", std, prog_bar=True)
        self.log("debug_z_norm", norm, prog_bar=True)
    
    def training_step(self, batch):
        loss = self.model(batch.x, batch.edge_index)
        self.log("train_loss", loss, prog_bar=True)
        if self.debug:
            self._debug_log(batch)
        return loss
    
    def validation_step(self, batch):
        loss = self.model(batch.x, batch.edge_index)
        if self.debug:
            self._debug_log(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.model._ema()
    
    def configure_optimizers(self):
        from hydra.utils import instantiate
        
        params = [
            {'params': self.model.student_encoder.parameters(), 'lr': self.learning_rate},
            {'params': self.model.predictor.parameters(), 'lr': self.learning_rate},
        ]
        
        optimizer = instantiate(self.optimizer_cfg, params=params)
        
        if self.scheduler_cfg is not None:
            scheduler = instantiate(self.scheduler_cfg, optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        
        return optimizer