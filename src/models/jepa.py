import torch
from torch import nn

seed = 1# из конфига


def sigreg(x: torch.Tensor, global_step: int, num_slices: int = 256) -> torch.Tensor:
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
from typing import Callable
import copy

class GraphJepa(nn.Module):
    def __init__(self, encoder, predictor ,ema = 0.996):
        self.mask_token = nn.Parameter(torch.tensor(1,encoder[0].in_channels))
        torch.nn.init.normal_(self.mask_token,std=0.2)

        self.student_encoder = encoder
        self.teach_encoder = copy.deepcopy(encoder)

        for p in self.teach_encoder.parametrs():
            p.requires_grad = False


        self.predictor = predictor
    @torch.no_grad()
    def _ema(self):
        for params_s , params_t in zip(self.student_encoder.parametrs(),self.teach_encoder.parametrs()):
            params_t.data.mul_(self.ema).add_((1 - self.ema)* params_s.data)


    def forward(self,x,edge_index, mask_function: Callable ,edge_weight = None ):
        context , target = mask_function(x, edge_index)
        context_enc = self.student_encoder(context)
        with torch.no_grad():
            teacher_enc = self.teach_encoder(target) # Еще подсказку добавить ну там было у меня такое
        out = self.predictor(context_enc)

        return loss(out,teacher_enc)


        
