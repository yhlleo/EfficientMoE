import math
import torch 
from torch.nn.parallel import DistributedDataParallel as DDP 
import torch.nn as nn
from tqdm import tqdm
from torchdiffeq import odeint
import torch.nn.functional as F

from utils import wandb_log

import os

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def expand_t_like_x(t, x):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t

class RectifiedFlow(torch.nn.Module):
    def __init__(self, model, ln=False):
        super().__init__()
        self.ln = ln
        self.model = model

    def forward(self, x, cond, train_step=0):

        b = x.size(0)
        z1 = x
        z0 = torch.randn_like(x)
        t = torch.rand((b,)).to(x)
        texp = expand_t_like_x(t, x)

        alpha_t = texp
        sigma_t = 1 - texp
        d_alpha_t = 1
        d_sigma_t = -1
        ut = d_alpha_t * z1 + d_sigma_t * z0
        zt = alpha_t * z1 + sigma_t * z0
        model_output = self.model(zt, t, cond, train_step=train_step) 

        terms = {}
        terms["loss"] = 0

        use_pos_weight = False

        if isinstance(model_output, tuple):
            loss_stratgy_name = model_output[1]
            if loss_stratgy_name == "Capacity_Pred":
                terms["cp_loss"] = 0
                layer_idx_list, ones_list, pred_c_list, CapacityPred_loss_weight = model_output[2:]
                for layer_idx, ones, pred_c in zip(layer_idx_list, ones_list, pred_c_list):
                    terms[f"Capacity_Pred_loss_{layer_idx}"] = nn.BCEWithLogitsLoss()(pred_c, ones)
                    terms["loss"] +=  terms[f"Capacity_Pred_loss_{layer_idx}"]  * CapacityPred_loss_weight
                    terms["cp_loss"] += terms[f"Capacity_Pred_loss_{layer_idx}"]  * CapacityPred_loss_weight

            else:
                #raise Exception("not defined training loss")
                pass

            model_output = model_output[0]

        if model_output.shape[1] != x.shape[1]:
            model_output, _ = model_output.chunk(2, dim=1)
        
        batchwise_mse = mean_flat(((model_output - ut) ** 2))
        terms["mse"] = batchwise_mse

        if "vb" in terms:
            terms["loss"] += terms["mse"].mean() + terms["vb"].mean()
        else:
            terms["loss"] += terms["mse"].mean()

        return terms

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0, progress=False, mode='euler'):
        print(f'Using {mode} Sampler')
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        device = z.device
        images = [z]
        # Use tqdm for progress bar if progress is True
        loop_range = tqdm(range(0, sample_steps, 1), desc="Sampling") if progress else range(0, sample_steps, 1)


        def fn(z, t, cond):
            vc = self.model(z, t, cond)
            if isinstance(vc, tuple):
                vc = vc[0]
            if vc.shape[1] != z.shape[1]:
                vc, _ = vc.chunk(2, dim=1)

            return vc

        def fn_v(z, t):
            vc = fn(z, t, cond)
            if null_cond is not None:
                vu = fn(z, t, null_cond)
                vc = vu + cfg * (vc - vu)
            return vc

        def _fn(t, z):
            t = torch.tensor([t] * b).to(z.device)
            return fn_v(z, t)

        def euler_step(z, i):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)
            vc = fn_v(z, t)
            z = z + dt * vc
            return z

        def heun_step(z, i):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)
            t_plus_1 = (i+1) / sample_steps
            t_plus_1 = torch.tensor([t_plus_1] * b).to(z.device)
            vc = fn_v(z, t)
            z_tilde_plus_1 = z + dt * vc
            vc_plus_1 = fn_v(z_tilde_plus_1, t_plus_1)
            z = z + 1/2 * dt * (vc + vc_plus_1)
            return z

        if 'torchdiff' in mode:
            mode = mode.split('-')[-1]
            self.atol = 1e-6
            self.rtol = 1e-3
            atol = [self.atol] * len(z) if isinstance(z, tuple) else [self.atol]
            rtol = [self.rtol] * len(z) if isinstance(z, tuple) else [self.rtol]
            t = torch.linspace(0, 1, sample_steps).to(z.device)

            samples = odeint(
                _fn,
                z,
                t,
                method=mode,
                atol=atol,
                rtol=rtol
            )
            images.append(samples[-1])

        else:
            for i in loop_range:
                os.environ["cur_step"] = f"{i:003d}"
                if 'euler' in mode:
                    z = euler_step(z, i)
                elif 'heun' in mode:
                    z = heun_step(z, i)
                else:
                    raise NotImplementedError(f"Mode '{mode}' is not implemented.")
            images.append(z)

        return images
