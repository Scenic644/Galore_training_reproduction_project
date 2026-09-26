# src/galore_optimizer.py
import torch
from torch.optim import AdamW
from .galore_projector import GaLoreProjector

class GaLoreAdamW(AdamW):
    """
    AdamW with gradient low-rank projection.
    Wraps PyTorch AdamW: projects gradients before optimizer.step(),
    projects updates back after.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, rank=128, update_proj_gap=200,
                 scale=0.25, proj_type='std'):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, **defaults)

        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.proj_type = proj_type
        self.projectors = {}  # param_id -> GaLoreProjector
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Only project 2D parameters (weight matrices)
                if p.grad.dim() == 2:
                    pid = id(p)
                    if pid not in self.projectors:
                        self.projectors[pid] = GaLoreProjector(
                            rank=self.rank,
                            update_proj_gap=self.update_proj_gap,
                            scale=self.scale,
                            proj_type=self.proj_type
                        )
                    proj = self.projectors[pid]
                    # Project gradient to low rank
                    low_rank_grad = proj.project(p.grad, self.step_count)
                    # Replace grad with low-rank version
                    p.grad.data = low_rank_grad

        # Standard AdamW step on low-rank gradients
        super().step(closure)

        # Project updates back to full space
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.dim() == 2:
                    pid = id(p)
                    if pid in self.projectors:
                        proj = self.projectors[pid]
                        # Note: AdamW has already updated p using low-rank grad.
                        # We need to re-project the effective update.
                        # (In practice, the official code does this differently—
                        #  see PROVENANCE.md for notes.)
