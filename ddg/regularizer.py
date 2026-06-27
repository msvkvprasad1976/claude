"""
ddg/regularizer.py  (CORRECTED)
===============================
Energy penalties applied to a per-point / per-atom latent matrix X whose row
count matches the Laplacian L. This removes the size mismatch that crashed the
original code (Laplacian over points/atoms vs loss over pooled per-sample rows).

  E_Dirichlet = 0.5 * Tr(X^T L X)            smoothness
  E_Willmore  = Σ_v ||(L X)_v||^2 * A_v      curvature (cotangent) / roughness (combinatorial)
  L_total     = L_task + λ1 E_Dirichlet + λ2 E_Willmore
"""
import torch
import torch.nn as nn
from typing import Tuple


class DDGRegularizer(nn.Module):
    def __init__(self, lambda1: float = 0.05, lambda2: float = 0.01):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    @staticmethod
    def _energies(X: torch.Tensor, L: torch.Tensor, areas: torch.Tensor):
        M = max(X.shape[0], 1)                            # vertex count
        LX = torch.sparse.mm(L, X)                       # (M, d)
        # Normalize by M so the penalty is a MEAN per-vertex energy. This keeps
        # the regularizer on the same scale as the task loss regardless of point
        # count or batch size, and makes lambda transferable across datasets.
        e_d = 0.5 * (X * LX).sum() / M
        e_w = ((LX * LX).sum(dim=-1) * areas).sum() / M
        return e_d, e_w

    def forward(self, X: torch.Tensor, task_loss: torch.Tensor,
                L: torch.Tensor, areas: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e_d, e_w = self._energies(X, L, areas)
        total = task_loss + self.lambda1 * e_d + self.lambda2 * e_w
        return total, e_d.detach(), e_w.detach()


class StabilityScore:
    """S = 1 / (1 + var) over n_trials of Gaussian vertex noise on batch.pos."""
    def __init__(self, sigma_noise: float = 0.02, n_trials: int = 5):
        self.sigma_noise = sigma_noise
        self.n_trials = n_trials

    @torch.no_grad()
    def compute(self, model, batch, device, criterion) -> float:
        model.eval()
        batch = batch.to(device)
        losses = []
        for _ in range(self.n_trials):
            noisy = batch.clone()
            noisy.pos = batch.pos + torch.randn_like(batch.pos) * self.sigma_noise
            out = model(noisy)
            losses.append(criterion(out, batch.y).item())
        var = float(torch.tensor(losses).var().item())
        return 1.0 / (1.0 + var)
