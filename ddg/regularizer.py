"""
ddg/regularizer.py
==================
DDG (Discrete Differential Geometry) regularizer and stability scoring.

DDGRegularizer applies a smoothness penalty via the graph Laplacian:
    E_smooth = (1/2) * x^T L x  (Dirichlet energy)

StabilityScore measures how stable the per-point/atom latent features are
across training (low variance = high stability).
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple


class DDGRegularizer(nn.Module):
    """
    Computes the DDG regularization energy on a per-point / per-atom feature
    matrix using a prebuilt sparse Laplacian.

    E = lambda1 * Dirichlet(x, L, areas) + lambda2 * mean_feature_norm_penalty

    Args:
        lambda1: weight for the Dirichlet (smoothness) energy term.
        lambda2: weight for the feature-norm stability term.
    """

    def __init__(self, lambda1: float = 0.05, lambda2: float = 0.01):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def dirichlet_energy(
        self,
        x: torch.Tensor,
        L: torch.Tensor,
        areas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        E_D = (1/2) * tr(x^T L x), optionally area-weighted.

        x      : (N, F) per-point feature matrix
        L      : (N, N) sparse Laplacian (COO)
        areas  : (N,) per-point areas for mass-weighted norm (optional)
        """
        # L x  ->  (N, F)
        Lx = torch.sparse.mm(L, x)
        # tr(x^T Lx) = sum_i x_i · (Lx)_i
        if areas is not None:
            w = areas.unsqueeze(1)          # (N, 1)
            energy = (x * Lx * w).sum()
        else:
            energy = (x * Lx).sum()
        return 0.5 * energy

    def forward(
        self,
        x: torch.Tensor,
        L: torch.Tensor,
        areas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=x.device)
        if self.lambda1 > 0.0:
            loss = loss + self.lambda1 * self.dirichlet_energy(x, L, areas)
        if self.lambda2 > 0.0:
            loss = loss + self.lambda2 * x.pow(2).mean()
        return loss


class StabilityScore:
    """
    Tracks the per-channel variance of the latent feature map across batches
    and returns a scalar stability score in [0, 1].

    Higher score = more stable (lower mean-channel variance).
    """

    def __init__(self, ema_alpha: float = 0.05):
        self.alpha = ema_alpha
        self._mean: Optional[torch.Tensor] = None
        self._sq_mean: Optional[torch.Tensor] = None

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Update EMA statistics with a new batch of per-point features (N, F)."""
        mu = x.mean(0)
        sq = x.pow(2).mean(0)
        if self._mean is None:
            self._mean = mu.clone()
            self._sq_mean = sq.clone()
        else:
            a = self.alpha
            self._mean = (1 - a) * self._mean + a * mu
            self._sq_mean = (1 - a) * self._sq_mean + a * sq

    @torch.no_grad()
    def score(self) -> float:
        """Return scalar stability score in [0, 1]. Call after >=1 update."""
        if self._mean is None:
            return 0.0
        var = (self._sq_mean - self._mean.pow(2)).clamp(min=0.0)
        mean_var = float(var.mean().item())
        # sigmoid-like mapping: var=0 -> 1.0, var grows -> approaches 0
        return float(1.0 / (1.0 + mean_var))

    def reset(self) -> None:
        self._mean = None
        self._sq_mean = None
