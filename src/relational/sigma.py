"""Per-sample noise scales σ(q) for ACR.

The variants follow docx §4.2:

* :func:`sigma_fixed` — global scalar (the Π-model baseline).
* :func:`sigma_jac` — ``c / ‖∂f/∂q‖``, computed with stop-gradient on σ.
* :func:`sigma_dens` — ``σ_max · min(1, d_k(q) / d̄_k)`` against labeled q's
  in the input space (1D experiment only — embeddings would be needed in
  higher dimensions, see docx R6).
* :func:`sigma_oracle_lipschitz` — uses the *true* gradient of ``y_star``
  rather than the model's. This is the upper bound any adaptive method
  could reach if it perfectly estimated local Lipschitz structure.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor, nn


SigmaFn = Callable[[Tensor], Tensor]


def sigma_fixed(value: float) -> SigmaFn:
    def _fn(q: Tensor) -> Tensor:
        return torch.full((q.shape[0], 1), float(value), device=q.device)

    return _fn


def sigma_jac(
    model: nn.Module,
    c: float,
    sigma_max: float,
    eps_norm: float = 1e-4,
) -> SigmaFn:
    """Closure returning σ(q) = clip(c / ‖∂f/∂q‖, 0, σ_max), detached.

    Stop-gradient mitigation (docx §4.5): we reuse a fresh leaf tensor for
    the gradient computation so no second-order graph is built and no
    gradient flows back into θ via σ.
    """

    def _fn(q: Tensor) -> Tensor:
        q_leaf = q.detach().clone().requires_grad_(True)
        out = model(q_leaf).sum()
        (grad,) = torch.autograd.grad(out, q_leaf, create_graph=False)
        # ‖∂f/∂q‖ along the input dimension.
        norm = grad.norm(dim=-1, keepdim=True).clamp_min(eps_norm)
        sigma = (c / norm).clamp(min=0.0, max=sigma_max)
        return sigma.detach()

    return _fn


def sigma_dens(
    labeled_x: Tensor,
    sigma_max: float,
    k: int = 3,
) -> SigmaFn:
    """σ_max · min(1, d_k(q) / d̄_k) where d_k is distance to k-th labeled neighbour.

    d̄_k is computed once over the labeled set (the average k-th-NN distance
    among labeled points themselves) and used as the normalising scale.

    Sign convention follows docx §4.2.2: points *far* from labeled data get
    larger σ (we are extrapolating, smooth aggressively); points *close*
    get smaller σ (respect local supervision).
    """
    labeled_x = labeled_x.detach()
    n = labeled_x.shape[0]
    k = min(k, max(1, n - 1))

    # Self-distances among labeled points to set the d̄_k scale.
    with torch.no_grad():
        d2 = torch.cdist(labeled_x, labeled_x)
        # exclude self (diagonal) by adding inf
        d2 = d2 + torch.eye(n, device=d2.device) * 1e9
        d_k_self, _ = torch.topk(d2, k=k, dim=-1, largest=False)
        d_bar = d_k_self[:, -1].mean().clamp_min(1e-6).item()

    def _fn(q: Tensor) -> Tensor:
        with torch.no_grad():
            d = torch.cdist(q, labeled_x)
            d_k_q, _ = torch.topk(d, k=k, dim=-1, largest=False)
            d_at_k = d_k_q[:, -1:].clamp_min(0.0)
            sigma = sigma_max * (d_at_k / d_bar).clamp(max=1.0)
        return sigma

    return _fn


def sigma_oracle_lipschitz(
    y_star_grad_fn: Callable[[Tensor], Tensor],
    c: float,
    sigma_max: float,
    eps_norm: float = 1e-4,
) -> SigmaFn:
    """Oracle σ(q) = clip(c / ‖∇y*(q)‖, 0, σ_max).

    Same closed form as σ_jac but uses ground-truth local Lipschitz,
    bypassing the self-referential fixed-point dynamics that σ_jac is
    subject to (docx §4.4).
    """

    def _fn(q: Tensor) -> Tensor:
        with torch.no_grad():
            g = y_star_grad_fn(q)
            norm = g.abs().clamp_min(eps_norm)
            sigma = (c / norm).clamp(min=0.0, max=sigma_max)
        return sigma

    return _fn


def y_star_grad_1d(x: Tensor) -> Tensor:
    """Analytic ``dy*/dx`` for the 1D synthetic task."""
    high = x > math.pi
    return torch.where(high, 5.0 * torch.cos(x), torch.cos(x))
