"""Losses used by the experiments.

The two characters here are:

* :func:`original_pdf_loss` — exact reproduction of the 2024-04-30 PDF's
  ``training_step`` from the slides, with a ``zero_transductive`` switch that
  replaces ``f(q)`` with zeros. Per the docx Cancellation Theorem (Appendix A,
  Corollary 2) the two should produce *identical* gradients.
* :func:`consistency_loss` — Π-model-style consistency on unlabeled points,
  with a callable ``sigma_fn(q) -> tensor`` supplying the per-sample noise
  scale. The fixed-σ baseline corresponds to ``sigma_fn = lambda q: σ * ones``.
"""

from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# --------- exp0 -----------------------------------------------------------------


def original_pdf_loss(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    transductive_x: Tensor,
    *,
    zero_transductive: bool = False,
) -> Tuple[Tensor, dict]:
    """Reproduce the PDF slide's ``training_step``.

    ``zero_transductive=True`` replaces ``model(transductive_x)`` with a
    zero tensor of the same shape; per the Cancellation Theorem the loss
    should be invariant to this swap (the ``f(q)`` terms appear with
    opposing coefficients summing to zero).
    """
    n = x.shape[0]
    y = y.to(dtype=torch.float32).view(n, 1)
    x = x.view(n, -1)

    x_rev = torch.flip(x, dims=(0,))
    y_rev = torch.flip(y, dims=(0,))

    preds = model(x)
    preds_rev = model(x_rev)

    if zero_transductive:
        # Skip the forward entirely — the algebra cancels the values anyway.
        # Returning zeros mirrors the math without spending FLOPs.
        t_preds = torch.zeros_like(preds)
    else:
        t_preds = model(transductive_x.view(transductive_x.shape[0], -1))

    preds_plus_t = preds + t_preds
    preds_rev_plus_t = preds_rev + t_preds
    preds_minus_rev = preds - preds_rev

    sup = F.mse_loss(preds, y)
    rel_with_t = F.mse_loss(preds_plus_t - preds_rev_plus_t, y - y_rev)
    rel = F.mse_loss(preds_minus_rev, y - y_rev)

    total = sup + rel_with_t + rel
    parts = {
        "supervised": sup.detach().item(),
        "relational_with_transductive": rel_with_t.detach().item(),
        "relational_pairwise": rel.detach().item(),
    }
    return total, parts


def supervised_mse_loss(
    model: nn.Module, x: Tensor, y: Tensor
) -> Tuple[Tensor, dict]:
    """Plain regression baseline (PDF ``version_15``)."""
    y = y.to(dtype=torch.float32).view(-1, 1)
    preds = model(x.view(x.shape[0], -1))
    loss = F.mse_loss(preds, y)
    return loss, {"supervised": loss.detach().item()}


# --------- exp1: symmetry-breaking noise on the transductive forward -----------


def _two_noisy_forwards(
    model: nn.Module,
    transductive_x: Tensor,
    *,
    noise_std: float,
    noise_space: str,
    noise_scale: Tensor | None,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor]:
    """Helper: produce ``z_a = f(q + ε_a)`` and ``z_b = f(q + ε_b)`` per the
    same conventions used by :func:`noisy_pdf_loss`. Used by both the
    relational variant (v17) and the plain Π-model variant (v18) so they
    share a noise process.
    """
    if noise_space not in {"input", "embedding"}:
        raise ValueError(noise_space)
    tx = transductive_x.view(transductive_x.shape[0], -1)

    if noise_space == "input":
        shape = tx.shape
        if noise_scale is None:
            scale = torch.ones(1, tx.shape[1], device=tx.device)
        else:
            scale = noise_scale.view(1, -1).to(tx.device)
        if generator is not None:
            eps_a = torch.randn(shape, generator=generator) * (noise_std * scale)
            eps_b = torch.randn(shape, generator=generator) * (noise_std * scale)
        else:
            eps_a = torch.randn_like(tx) * (noise_std * scale)
            eps_b = torch.randn_like(tx) * (noise_std * scale)
        z_a = model(tx + eps_a)
        z_b = model(tx + eps_b)
    else:  # embedding
        with torch.no_grad():
            emb_probe = model.encoder(tx)
        shape = tuple(emb_probe.shape)
        if noise_scale is None:
            scale = emb_probe.std(dim=0, keepdim=True).clamp_min(1e-6)
        else:
            scale = noise_scale.view(1, -1).to(tx.device)
        if generator is not None:
            eps_a = torch.randn(shape, generator=generator) * (noise_std * scale)
            eps_b = torch.randn(shape, generator=generator) * (noise_std * scale)
        else:
            eps_a = torch.randn(shape, device=tx.device) * (noise_std * scale)
            eps_b = torch.randn(shape, device=tx.device) * (noise_std * scale)
        z_a = model.forward_with_emb_noise(tx, eps_a)
        z_b = model.forward_with_emb_noise(tx, eps_b)
    return z_a, z_b


def consistency_only_loss(
    model: nn.Module,
    q: Tensor,
    *,
    noise_std: float,
    noise_space: str = "input",
    noise_scale: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Pure ``MSE(f(q + ε_a), f(q + ε_b))`` — no labelled data involved.

    Used by the matched-optimiser-step fairness control (exp1e): after
    each main labelled-batch step we can run K extra optimiser steps on
    fresh unlabelled batches using only this consistency term, so that
    v18 / v17 can match v15's compute budget while burning the extra
    steps on unlabelled signal rather than re-processing labels.
    """
    z_a, z_b = _two_noisy_forwards(
        model, q,
        noise_std=noise_std,
        noise_space=noise_space,
        noise_scale=noise_scale,
        generator=generator,
    )
    return F.mse_loss(z_a, z_b)


def pi_model_pdf_loss(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    transductive_x: Tensor,
    *,
    noise_std: float,
    noise_space: str = "input",
    noise_scale: Tensor | None = None,
    pi_weight: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tuple[Tensor, dict]:
    """The "v18" baseline: supervised + 2× reversal-augmentation + plain
    Π-model consistency on the unlabeled forwards.

    By the exp1 algebra,
        v17 = v18 + (mean-zero cross term)
    so v17 and v18 should be statistically indistinguishable in
    expectation. This function exists to make that A/B comparison
    explicit. The noise process matches :func:`noisy_pdf_loss` so a
    matched-seed run can produce identical ε draws across the two losses.
    """
    n = x.shape[0]
    y = y.to(dtype=torch.float32).view(n, 1)
    x = x.view(n, -1)
    x_rev = torch.flip(x, dims=(0,))
    y_rev = torch.flip(y, dims=(0,))

    preds = model(x)
    preds_rev = model(x_rev)

    z_a, z_b = _two_noisy_forwards(
        model, transductive_x,
        noise_std=noise_std, noise_space=noise_space,
        noise_scale=noise_scale, generator=generator,
    )

    sup = F.mse_loss(preds, y)
    # Match v16's structure: two copies of the reversal-augmentation MSE
    # (the cancelling middle term in v16 reduces to this exactly).
    rel = F.mse_loss(preds - preds_rev, y - y_rev)
    pi = F.mse_loss(z_a, z_b)

    total = sup + 2.0 * rel + pi_weight * pi
    parts = {
        "supervised": sup.detach().item(),
        "relational_pairwise_x2": (2.0 * rel).detach().item(),
        "pi_model_consistency": (pi_weight * pi).detach().item(),
    }
    return total, parts


def noisy_pdf_loss(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    transductive_x: Tensor,
    *,
    noise_std: float,
    noise_space: str = "input",  # "input" | "embedding"
    noise_scale: Tensor | None = None,
    zero_transductive: bool = False,
    generator: torch.Generator | None = None,
) -> Tuple[Tensor, dict]:
    """PDF loss with two *different* noisy forwards on the transductive sample.

    The original PDF loss has ``(p + f(q)) - (p_rev + f(q))`` — the two
    ``f(q)`` terms cancel. We replace them with ``f(q + ε_a)`` and
    ``f(q + ε_b)`` where ε_a ≠ ε_b are drawn independently per step. The
    transductive contribution becomes ``z_a - z_b``, a non-zero quantity
    that carries gradient through θ.

    ``noise_space="input"`` perturbs the raw input.
    ``noise_space="embedding"`` perturbs the bottleneck embedding produced
    by ``model.encoder``; the model must expose ``forward_with_emb_noise``.

    ``noise_scale``: optional per-dimension multiplier (broadcasted along
    the batch axis). If supplied, the perturbation is
    ``ε = noise_std · noise_scale · z``, ``z ~ N(0, I)``. This preserves
    the relative structure of the input/embedding by giving low-variance
    dimensions tiny noise and high-variance dimensions more — ``noise_std``
    becomes a fraction of typical per-dimension variation. For the
    embedding case, if ``noise_scale`` is None we compute it on-the-fly
    from the current batch's embedding stds.

    ``zero_transductive=True`` is the falsification probe: it replaces the
    noisy transductive forwards with zeros. If the gradients of the noisy
    and zeroed variants differ substantially, the symmetry is broken and
    the unlabeled data is contributing.
    """
    if noise_space not in {"input", "embedding"}:
        raise ValueError(noise_space)

    n = x.shape[0]
    y = y.to(dtype=torch.float32).view(n, 1)
    x = x.view(n, -1)

    x_rev = torch.flip(x, dims=(0,))
    y_rev = torch.flip(y, dims=(0,))

    preds = model(x)
    preds_rev = model(x_rev)

    if zero_transductive:
        z_a = torch.zeros_like(preds)
        z_b = torch.zeros_like(preds)
    else:
        if noise_space == "input":
            tx = transductive_x.view(transductive_x.shape[0], -1)
            shape = tx.shape
            if noise_scale is None:
                scale = torch.ones(1, tx.shape[1], device=tx.device)
            else:
                scale = noise_scale.view(1, -1).to(tx.device)
            if generator is not None:
                eps_a = torch.randn(shape, generator=generator) * (noise_std * scale)
                eps_b = torch.randn(shape, generator=generator) * (noise_std * scale)
            else:
                eps_a = torch.randn_like(tx) * (noise_std * scale)
                eps_b = torch.randn_like(tx) * (noise_std * scale)
            z_a = model(tx + eps_a)
            z_b = model(tx + eps_b)
        else:  # embedding-space noise
            tx = transductive_x.view(transductive_x.shape[0], -1)
            with torch.no_grad():
                # One forward to get the embedding shape and (if requested)
                # the per-dim std for variance-scaled noise. We do not flow
                # gradients through the scale.
                emb_probe = model.encoder(tx)
            shape = tuple(emb_probe.shape)
            if noise_scale is None:
                scale = emb_probe.std(dim=0, keepdim=True).clamp_min(1e-6)
            else:
                scale = noise_scale.view(1, -1).to(tx.device)
            if generator is not None:
                eps_a = torch.randn(shape, generator=generator) * (noise_std * scale)
                eps_b = torch.randn(shape, generator=generator) * (noise_std * scale)
            else:
                eps_a = torch.randn(shape, device=tx.device) * (noise_std * scale)
                eps_b = torch.randn(shape, device=tx.device) * (noise_std * scale)
            z_a = model.forward_with_emb_noise(tx, eps_a)
            z_b = model.forward_with_emb_noise(tx, eps_b)

    sup = F.mse_loss(preds, y)
    rel_with_noisy_t = F.mse_loss((preds + z_a) - (preds_rev + z_b), y - y_rev)
    rel = F.mse_loss(preds - preds_rev, y - y_rev)

    total = sup + rel_with_noisy_t + rel
    parts = {
        "supervised": sup.detach().item(),
        "relational_with_noisy_transductive": rel_with_noisy_t.detach().item(),
        "relational_pairwise": rel.detach().item(),
    }
    return total, parts


# --------- exp2 -----------------------------------------------------------------


SigmaFn = Callable[[Tensor], Tensor]


def consistency_loss(
    model: nn.Module,
    q: Tensor,
    sigma_fn: SigmaFn,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Π-model consistency: ``MSE(f(q), f(q + ε))`` with ``ε ~ N(0, σ(q)²·I)``.

    ``sigma_fn`` returns a per-sample positive scalar tensor of shape
    ``(n, 1)``. It is *expected to be detached*; this function does not
    re-detach.
    """
    sigma = sigma_fn(q)
    if sigma.dim() == 1:
        sigma = sigma.unsqueeze(-1)

    if generator is not None:
        eps = torch.randn(q.shape, generator=generator, device=q.device) * sigma
    else:
        eps = torch.randn_like(q) * sigma

    f_q = model(q)
    f_q_eps = model(q + eps)
    return F.mse_loss(f_q, f_q_eps)


def lambda_rampup(epoch: int, n_rampup: int = 80, max_lambda: float = 1.0) -> float:
    """Sigmoid ramp-up of the consistency weight (docx §4.1)."""
    if n_rampup <= 0:
        return max_lambda
    t = max(0.0, min(1.0, epoch / n_rampup))
    # Standard SSL exponential ramp; matches Π-model / Mean Teacher conventions.
    import math

    return max_lambda * math.exp(-5.0 * (1.0 - t) ** 2)
