"""Generic training loops for the experiments."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .losses import (
    consistency_loss,
    consistency_only_loss,
    lambda_rampup,
    noisy_pdf_loss,
    original_pdf_loss,
    pi_model_pdf_loss,
    supervised_mse_loss,
)


# --------- common ---------------------------------------------------------------


def seed_everything(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_mse(model: nn.Module, x: Tensor, y: Tensor) -> float:
    model.eval()
    with torch.no_grad():
        preds = model(x.view(x.shape[0], -1) if x.dim() > 2 else x)
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        return torch.nn.functional.mse_loss(preds, y.float()).item()


# --------- exp0: PDF-style runs -------------------------------------------------


@dataclass
class PDFRunResult:
    variant: str
    seed: int
    train_mse: list = field(default_factory=list)
    val_mse: list = field(default_factory=list)
    # Per-step total loss values, kept so we can diff v16 vs v16_zeroed.
    step_losses: list = field(default_factory=list)
    seconds: float = 0.0
    # Best val MSE across training and the epoch it was achieved at.
    # Populated automatically; useful when a long run (e.g. v15_4x) overfits.
    val_mse_best: float = float("inf")
    val_mse_best_epoch: int = -1
    # Number of extra consistency-only optimiser steps interleaved with
    # the labelled-batch steps. ``0`` for the standard 1x runs.
    extra_consistency_steps_per_labeled_step: int = 0
    # Held-out test eval (populated only when ``test_loader`` is passed).
    # The "unbiased" reporting number is ``test_mse_at_best_val_epoch``:
    # we select the early-stop epoch on val, then report test MSE there.
    test_mse: list = field(default_factory=list)
    test_mse_at_best_val_epoch: float = float("inf")
    test_mse_best: float = float("inf")
    test_mse_best_epoch: int = -1


_PDF_VARIANTS = {
    "v15", "v16", "v16_zeroed",
    "v17_input", "v17_emb",
    "v17_input_var", "v17_emb_var",
    # v18 = v16 + plain Π-model on the unlabeled forwards (matched-noise
    # baseline used to A/B-test whether the relational wrapping in v17 adds
    # anything beyond standard consistency).
    "v18_pi_input", "v18_pi_input_var",
    "v18_pi_emb", "v18_pi_emb_var",
}


def run_pdf_experiment(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    transductive_pool: Tensor,
    *,
    variant: str,
    epochs: int,
    lr: float,
    seed: int,
    log_steps: bool = False,
    noise_std: float = 0.5,
    input_noise_scale: Tensor | None = None,
    extra_consistency_steps_per_labeled_step: int = 0,
    test_loader: DataLoader | None = None,
) -> PDFRunResult:
    """Train a single PDF-style run.

    Variants:
      v15            — supervised baseline.
      v16            — original PDF "transductive" loss.
      v16_zeroed     — original loss with f(q) := 0 (cancellation probe).
      v17_input      — noisy transductive forwards in input space (isotropic).
      v17_emb        — noisy transductive forwards in embedding space (isotropic).
      v17_input_var  — input-space noise scaled by per-pixel std of the
                       transductive pool. ``input_noise_scale`` must be
                       provided (shape ``(input_dim,)``).
      v17_emb_var    — embedding-space noise scaled by per-coord std of the
                       embedding, computed on-the-fly per batch.
      v18_pi_input   — supervised + 2×reversal-aug + plain Π-model
                       consistency on q with isotropic input-space noise.
      v18_pi_input_var — same with per-pixel-std-scaled noise; needs
                       ``input_noise_scale``.
      v18_pi_emb     — same with isotropic embedding-space noise.
      v18_pi_emb_var — same with per-coord-std-scaled embedding noise.

    ``noise_std`` controls the symmetry-breaking noise for v17_*/v18_*.
    """
    if variant not in _PDF_VARIANTS:
        raise ValueError(f"unknown variant {variant}")
    if variant in ("v17_input_var", "v18_pi_input_var") and input_noise_scale is None:
        raise ValueError(f"{variant} requires input_noise_scale")

    seed_everything(seed)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    g_trans = torch.Generator().manual_seed(seed + 1)
    g_noise = torch.Generator().manual_seed(seed + 2)

    result = PDFRunResult(
        variant=variant, seed=seed,
        extra_consistency_steps_per_labeled_step=extra_consistency_steps_per_labeled_step,
    )
    t0 = time.time()
    pool_n = transductive_pool.shape[0]

    # Determine noise-related params once; they are reused both in the main
    # loss and in any extra consistency-only steps.
    if variant in ("v15", "v16", "v16_zeroed"):
        space = None
        scale = None
        consistency_capable = False
    else:
        space = "input" if "input" in variant else "embedding"
        is_var = variant.endswith("_var")
        if is_var:
            scale = input_noise_scale if space == "input" else None
        else:
            scale = None
        consistency_capable = True

    if extra_consistency_steps_per_labeled_step > 0 and not consistency_capable:
        raise ValueError(
            f"extra_consistency_steps_per_labeled_step > 0 needs a consistency-capable "
            f"variant; got {variant!r}"
        )

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x, y in train_loader:
            n = x.shape[0]
            if variant == "v15":
                loss, _ = supervised_mse_loss(model, x, y)
            elif variant in ("v16", "v16_zeroed"):
                idx = torch.randint(0, pool_n, (n,), generator=g_trans)
                t_x = transductive_pool[idx]
                loss, _ = original_pdf_loss(
                    model, x, y,
                    transductive_x=t_x,
                    zero_transductive=(variant == "v16_zeroed"),
                )
            else:  # v17_* or v18_pi_*
                idx = torch.randint(0, pool_n, (n,), generator=g_trans)
                t_x = transductive_pool[idx]
                if variant.startswith("v17_"):
                    loss, _ = noisy_pdf_loss(
                        model, x, y,
                        transductive_x=t_x,
                        noise_std=noise_std,
                        noise_space=space,
                        noise_scale=scale,
                        generator=g_noise,
                    )
                else:  # v18_pi_*
                    loss, _ = pi_model_pdf_loss(
                        model, x, y,
                        transductive_x=t_x,
                        noise_std=noise_std,
                        noise_space=space,
                        noise_scale=scale,
                        generator=g_noise,
                    )
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_losses.append(loss.detach().item())
            if log_steps:
                result.step_losses.append(loss.detach().item())

            # --- Extra consistency-only steps (matched-step fairness) ---
            # K extra optimiser steps per labelled-batch step, each on a
            # fresh unlabelled batch with the same noise process. No
            # supervised signal — pure MSE(z_a, z_b) on q.
            for _ in range(extra_consistency_steps_per_labeled_step):
                idx = torch.randint(0, pool_n, (n,), generator=g_trans)
                t_x = transductive_pool[idx]
                c_loss = consistency_only_loss(
                    model, t_x,
                    noise_std=noise_std,
                    noise_space=space,
                    noise_scale=scale,
                    generator=g_noise,
                )
                optim.zero_grad()
                c_loss.backward()
                optim.step()

        # Eval
        model.eval()
        with torch.no_grad():
            v_losses = []
            for xv, yv in val_loader:
                nv = xv.shape[0]
                preds = model(xv.view(nv, -1))
                v_losses.append(
                    torch.nn.functional.mse_loss(
                        preds, yv.float().view(nv, 1)
                    ).item()
                )
            t_losses: list[float] = []
            if test_loader is not None:
                for xt, yt in test_loader:
                    nt = xt.shape[0]
                    preds = model(xt.view(nt, -1))
                    t_losses.append(
                        torch.nn.functional.mse_loss(
                            preds, yt.float().view(nt, 1)
                        ).item()
                    )

        train_mse_epoch = sum(train_losses) / len(train_losses)
        val_mse_epoch = sum(v_losses) / len(v_losses)
        result.train_mse.append(train_mse_epoch)
        result.val_mse.append(val_mse_epoch)
        if test_loader is not None:
            test_mse_epoch = sum(t_losses) / len(t_losses)
            result.test_mse.append(test_mse_epoch)
            if test_mse_epoch < result.test_mse_best:
                result.test_mse_best = test_mse_epoch
                result.test_mse_best_epoch = epoch
        if val_mse_epoch < result.val_mse_best:
            result.val_mse_best = val_mse_epoch
            result.val_mse_best_epoch = epoch
            # Capture the held-out test number AT THE EPOCH WE'D EARLY-STOP AT.
            # This is the unbiased reporting value: epoch chosen by val, eval on test.
            if test_loader is not None:
                result.test_mse_at_best_val_epoch = result.test_mse[-1]

    result.seconds = time.time() - t0
    return result


# --------- exp2: ACR on 1D ------------------------------------------------------


@dataclass
class ACRRunResult:
    variant: str
    seed: int
    config: dict = field(default_factory=dict)
    train_loss: list = field(default_factory=list)
    test_mse: list = field(default_factory=list)
    test_mse_final: float = float("nan")
    sigma_grid_x: list = field(default_factory=list)
    sigma_grid_value: list = field(default_factory=list)
    seconds: float = 0.0


def run_acr_experiment(
    model: nn.Module,
    data: dict,
    sigma_fn_factory: Callable[[nn.Module, Tensor], Callable] | None,
    *,
    variant: str,
    seed: int,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    consistency_weight_max: float = 1.0,
    rampup_epochs: int = 80,
    sigma_grid_n: int = 200,
    config: dict | None = None,
) -> ACRRunResult:
    """Train one model on the synthetic 1D regression with optional consistency.

    ``sigma_fn_factory(model, labeled_x) -> sigma_fn`` is rebuilt each epoch
    so signal-driven variants can use the *current* model state. Pass ``None``
    for the supervised-only baseline.
    """
    seed_everything(seed)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    rng_eps = torch.Generator().manual_seed(seed + 7)
    rng_lab = torch.Generator().manual_seed(seed + 11)
    rng_unl = torch.Generator().manual_seed(seed + 13)

    x_lab, y_lab = data["x_lab"], data["y_lab"]
    x_unl = data["x_unl"]
    x_test, y_test = data["x_test"], data["y_test"]

    n_lab = x_lab.shape[0]
    n_unl = x_unl.shape[0]

    result = ACRRunResult(variant=variant, seed=seed, config=config or {})
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        lam = lambda_rampup(epoch, n_rampup=rampup_epochs, max_lambda=consistency_weight_max)

        # One pass over labeled data; sample matching unlabeled batches.
        perm = torch.randperm(n_lab, generator=rng_lab)
        epoch_losses = []
        for i in range(0, n_lab, batch_size):
            ids = perm[i : i + batch_size]
            xb, yb = x_lab[ids], y_lab[ids]

            sup, _ = supervised_mse_loss(model, xb, yb)
            total = sup

            if sigma_fn_factory is not None and lam > 0:
                # Build sigma_fn against the current model.
                sigma_fn = sigma_fn_factory(model, x_lab)
                u_ids = torch.randint(0, n_unl, (batch_size,), generator=rng_unl)
                qb = x_unl[u_ids]
                cons = consistency_loss(model, qb, sigma_fn, generator=rng_eps)
                total = total + lam * cons

            optim.zero_grad()
            total.backward()
            optim.step()
            epoch_losses.append(total.detach().item())

        result.train_loss.append(sum(epoch_losses) / len(epoch_losses))
        result.test_mse.append(evaluate_mse(model, x_test, y_test))

    # Sample σ(q) on a fixed grid at convergence for plotting.
    import math

    grid = torch.linspace(0.0, 2.0 * math.pi, sigma_grid_n).unsqueeze(1)
    if sigma_fn_factory is not None:
        sigma_fn = sigma_fn_factory(model, x_lab)
        # Don't wrap in no_grad — σ_jac needs autograd active to compute ‖∂f/∂q‖.
        # Each sigma_fn detaches its own outputs.
        sigma_vals = sigma_fn(grid).detach().squeeze(-1).cpu().tolist()
    else:
        sigma_vals = [0.0] * sigma_grid_n
    result.sigma_grid_x = grid.squeeze(-1).cpu().tolist()
    result.sigma_grid_value = sigma_vals

    result.test_mse_final = result.test_mse[-1]
    result.seconds = time.time() - t0
    return result


def save_run(path: Path, run) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(asdict(run), f, indent=2)
