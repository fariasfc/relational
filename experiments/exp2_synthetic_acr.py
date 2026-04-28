"""Experiment 2 — adaptive consistency on a 1D regression with non-uniform Lipschitz.

We compare on ``y*(x) = sin(x)`` for ``x ∈ [0, π]`` and ``5·sin(x)`` for
``(π, 2π]`` (5× Lipschitz contrast). Variants:

* ``supervised`` — labeled-only baseline (lower bound).
* ``fixed_best`` — Π-model consistency with the best of a small σ sweep.
* ``sigma_jac`` — σ ∝ 1 / ‖∂f/∂q‖, stop-gradient.
* ``sigma_dens`` — σ ∝ k-NN distance to labeled inputs.
* ``oracle``     — σ ∝ 1 / ‖∇y*(q)‖ (upper bound).

Two regimes: ``in_dist`` (labeled covers full domain) and ``shift``
(labeled only on the flat half). Five seeds each.

Pass criteria (docx Phase 2):
  in-dist: at least one adaptive variant closes ≥ 40 % of the
  (fixed-σ → oracle) gap. shift: ≥ 60 %.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch

from relational.data import synthetic_1d
from relational.models import TinyMLP
from relational.plotting import save_bar, save_curves, save_sigma_curves
from relational.sigma import (
    sigma_dens,
    sigma_fixed,
    sigma_jac,
    sigma_oracle_lipschitz,
    y_star_grad_1d,
)
from relational.train import run_acr_experiment, save_run


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "exp2"


SIGMA_MAX = 0.5
C_JAC = 0.1
C_ORACLE = 0.1


def _fixed_factory(value: float):
    def f(_, __):
        return sigma_fixed(value)

    return f


def _jac_factory(c: float, sigma_max: float):
    def f(model, _labeled_x):
        return sigma_jac(model, c=c, sigma_max=sigma_max)

    return f


def _dens_factory(sigma_max: float, k: int = 3):
    def f(_, labeled_x):
        return sigma_dens(labeled_x, sigma_max=sigma_max, k=k)

    return f


def _oracle_factory(c: float, sigma_max: float):
    def f(_, __):
        return sigma_oracle_lipschitz(y_star_grad_1d, c=c, sigma_max=sigma_max)

    return f


SIGMA_SWEEP = [0.01, 0.05, 0.1, 0.2, 0.5]


def run_regime(regime: str, shifted: bool, seeds: list[int]) -> dict:
    out = OUT / regime
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n=== regime: {regime} (shifted={shifted}) ===")

    # Generate a single dataset per regime; reuse across seeds so the only
    # source of variation is model init / training noise.
    data = synthetic_1d(
        n_labeled=50,
        n_unlabeled=2_000,
        n_test=1_000,
        shifted=shifted,
        seed=0,
    )
    # Save the dataset so plots can be reproduced.
    torch.save(data, out / "dataset.pt")

    runs: dict[str, list] = {}

    def run_variant(name: str, factory, config: dict):
        results = []
        for s in seeds:
            model = TinyMLP()
            res = run_acr_experiment(
                model,
                data,
                factory,  # may be None for the supervised baseline
                variant=name,
                seed=s,
                config=config,
            )
            save_run(out / f"{name}_seed{s}.json", res)
            results.append(res)
            print(
                f"  {name} seed={s}: test_mse={res.test_mse_final:.4f} "
                f"({res.seconds:.1f}s)"
            )
        runs[name] = results

    # --- Supervised (no consistency loss)
    run_variant("supervised", None, {})

    # --- Fixed-σ sweep: pick best by mean final test MSE.
    fixed_sweep = {}
    for sigma in SIGMA_SWEEP:
        name = f"fixed_sigma_{sigma}"
        run_variant(name, _fixed_factory(sigma), {"sigma": sigma})
        fixed_sweep[sigma] = sum(r.test_mse_final for r in runs[name]) / len(seeds)

    best_sigma = min(fixed_sweep, key=fixed_sweep.get)
    print(f"  -> best fixed σ = {best_sigma}  (mean test MSE = {fixed_sweep[best_sigma]:.4f})")
    runs["fixed_best"] = runs[f"fixed_sigma_{best_sigma}"]

    # --- Adaptive variants
    run_variant(
        "sigma_jac",
        _jac_factory(c=C_JAC, sigma_max=SIGMA_MAX),
        {"c": C_JAC, "sigma_max": SIGMA_MAX},
    )
    run_variant(
        "sigma_dens",
        _dens_factory(sigma_max=SIGMA_MAX, k=3),
        {"sigma_max": SIGMA_MAX, "k": 3},
    )
    run_variant(
        "oracle",
        _oracle_factory(c=C_ORACLE, sigma_max=SIGMA_MAX),
        {"c": C_ORACLE, "sigma_max": SIGMA_MAX},
    )

    # --- Aggregate ---
    def stats(name):
        vals = [r.test_mse_final for r in runs[name]]
        n = len(vals)
        m = sum(vals) / n
        v = sum((x - m) ** 2 for x in vals) / max(1, n - 1)
        return m, v ** 0.5

    summary_variants = ["supervised", "fixed_best", "sigma_jac", "sigma_dens", "oracle"]
    means = {n: stats(n)[0] for n in summary_variants}
    stds = {n: stats(n)[1] for n in summary_variants}

    fixed_mse = means["fixed_best"]
    oracle_mse = means["oracle"]
    gap = fixed_mse - oracle_mse  # >0 if oracle helps
    gap_closed: dict[str, float | None] = {}
    for n in ("sigma_jac", "sigma_dens"):
        if gap > 1e-3:  # there's meaningful headroom
            gap_closed[n] = (fixed_mse - means[n]) / gap
        else:
            # No headroom (oracle is no better than fixed-σ on this benchmark).
            # Reporting raw delta vs fixed instead, signed.
            gap_closed[n] = None

    # Headroom is the gap between fixed-σ and oracle. If <= 0 there is
    # nothing for an adaptive method to exploit on this benchmark.
    summary = {
        "regime": regime,
        "shifted": shifted,
        "fixed_sigma_sweep": fixed_sweep,
        "best_fixed_sigma": best_sigma,
        "test_mse_mean": means,
        "test_mse_std": stds,
        "fixed_to_oracle_gap": gap,
        "gap_closed_fraction": gap_closed,
        "delta_vs_fixed": {
            n: fixed_mse - means[n] for n in ("sigma_jac", "sigma_dens", "oracle")
        },
        "delta_vs_supervised": {
            n: means["supervised"] - means[n]
            for n in ("fixed_best", "sigma_jac", "sigma_dens", "oracle")
        },
    }
    with (out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # --- Plots ---
    save_bar(
        out / "summary.png",
        labels=summary_variants,
        means=[means[n] for n in summary_variants],
        stds=[stds[n] for n in summary_variants],
        ylabel="test MSE (mean ± std over 5 seeds)",
        title=f"exp2 / {regime}: test MSE",
    )

    # σ(q) curves at convergence — average across seeds.
    grid_x = runs["sigma_jac"][0].sigma_grid_x
    sigmas_to_plot = {}
    for n in ("sigma_jac", "sigma_dens", "oracle"):
        cols = [r.sigma_grid_value for r in runs[n]]
        avg = [sum(c[i] for c in cols) / len(cols) for i in range(len(grid_x))]
        sigmas_to_plot[n] = avg
    sigmas_to_plot[f"fixed σ={best_sigma}"] = [best_sigma] * len(grid_x)
    save_sigma_curves(
        out / "sigma_curves.png",
        grid_x,
        sigmas_to_plot,
        title=f"exp2 / {regime}: σ(q) at convergence (mean of seeds)",
    )

    # Test-MSE training curves for the headline variants.
    save_curves(
        out / "test_mse_curves.png",
        {
            "supervised": [
                sum(r.test_mse[i] for r in runs["supervised"]) / len(seeds)
                for i in range(len(runs["supervised"][0].test_mse))
            ],
            f"fixed σ={best_sigma}": [
                sum(r.test_mse[i] for r in runs["fixed_best"]) / len(seeds)
                for i in range(len(runs["fixed_best"][0].test_mse))
            ],
            "sigma_jac": [
                sum(r.test_mse[i] for r in runs["sigma_jac"]) / len(seeds)
                for i in range(len(runs["sigma_jac"][0].test_mse))
            ],
            "sigma_dens": [
                sum(r.test_mse[i] for r in runs["sigma_dens"]) / len(seeds)
                for i in range(len(runs["sigma_dens"][0].test_mse))
            ],
            "oracle": [
                sum(r.test_mse[i] for r in runs["oracle"]) / len(seeds)
                for i in range(len(runs["oracle"][0].test_mse))
            ],
        },
        ylabel="test MSE",
        title=f"exp2 / {regime}: test MSE during training",
        logy=True,
    )

    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [0, 1, 2, 3, 4]

    in_dist = run_regime("in_dist", shifted=False, seeds=seeds)
    shift = run_regime("shift", shifted=True, seeds=seeds)

    def _pass(s, threshold):
        if s["fixed_to_oracle_gap"] <= 1e-3:
            return None  # no headroom; criterion is undefined
        vals = [v for v in s["gap_closed_fraction"].values() if v is not None]
        return any(v >= threshold for v in vals)

    overall = {
        "in_dist": in_dist,
        "shift": shift,
        "pass_criteria": {
            "in_dist_min_gap_closed": 0.40,
            "shift_min_gap_closed": 0.60,
        },
        "in_dist_pass": _pass(in_dist, 0.40),
        "shift_pass": _pass(shift, 0.60),
    }
    with (OUT / "summary.json").open("w") as f:
        json.dump(overall, f, indent=2)

    print()
    print("=== exp2 summary ===")
    for regime, s in (("in_dist", in_dist), ("shift", shift)):
        print(f"\n{regime}:")
        for n, m in s["test_mse_mean"].items():
            print(f"  {n:14s}: {m:.4f} ± {s['test_mse_std'][n]:.4f}")
        print(f"  best fixed σ : {s['best_fixed_sigma']}")
        print(f"  fixed→oracle headroom : {s['fixed_to_oracle_gap']:+.4f}")
        gc_pieces = []
        for n, v in s["gap_closed_fraction"].items():
            gc_pieces.append(f"{n}={'undef' if v is None else f'{v:+.1%}'}")
        print(f"  gap closed   : " + ", ".join(gc_pieces))
    print()
    in_p = overall["in_dist_pass"]
    sh_p = overall["shift_pass"]
    print(f"in-dist pass (≥40% gap closed): {'no headroom' if in_p is None else in_p}")
    print(f"shift   pass (≥60% gap closed): {'no headroom' if sh_p is None else sh_p}")
    print(f"Wrote results to {OUT}")


if __name__ == "__main__":
    main()
