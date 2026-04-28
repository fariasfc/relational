"""Experiment 1h — extrapolate the consistency-step ratio K.

exp1e fixed K = 3 (3 unlabelled-only optimiser steps per labelled step).
Here we sweep K ∈ {0, 1, 3, 10, 30, 100} at the SSL sweet spot from
exp1f (n_train = 250) and at one larger budget (n_train = 1 000) to see
how the SSL benefit scales when we let the consistency phase dominate.

Two competing effects:

* Larger K  → stronger smoothness regularisation per labelled signal.
* Larger K  → less frequent supervised anchor; ``MSE(z_a, z_b)`` has a
              trivial minimum at ``f ≡ const``, so the model can drift
              toward predicting a constant.

We therefore report:

* ``val_mse_best``   — early-stopping proxy.
* ``val_mse_final``  — last-epoch val MSE; large gap to ``best`` flags drift.
* ``val_mse_best_epoch`` — when best happened. If best happens at epoch 1
                            and the curve climbs after, we are in collapse.

We also evaluate on the held-out test split (``test_mse_at_best_val_epoch``)
to keep the unbiased reporting protocol from exp1g.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from relational.data import (
    mnist_loaders_with_test,
    mnist_regression_with_test,
)
from relational.models import PDFRegressor
from relational.train import run_pdf_experiment, save_run


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT = ROOT / "results" / "exp1h"

LABEL_BUDGETS = [250, 1_000]
SEEDS = [0, 1, 2]
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
SIGMA = 0.5
K_SWEEP = [0, 1, 3, 10, 30, 100]   # 0 = v18_1x; 3 = v18_split from exp1e


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    pool_inductive_set, _, _ = mnist_regression_with_test(
        DATA_DIR, n_train=2_000, n_val=10, n_test=10, seed=999
    )
    pool_inductive = torch.stack([x for x, _ in pool_inductive_set])
    pixel_std = pool_inductive.view(pool_inductive.shape[0], -1).std(dim=0)

    _, val_set, test_set = mnist_regression_with_test(
        DATA_DIR, n_train=10, n_val=1_000, n_test=1_000, seed=0
    )

    runs: dict[str, list] = {}

    for n_train in LABEL_BUDGETS:
        # n=1000 with K=100 = ~155k optimiser steps per run = ~13 min × 3 seeds.
        # We cap K at 30 for the larger budget to keep total runtime tractable.
        k_for_budget = K_SWEEP if n_train == 250 else [k for k in K_SWEEP if k <= 30]

        print(f"\n=== label budget: {n_train} ===")
        train_set, _, _ = mnist_regression_with_test(
            DATA_DIR, n_train=n_train, n_val=10, n_test=10, seed=0
        )
        train_loader, val_loader, test_loader = mnist_loaders_with_test(
            train_set, val_set, test_set, batch_size=BATCH_SIZE, seed=0
        )

        for seed in SEEDS:
            # v15 reference (no consistency) at this budget for context.
            ref_key = f"v15@n{n_train}"
            runs.setdefault(ref_key, [])
            model = PDFRegressor()
            res = run_pdf_experiment(
                model, train_loader, val_loader, pool_inductive,
                variant="v15", epochs=EPOCHS, lr=LR, seed=seed,
                test_loader=test_loader,
            )
            save_run(OUT / f"v15_n{n_train}_seed{seed}.json", res)
            runs[ref_key].append(res)
            print(
                f"  n={n_train} seed={seed} v15:    "
                f"val_best={res.val_mse_best:.4f} test@best={res.test_mse_at_best_val_epoch:.4f} "
                f"(epoch {res.val_mse_best_epoch}, {res.seconds:.1f}s)"
            )

            for k in k_for_budget:
                key = f"K={k}@n{n_train}"
                runs.setdefault(key, [])
                model = PDFRegressor()
                res = run_pdf_experiment(
                    model, train_loader, val_loader, pool_inductive,
                    variant="v18_pi_input_var", epochs=EPOCHS, lr=LR, seed=seed,
                    noise_std=SIGMA, input_noise_scale=pixel_std,
                    extra_consistency_steps_per_labeled_step=k,
                    test_loader=test_loader,
                )
                save_run(OUT / f"K{k}_n{n_train}_seed{seed}.json", res)
                runs[key].append(res)
                drift = res.val_mse[-1] - res.val_mse_best
                print(
                    f"  n={n_train} seed={seed} K={k:>3d}: "
                    f"val_best={res.val_mse_best:.4f} (ep {res.val_mse_best_epoch:>2d}) "
                    f"final={res.val_mse[-1]:.4f} drift={drift:+.3f} "
                    f"test@best={res.test_mse_at_best_val_epoch:.4f} "
                    f"({res.seconds:.1f}s)"
                )

    # --- aggregate ---
    def agg(key, attr_or_idx):
        if attr_or_idx == "final":
            vals = [r.val_mse[-1] for r in runs[key]]
        elif attr_or_idx == "drift":
            vals = [r.val_mse[-1] - r.val_mse_best for r in runs[key]]
        elif attr_or_idx == "best_epoch":
            vals = [r.val_mse_best_epoch for r in runs[key]]
        else:
            vals = [getattr(r, attr_or_idx) for r in runs[key]]
        m = sum(vals) / len(vals)
        v = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)
        return m, v ** 0.5

    summary: dict = {
        "label_budgets": LABEL_BUDGETS,
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "sigma": SIGMA,
        "K_sweep": K_SWEEP,
    }
    summary["val_best"] = {k: agg(k, "val_mse_best")[0] for k in runs}
    summary["val_best_std"] = {k: agg(k, "val_mse_best")[1] for k in runs}
    summary["val_final"] = {k: agg(k, "final")[0] for k in runs}
    summary["drift"] = {k: agg(k, "drift")[0] for k in runs}
    summary["best_epoch"] = {k: agg(k, "best_epoch")[0] for k in runs}
    summary["test_at_best_val"] = {k: agg(k, "test_mse_at_best_val_epoch")[0] for k in runs}
    summary["test_at_best_val_std"] = {k: agg(k, "test_mse_at_best_val_epoch")[1] for k in runs}

    with (OUT / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # --- plots ---
    import matplotlib.pyplot as plt

    # Main: best val MSE vs K, with v15 baseline as a horizontal reference.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, n in zip(axes, LABEL_BUDGETS):
        k_for_budget = K_SWEEP if n == 250 else [k for k in K_SWEEP if k <= 30]
        ks = k_for_budget
        # Use K=0+1 to avoid log(0); plot K on a "K+1" log axis for clarity.
        xs = [k + 1 for k in ks]
        best = [summary["val_best"][f"K={k}@n{n}"] for k in ks]
        best_std = [summary["val_best_std"][f"K={k}@n{n}"] for k in ks]
        final = [summary["val_final"][f"K={k}@n{n}"] for k in ks]
        test = [summary["test_at_best_val"][f"K={k}@n{n}"] for k in ks]
        ax.errorbar(xs, best, yerr=best_std, capsize=3, marker="o",
                    color="tab:blue", label="best val MSE")
        ax.plot(xs, final, marker="x", linestyle="--", color="tab:red",
                label="final val MSE (drift if > best)")
        ax.plot(xs, test, marker="s", color="tab:green", linestyle=":",
                label="held-out test @ best-val epoch")
        ax.axhline(summary["val_best"][f"v15@n{n}"], color="black", linestyle="-.",
                   label="v15 supervised (best val)")
        ax.set_xscale("log")
        ax.set_xlabel("K + 1  (extra consistency-only steps per labelled step, +1 for log axis)")
        ax.set_ylabel("MSE")
        ax.set_title(f"n_train = {n}")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("exp1h — does the SSL benefit keep growing with K?")
    fig.tight_layout()
    fig.savefig(OUT / "k_sweep.png", dpi=140)
    plt.close(fig)

    # Trajectories: val MSE per epoch at each K. Drift is most visible here.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.cm.viridis
    for ax, n in zip(axes, LABEL_BUDGETS):
        k_for_budget = K_SWEEP if n == 250 else [k for k in K_SWEEP if k <= 30]
        for k in k_for_budget:
            curves = [r.val_mse for r in runs[f"K={k}@n{n}"]]
            mean_curve = [
                sum(c[i] for c in curves) / len(curves)
                for i in range(len(curves[0]))
            ]
            color = cmap(k_for_budget.index(k) / max(1, len(k_for_budget) - 1))
            ax.plot(range(len(mean_curve)), mean_curve, label=f"K={k}", color=color)
        v15_curves = [r.val_mse for r in runs[f"v15@n{n}"]]
        v15_mean = [
            sum(c[i] for c in v15_curves) / len(v15_curves)
            for i in range(len(v15_curves[0]))
        ]
        ax.plot(range(len(v15_mean)), v15_mean, label="v15", color="black",
                linestyle=":", linewidth=1.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel("val MSE (mean over seeds)")
        ax.set_yscale("log")
        ax.set_title(f"n_train = {n}")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("exp1h — val-MSE trajectories at each K")
    fig.tight_layout()
    fig.savefig(OUT / "trajectories.png", dpi=140)
    plt.close(fig)

    # --- print ---
    print("\n=== summary (mean over seeds) ===")
    for n in LABEL_BUDGETS:
        print(f"\nn_train = {n}")
        v15_v = summary["val_best"][f"v15@n{n}"]
        print(f"  v15 (supervised reference): val_best = {v15_v:.4f}")
        k_for_budget = K_SWEEP if n == 250 else [k for k in K_SWEEP if k <= 30]
        for k in k_for_budget:
            key = f"K={k}@n{n}"
            print(
                f"  K={k:>3d}: val_best={summary['val_best'][key]:.4f}±{summary['val_best_std'][key]:.4f}  "
                f"final={summary['val_final'][key]:.4f}  "
                f"drift={summary['drift'][key]:+.4f}  "
                f"best_epoch≈{summary['best_epoch'][key]:.1f}  "
                f"test@best={summary['test_at_best_val'][key]:.4f}"
            )
    print(f"\nWrote results to {OUT}")


if __name__ == "__main__":
    main()
