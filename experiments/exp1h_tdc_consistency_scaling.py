"""Experiment 1h-TDC — K-sweep on TDC Lipophilicity_AstraZeneca.

Mirror of ``exp1h_consistency_scaling.py`` for the molecular setting.
Question: does the K-optimum-shifts-with-labelled-budget pattern we saw
on MNIST regression transfer to a real molecular regression task with
Mordred descriptors?
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from relational.data_tdc import (
    load_tdc_lipophilicity,
    subsample_train,
    tdc_loaders,
)
from relational.models import TabularMLP
from relational.train import run_pdf_experiment, save_run


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "exp1h_tdc"

LABEL_BUDGETS = [250, 1_000]
SEEDS = [0, 1, 2]
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
SIGMA = 0.5
K_SWEEP = [0, 1, 3, 10, 30, 100]


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("[exp1h_tdc] Loading TDC Lipophilicity_AstraZeneca + Mordred features…")
    ds = load_tdc_lipophilicity()
    in_dim = ds.train.X.shape[1]
    print(f"  train={ds.train.X.shape[0]} val={ds.val.X.shape[0]} test={ds.test.X.shape[0]}  d={in_dim}")

    runs: dict[str, list] = {}

    for n_train in LABEL_BUDGETS:
        # Cap K for the larger budget — n=1000 with K=100 has 31 batches × 50 epochs
        # × 101 steps/batch = 156 550 optimiser steps per run, ~5 min/run × 3 seeds.
        k_for_budget = K_SWEEP if n_train == 250 else [k for k in K_SWEEP if k <= 30]

        print(f"\n=== label budget: {n_train} ===")
        for seed in SEEDS:
            labeled, unlabeled = subsample_train(ds.train, n_labeled=n_train, seed=seed)
            train_loader, val_loader, test_loader = tdc_loaders(
                labeled, ds.val, ds.test, batch_size=BATCH_SIZE, seed=0
            )
            pool = unlabeled.X if unlabeled.X.shape[0] > 0 else labeled.X
            scale = labeled.X.std(dim=0).clamp_min(1e-6)

            # v15 reference at this budget for context.
            ref_key = f"v15@n{n_train}"
            runs.setdefault(ref_key, [])
            model = TabularMLP(in_dim=in_dim)
            res = run_pdf_experiment(
                model, train_loader, val_loader, pool,
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
                model = TabularMLP(in_dim=in_dim)
                res = run_pdf_experiment(
                    model, train_loader, val_loader, pool,
                    variant="v18_pi_input_var", epochs=EPOCHS, lr=LR, seed=seed,
                    noise_std=SIGMA, input_noise_scale=scale,
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
        "task": ds.name,
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, n in zip(axes, LABEL_BUDGETS):
        k_for_budget = K_SWEEP if n == 250 else [k for k in K_SWEEP if k <= 30]
        xs = [k + 1 for k in k_for_budget]
        best = [summary["val_best"][f"K={k}@n{n}"] for k in k_for_budget]
        best_std = [summary["val_best_std"][f"K={k}@n{n}"] for k in k_for_budget]
        final = [summary["val_final"][f"K={k}@n{n}"] for k in k_for_budget]
        test = [summary["test_at_best_val"][f"K={k}@n{n}"] for k in k_for_budget]
        ax.errorbar(xs, best, yerr=best_std, capsize=3, marker="o",
                    color="tab:blue", label="best val MSE")
        ax.plot(xs, final, marker="x", linestyle="--", color="tab:red",
                label="final val MSE (drift if > best)")
        ax.plot(xs, test, marker="s", color="tab:green", linestyle=":",
                label="held-out scaffold test @ best-val")
        ax.axhline(summary["val_best"][f"v15@n{n}"], color="black", linestyle="-.",
                   label="v15 supervised (best val)")
        ax.set_xscale("log")
        ax.set_xlabel("K + 1 (extra consistency-only steps per labelled step)")
        ax.set_ylabel("MSE")
        ax.set_title(f"n_labeled = {n}")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("exp1h_tdc — K-sweep on Lipophilicity_AZ (Mordred + TabularMLP)")
    fig.tight_layout()
    fig.savefig(OUT / "k_sweep.png", dpi=140)
    plt.close(fig)

    print("\n=== summary (mean over seeds) ===")
    for n in LABEL_BUDGETS:
        print(f"\nn_labeled = {n}")
        v15_v = summary["val_best"][f"v15@n{n}"]
        print(f"  v15 (supervised reference): val_best = {v15_v:.4f}")
        k_for_budget = K_SWEEP if n == 250 else [k for k in K_SWEEP if k <= 30]
        for k in k_for_budget:
            key = f"K={k}@n{n}"
            print(
                f"  K={k:>3d}: val_best={summary['val_best'][key]:.4f}±{summary['val_best_std'][key]:.4f}  "
                f"final={summary['val_final'][key]:.4f}  drift={summary['drift'][key]:+.4f}  "
                f"best_ep≈{summary['best_epoch'][key]:.1f}  "
                f"test@best={summary['test_at_best_val'][key]:.4f}"
            )
    print(f"\nWrote results to {OUT}")


if __name__ == "__main__":
    main()
