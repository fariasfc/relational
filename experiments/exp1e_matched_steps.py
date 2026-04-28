"""Experiment 1e — matched-optimiser-step fairness control.

Setting: in label-expensive domains (e.g. molecular property prediction)
compute is cheap and labels are scarce. The right fairness control is
*not* matched compute — it is "given the same number of optimiser steps,
does spending some of them on unlabelled consistency beat spending all of
them on the labelled set?"

Six variants per label budget, two of each method family:

* ``v15_1x``       — supervised, ``epochs`` epochs.
* ``v15_4x``       — supervised, ``4 × epochs`` epochs (matched optimiser
                     steps to v17_split / v18_split, all spent on labels).
* ``v17_1x``       — relational + noisy transductive, 1× epochs.
* ``v17_split``    — same Phase-1 loss; after each labelled step do K=3
                     extra ``MSE(z_a, z_b)`` consistency-only steps. Total
                     optimiser-step count matches v15_4x.
* ``v18_1x``       — v16 + plain Π-model, 1× epochs.
* ``v18_split``    — same Phase-1 loss + K=3 extra consistency-only steps.

Both ``_split`` variants and ``v15_4x`` end up with **600 optimiser steps
at n_train = 100, batch 32, 50 epochs** — same parameter-update budget,
same wall-clock. The split variants spend 1/4 of those on
labelled+consistency, 3/4 on unlabelled-only consistency.

Decision rules:
  * v17/v18 _split ≫ v15_4x  → unlabelled data adds learning signal that
    cannot be obtained by re-processing labels. **The hypothesis the
    domain needs to be true.**
  * v15_4x ≈ v17/v18 _split  → v15_1x was just under-trained; SSL win was
    a compute confound.
  * v15_4x ≫ v17/v18 _split  → consistency-only steps drift / collapse.

Reported numbers use **best val MSE during training** (early-stopping
proxy) to avoid penalising long runs for late-epoch drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from relational.data import mnist_loaders, mnist_regression
from relational.models import PDFRegressor
from relational.train import run_pdf_experiment, save_run


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT = ROOT / "results" / "exp1e"

LABEL_BUDGETS = [100, 250, 1_000]
SEEDS = [0, 1, 2, 3, 4]
BASE_EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
SIGMA = 0.5
EXTRA_K = 3   # 3 consistency-only steps per labelled step → 4× total step count


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    pool, _ = mnist_regression(DATA_DIR, n_train=5_000, n_val=10, seed=999)
    pool_x = torch.stack([x for x, _ in pool])
    pixel_std = pool_x.view(pool_x.shape[0], -1).std(dim=0)

    _, val_set = mnist_regression(DATA_DIR, n_train=10, n_val=1_000, seed=0)

    # (variant, epochs, extra_consistency_steps_per_labeled_step, label)
    plans: list[tuple[str, int, int, str]] = [
        ("v15",                BASE_EPOCHS,     0, "v15_1x"),
        ("v15",                BASE_EPOCHS * 4, 0, "v15_4x"),
        ("v17_input_var",      BASE_EPOCHS,     0, "v17_1x"),
        ("v17_input_var",      BASE_EPOCHS,     EXTRA_K, "v17_split"),
        ("v18_pi_input_var",   BASE_EPOCHS,     0, "v18_1x"),
        ("v18_pi_input_var",   BASE_EPOCHS,     EXTRA_K, "v18_split"),
    ]

    runs: dict[str, list] = {}

    for n_train in LABEL_BUDGETS:
        print(f"\n=== label budget: {n_train} ===")
        train_set, _ = mnist_regression(DATA_DIR, n_train=n_train, n_val=10, seed=0)
        train_loader, val_loader = mnist_loaders(
            train_set, val_set, batch_size=BATCH_SIZE, seed=0
        )

        for seed in SEEDS:
            for variant, epochs, extra_k, label in plans:
                key = f"{label}@n{n_train}"
                runs.setdefault(key, [])

                kwargs = {}
                if variant != "v15":
                    kwargs["noise_std"] = SIGMA
                    kwargs["input_noise_scale"] = pixel_std

                model = PDFRegressor()
                res = run_pdf_experiment(
                    model, train_loader, val_loader, pool_x,
                    variant=variant, epochs=epochs, lr=LR, seed=seed,
                    extra_consistency_steps_per_labeled_step=extra_k,
                    **kwargs,
                )
                save_run(OUT / f"{label}_n{n_train}_seed{seed}.json", res)
                runs[key].append(res)
                print(
                    f"  n={n_train} seed={seed} {label:12s}: "
                    f"final={res.val_mse[-1]:.4f}, best={res.val_mse_best:.4f} "
                    f"(epoch {res.val_mse_best_epoch}, {res.seconds:.1f}s)"
                )

    # --- summary using best val MSE ---
    def stats(key, attr):
        vals = [getattr(r, attr) if not isinstance(attr, str) or attr in ("val_mse_best",)
                else r.val_mse[-1] for r in runs[key]]
        if attr == "val_mse_best":
            vals = [r.val_mse_best for r in runs[key]]
        elif attr == "final":
            vals = [r.val_mse[-1] for r in runs[key]]
        m = sum(vals) / len(vals)
        v = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)
        return m, v ** 0.5

    means_best = {k: stats(k, "val_mse_best")[0] for k in runs}
    stds_best = {k: stats(k, "val_mse_best")[1] for k in runs}
    means_final = {k: stats(k, "final")[0] for k in runs}

    summary = {
        "label_budgets": LABEL_BUDGETS,
        "seeds": SEEDS,
        "base_epochs": BASE_EPOCHS,
        "extra_k": EXTRA_K,
        "sigma": SIGMA,
        "val_mse_best_mean": means_best,
        "val_mse_best_std": stds_best,
        "val_mse_final_mean": means_final,
    }

    # Paired tests at each budget: v15_4x vs v17/v18_split with shared seeds.
    paired = {}
    for n in LABEL_BUDGETS:
        sub = {}
        for split_label in ("v17_split", "v18_split"):
            v15_4x = [r.val_mse_best for r in runs[f"v15_4x@n{n}"]]
            split = [r.val_mse_best for r in runs[f"{split_label}@n{n}"]]
            diffs = [a - b for a, b in zip(v15_4x, split)]  # positive = split better
            mean_d = sum(diffs) / len(diffs)
            std_d = (sum((d - mean_d) ** 2 for d in diffs) / max(1, len(diffs) - 1)) ** 0.5
            sem = std_d / (len(diffs) ** 0.5)
            sub[split_label] = {
                "diffs": diffs,
                "mean_diff": mean_d,
                "sem_diff": sem,
                # split wins if mean_diff > 2 sem (one-sided)
                "split_significantly_better": mean_d > 2 * sem,
            }
        paired[f"n={n}"] = sub
    summary["paired_v15_4x_minus_split"] = paired

    with (OUT / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # --- plots ---
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(LABEL_BUDGETS), figsize=(14, 4.5))
    variant_order = ["v15_1x", "v15_4x", "v17_1x", "v17_split", "v18_1x", "v18_split"]
    colors = ["tab:gray", "tab:blue", "tab:orange", "tab:red", "tab:green", "tab:purple"]
    for ax, n in zip(axes, LABEL_BUDGETS):
        keys = [f"{v}@n{n}" for v in variant_order]
        ms = [means_best[k] for k in keys]
        es = [stds_best[k] for k in keys]
        ax.bar(range(len(keys)), ms, yerr=es, capsize=4, color=colors)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(variant_order, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("best val MSE")
        ax.set_title(f"n_train = {n}")
    fig.suptitle(
        "exp1e — matched-step fairness control (best val MSE, mean ± std over 5 seeds)\n"
        f"split variants do {EXTRA_K} extra consistency-only steps per labelled step (= 4× total optimiser steps)"
    )
    fig.tight_layout()
    fig.savefig(OUT / "summary.png", dpi=140)
    plt.close(fig)

    # Val-MSE trajectory plot at one label budget — shows whether _split
    # variants drift after their best epoch (a known risk).
    fig, axes = plt.subplots(1, len(LABEL_BUDGETS), figsize=(14, 4))
    for ax, n in zip(axes, LABEL_BUDGETS):
        for v, c in zip(("v15_1x", "v15_4x", "v18_1x", "v18_split"),
                        ["tab:gray", "tab:blue", "tab:green", "tab:purple"]):
            curves = [r.val_mse for r in runs[f"{v}@n{n}"]]
            max_len = max(len(c) for c in curves)
            # pad shorter curves with their last value
            padded = [c + [c[-1]] * (max_len - len(c)) for c in curves]
            mean = [sum(p[i] for p in padded) / len(padded) for i in range(max_len)]
            ax.plot(range(max_len), mean, label=v, color=c)
        ax.set_xlabel("epoch")
        ax.set_ylabel("val MSE (mean over seeds)")
        ax.set_title(f"n_train = {n}")
        ax.legend(fontsize=8)
        ax.set_yscale("log")
    fig.suptitle("exp1e — val MSE trajectories")
    fig.tight_layout()
    fig.savefig(OUT / "trajectories.png", dpi=140)
    plt.close(fig)

    print("\n=== summary (best val MSE) ===")
    for n in LABEL_BUDGETS:
        print(f"\nn_train = {n}")
        for v in variant_order:
            k = f"{v}@n{n}"
            print(f"  {v:12s}: best={means_best[k]:.4f} ± {stds_best[k]:.4f}  "
                  f"(final={means_final[k]:.4f})")
        for split_label in ("v17_split", "v18_split"):
            p = paired[f"n={n}"][split_label]
            verdict = (
                "split BETTER" if p["split_significantly_better"]
                else "indistinguishable / split worse"
            )
            print(
                f"  paired (v15_4x − {split_label}): mean = {p['mean_diff']:+.4f} "
                f"(sem = {p['sem_diff']:.4f}) → {verdict}"
            )
    print(f"\nWrote results to {OUT}")


if __name__ == "__main__":
    main()
