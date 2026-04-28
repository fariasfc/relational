"""Experiment 1f — learning curve and inductive-vs-transductive comparison.

Two questions:

1. **How does the SSL benefit scale with labelled-data budget?** Sweeps
   ``n_train ∈ {50, 100, 200, 500, 1000, 2000, 5000}``. We expect the
   v18-vs-v15 gap to be largest at low budgets and shrink as supervision
   becomes plentiful.

2. **Does using the test set's *inputs* (without labels) as the
   unlabelled pool — the classical transductive setting — beat using a
   separate unlabelled pool of the same size?** Directly relevant to
   molecular property prediction, where the screening set is typically
   known in advance.

Variants (each at every budget × seed):

* ``v15``                 — supervised baseline.
* ``v16``                 — supervised + reversal-augmentation (no
                            transductive forwards contribute gradient).
* ``v17_inductive``       — relational + noisy transductive, separate
                            unlabelled pool from MNIST train.
* ``v17_transductive``    — same loss but pool = val-set inputs.
* ``v18_inductive``       — v16 + plain Π-model on q, separate pool.
* ``v18_transductive``    — same loss; pool = val-set inputs.

For the transductive variants the val labels are never used during
training — only the input tensors. This is the actual Vapnik framing.

Reported numbers use **best val MSE during training** so long-running
configurations aren't penalised by late-epoch drift.
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
OUT = ROOT / "results" / "exp1f"

LABEL_BUDGETS = [50, 100, 200, 500, 1_000, 2_000, 5_000]
SEEDS = [0, 1, 2]
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
SIGMA = 0.5


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Inductive pool: separate slice of MNIST. Held constant across all
    # runs so the only thing that varies between inductive and
    # transductive is *which* unlabelled tensors the consistency loss
    # sees.
    pool_inductive_set, _ = mnist_regression(
        DATA_DIR, n_train=2_000, n_val=10, seed=999
    )
    pool_inductive = torch.stack([x for x, _ in pool_inductive_set])
    # Per-pixel std for variance scaling — computed once on the inductive
    # pool. (Using a separate pool's std for the transductive scale keeps
    # the noise process identical across the two.)
    pixel_std = pool_inductive.view(pool_inductive.shape[0], -1).std(dim=0)

    # Validation set (1 000 samples). For the *transductive* variants we
    # also use this set's INPUTS as the unlabelled pool — labels are
    # never touched.
    _, val_set = mnist_regression(DATA_DIR, n_train=10, n_val=1_000, seed=0)
    pool_transductive = torch.stack([x for x, _ in val_set])

    # (variant, label, pool_kind, pool_tensor)
    plans: list[tuple[str, str, str, torch.Tensor | None]] = [
        ("v15",              "v15",              "none",         None),
        ("v16",              "v16",              "inductive",    pool_inductive),
        ("v17_input_var",    "v17_inductive",    "inductive",    pool_inductive),
        ("v17_input_var",    "v17_transductive", "transductive", pool_transductive),
        ("v18_pi_input_var", "v18_inductive",    "inductive",    pool_inductive),
        ("v18_pi_input_var", "v18_transductive", "transductive", pool_transductive),
    ]

    runs: dict[str, list] = {}

    for n_train in LABEL_BUDGETS:
        print(f"\n=== label budget: {n_train} ===")
        train_set, _ = mnist_regression(DATA_DIR, n_train=n_train, n_val=10, seed=0)
        train_loader, val_loader = mnist_loaders(
            train_set, val_set, batch_size=BATCH_SIZE, seed=0
        )

        for seed in SEEDS:
            for variant, label, pool_kind, pool in plans:
                key = f"{label}@n{n_train}"
                runs.setdefault(key, [])

                kwargs = {}
                if variant.startswith(("v17_", "v18_")):
                    kwargs["noise_std"] = SIGMA
                    kwargs["input_noise_scale"] = pixel_std

                # v15 doesn't use the pool. We pass `pool_inductive` so
                # the function signature is satisfied; it's never read.
                pool_for_run = pool if pool is not None else pool_inductive

                model = PDFRegressor()
                res = run_pdf_experiment(
                    model, train_loader, val_loader, pool_for_run,
                    variant=variant, epochs=EPOCHS, lr=LR, seed=seed,
                    **kwargs,
                )
                save_run(OUT / f"{label}_n{n_train}_seed{seed}.json", res)
                runs[key].append(res)
                print(
                    f"  n={n_train} seed={seed} {label:18s} ({pool_kind:12s}): "
                    f"best={res.val_mse_best:.4f} (epoch {res.val_mse_best_epoch}, "
                    f"{res.seconds:.1f}s)"
                )

    # --- aggregate ---
    def stats(key):
        vals = [r.val_mse_best for r in runs[key]]
        m = sum(vals) / len(vals)
        v = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)
        return m, v ** 0.5

    means = {k: stats(k)[0] for k in runs}
    stds = {k: stats(k)[1] for k in runs}

    summary = {
        "label_budgets": LABEL_BUDGETS,
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "sigma": SIGMA,
        "val_mse_best_mean": means,
        "val_mse_best_std": stds,
        # Δ vs v15 (positive = better than supervised)
        "delta_vs_v15": {
            f"{label}@n{n}": means[f"v15@n{n}"] - means[f"{label}@n{n}"]
            for n in LABEL_BUDGETS
            for label in ("v16", "v17_inductive", "v17_transductive",
                          "v18_inductive", "v18_transductive")
        },
        # Δ transductive vs inductive at each budget (positive = transductive better)
        "delta_transductive_vs_inductive": {
            f"v18@n{n}": means[f"v18_inductive@n{n}"] - means[f"v18_transductive@n{n}"]
            for n in LABEL_BUDGETS
        }
        | {
            f"v17@n{n}": means[f"v17_inductive@n{n}"] - means[f"v17_transductive@n{n}"]
            for n in LABEL_BUDGETS
        },
    }
    with (OUT / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # --- plots ---
    import matplotlib.pyplot as plt

    # Main learning curve: val MSE vs n_train on log-log axes
    fig, ax = plt.subplots(figsize=(8, 5))
    style = {
        "v15":              dict(label="v15 (supervised)",        color="black",     marker="o", linestyle=":"),
        "v16":              dict(label="v16 (sup+reversal-aug)",  color="tab:gray",  marker="s", linestyle="--"),
        "v17_inductive":    dict(label="v17 inductive pool",      color="tab:orange", marker="^"),
        "v17_transductive": dict(label="v17 transductive (val)",  color="tab:red",   marker="v"),
        "v18_inductive":    dict(label="v18 inductive pool",      color="tab:green", marker="D"),
        "v18_transductive": dict(label="v18 transductive (val)",  color="tab:purple", marker="*"),
    }
    for label, sty in style.items():
        ys = [means[f"{label}@n{n}"] for n in LABEL_BUDGETS]
        es = [stds[f"{label}@n{n}"] for n in LABEL_BUDGETS]
        ax.errorbar(LABEL_BUDGETS, ys, yerr=es, capsize=3, **sty)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n_train (labelled samples)")
    ax.set_ylabel("best val MSE (mean ± std over seeds)")
    ax.set_title("exp1f — learning curve: SSL benefit vs labelled-data budget")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "learning_curve.png", dpi=140)
    plt.close(fig)

    # Δ vs v15 plot — shows the absolute MSE reduction from each method
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, sty in style.items():
        if label == "v15":
            continue
        deltas = [
            means[f"v15@n{n}"] - means[f"{label}@n{n}"]
            for n in LABEL_BUDGETS
        ]
        ax.plot(LABEL_BUDGETS, deltas, **sty)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("n_train")
    ax.set_ylabel("Δ val MSE vs v15  (positive = better than supervised)")
    ax.set_title("exp1f — absolute SSL benefit by label budget")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "delta_vs_supervised.png", dpi=140)
    plt.close(fig)

    # Transductive vs inductive head-to-head
    fig, ax = plt.subplots(figsize=(8, 4.5))
    v17_diff = [
        means[f"v17_inductive@n{n}"] - means[f"v17_transductive@n{n}"]
        for n in LABEL_BUDGETS
    ]
    v18_diff = [
        means[f"v18_inductive@n{n}"] - means[f"v18_transductive@n{n}"]
        for n in LABEL_BUDGETS
    ]
    ax.plot(LABEL_BUDGETS, v17_diff, marker="o", color="tab:orange", label="v17")
    ax.plot(LABEL_BUDGETS, v18_diff, marker="s", color="tab:green", label="v18")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("n_train")
    ax.set_ylabel("inductive − transductive  (positive = transductive wins)")
    ax.set_title("exp1f — transductive (val-set pool) vs inductive (separate pool)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "transductive_vs_inductive.png", dpi=140)
    plt.close(fig)

    print("\n=== summary (best val MSE, mean ± std over seeds) ===")
    print(f"{'n_train':>8s}  " + "  ".join(f"{lab:>18s}" for lab in style))
    for n in LABEL_BUDGETS:
        row = [f"{n:>8d}"]
        for lab in style:
            row.append(f"{means[f'{lab}@n{n}']:>10.4f} ± {stds[f'{lab}@n{n}']:.3f}")
        print("  ".join(row))

    print("\n=== Δ transductive vs inductive (positive = transductive wins) ===")
    for n in LABEL_BUDGETS:
        v17_d = means[f"v17_inductive@n{n}"] - means[f"v17_transductive@n{n}"]
        v18_d = means[f"v18_inductive@n{n}"] - means[f"v18_transductive@n{n}"]
        print(f"  n={n:5d}: v17 = {v17_d:+.4f}, v18 = {v18_d:+.4f}")

    print(f"\nWrote results to {OUT}")


if __name__ == "__main__":
    main()
