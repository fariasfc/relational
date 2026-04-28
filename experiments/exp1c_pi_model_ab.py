"""Experiment 1c — direct A/B: v17 (relational + noisy) vs v18 (v16 + plain Π-model).

By the exp1 algebra,
    v17 = v18 + (mean-zero cross term)
so v17 and v18 should be statistically indistinguishable in expectation.
If the empirics confirm that, the relational wrapping in v17 is cosmetic
and we should adopt v18 (= v16 + plain Π-model on q with matched σ) as
the headline method going forward.

Setup mirrors exp1b — n_train ∈ {100, 250, 1 000}, 5 seeds, σ = 0.5
variance-scaled. Crucially, v17 and v18 share *identical* RNG seeds for
model init, data shuffling, transductive sampling, and noise draws — so
the only difference between the two trajectories is the loss formula
itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from relational.data import mnist_loaders, mnist_regression
from relational.models import PDFRegressor
from relational.plotting import save_curves
from relational.train import run_pdf_experiment, save_run


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT = ROOT / "results" / "exp1c"

LABEL_BUDGETS = [100, 250, 1_000]
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
SIGMA = 0.5  # the σ that worked in exp1b


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    pool, _ = mnist_regression(DATA_DIR, n_train=5_000, n_val=10, seed=999)
    pool_x = torch.stack([x for x, _ in pool])
    pixel_std = pool_x.view(pool_x.shape[0], -1).std(dim=0)

    _, val_set = mnist_regression(DATA_DIR, n_train=10, n_val=1_000, seed=0)

    runs: dict[str, list] = {}

    for n_train in LABEL_BUDGETS:
        print(f"\n=== label budget: {n_train} ===")
        train_set, _ = mnist_regression(DATA_DIR, n_train=n_train, n_val=10, seed=0)
        train_loader, val_loader = mnist_loaders(
            train_set, val_set, batch_size=BATCH_SIZE, seed=0
        )

        for seed in SEEDS:
            for variant in ("v16", "v17_input_var", "v18_pi_input_var"):
                key = f"{variant}@n{n_train}"
                runs.setdefault(key, [])
                model = PDFRegressor()
                # Pass σ to v17/v18, ignored for v16.
                kwargs = {}
                if variant != "v16":
                    kwargs["noise_std"] = SIGMA
                    kwargs["input_noise_scale"] = pixel_std
                res = run_pdf_experiment(
                    model, train_loader, val_loader, pool_x,
                    variant=variant, epochs=EPOCHS, lr=LR, seed=seed,
                    **kwargs,
                )
                save_run(OUT / f"{variant}_n{n_train}_seed{seed}.json", res)
                runs[key].append(res)
                print(
                    f"  n={n_train} seed={seed} {variant:20s}: "
                    f"val_mse={res.val_mse[-1]:.4f} ({res.seconds:.1f}s)"
                )

    # --- summary + paired comparison ---
    def stats(key):
        vals = [r.val_mse[-1] for r in runs[key]]
        m = sum(vals) / len(vals)
        v = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)
        return m, v ** 0.5, vals

    means = {k: stats(k)[0] for k in runs}
    stds = {k: stats(k)[1] for k in runs}

    # Per-seed paired diffs between v17 and v18 — the right way to
    # compare with shared RNG. If v17 ≡ v18 in expectation, the paired
    # diffs should be small relative to the cross-method spread.
    paired = {}
    for n in LABEL_BUDGETS:
        v17 = stats(f"v17_input_var@n{n}")[2]
        v18 = stats(f"v18_pi_input_var@n{n}")[2]
        diffs = [a - b for a, b in zip(v17, v18)]
        mean_d = sum(diffs) / len(diffs)
        std_d = (sum((d - mean_d) ** 2 for d in diffs) / max(1, len(diffs) - 1)) ** 0.5
        # Standard error of the paired-diff mean.
        sem = std_d / (len(diffs) ** 0.5)
        paired[f"n={n}"] = {
            "v17_minus_v18_per_seed": diffs,
            "mean_diff": mean_d,
            "std_diff": std_d,
            "sem_diff": sem,
            # If |mean| < 2*sem, the two methods are indistinguishable at
            # this seed budget.
            "indistinguishable_2sem": abs(mean_d) < 2 * sem,
        }

    summary = {
        "label_budgets": LABEL_BUDGETS,
        "seeds": SEEDS,
        "sigma": SIGMA,
        "val_mse_mean": means,
        "val_mse_std": stds,
        "paired_v17_minus_v18": paired,
    }
    with (OUT / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # --- plots ---
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(LABEL_BUDGETS), figsize=(13, 4))
    for ax, n in zip(axes, LABEL_BUDGETS):
        ks = [f"v16@n{n}", f"v17_input_var@n{n}", f"v18_pi_input_var@n{n}"]
        labels = ["v16", "v17_input_var\n(σ=0.5)", "v18_pi_input_var\n(σ=0.5)"]
        ms = [means[k] for k in ks]
        es = [stds[k] for k in ks]
        ax.bar(range(len(ks)), ms, yerr=es, capsize=4)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("val MSE")
        ax.set_title(f"n_train = {n}")
    fig.suptitle("exp1c — v17 (relational+noisy) vs v18 (v16 + plain Π-model)")
    fig.tight_layout()
    fig.savefig(OUT / "summary.png", dpi=140)
    plt.close(fig)

    # Paired-diff plot: per-seed differences should straddle zero if
    # v17 ≡ v18 in expectation.
    fig, ax = plt.subplots(figsize=(8, 4))
    for n in LABEL_BUDGETS:
        diffs = paired[f"n={n}"]["v17_minus_v18_per_seed"]
        ax.scatter([n] * len(diffs), diffs, alpha=0.6, s=40)
        ax.scatter([n], [paired[f"n={n}"]["mean_diff"]],
                   marker="_", s=300, color="red")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("n_train")
    ax.set_ylabel("v17 val_mse − v18 val_mse  (per seed)")
    ax.set_title("exp1c — paired-difference test")
    fig.tight_layout()
    fig.savefig(OUT / "paired_diff.png", dpi=140)
    plt.close(fig)

    print("\n=== summary ===")
    for n in LABEL_BUDGETS:
        print(f"\nn_train = {n}")
        for k in (f"v16@n{n}", f"v17_input_var@n{n}", f"v18_pi_input_var@n{n}"):
            print(f"  {k:30s}: {means[k]:.4f} ± {stds[k]:.4f}")
        p = paired[f"n={n}"]
        verdict = "indistinguishable" if p["indistinguishable_2sem"] else "DIFFER"
        print(
            f"  paired (v17 − v18): mean = {p['mean_diff']:+.4f} "
            f"(sem = {p['sem_diff']:.4f})  →  {verdict}"
        )
    print(f"\nWrote results to {OUT}")


if __name__ == "__main__":
    main()
