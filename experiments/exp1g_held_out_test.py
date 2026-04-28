"""Experiment 1g — exp1f rerun on a held-out test split.

exp1f conflated three uses of the same MNIST test split: per-epoch eval,
"best epoch" selection, and the transductive unlabelled pool. For
inductive variants this is just standard model-selection bias (small);
for transductive variants it's worse because the model is *explicitly
tailored* to the eval set's input tensors at training time.

Here we split the 10 000-sample MNIST test set into:

* ``val_set``  — 1 000 samples, used for "best epoch" selection.
* ``test_set`` — 1 000 disjoint samples, **never** seen during training,
                 used only for unbiased final reporting.

Variants and grid match exp1f. We report two numbers per cell:

* ``val_mse_best``                  — the in-domain number (== exp1f's metric).
* ``test_mse_at_best_val_epoch``    — held-out, unbiased.

Decision logic for transductive vs inductive:

* If ``val ≈ test`` for transductive  → genuine Vapnik benefit; tailoring
                                         to known inputs also generalises.
* If ``val ≪ test`` for transductive  → tailoring artifact; the win was
                                         specific to the points the model
                                         saw during training.
* For inductive variants the val/test gap is just standard model-selection
  bias and should be small at any label budget.
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
OUT = ROOT / "results" / "exp1g"

LABEL_BUDGETS = [50, 100, 200, 500, 1_000, 2_000, 5_000]
SEEDS = [0, 1, 2]
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
SIGMA = 0.5


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Inductive pool — separate slice of MNIST train. Held constant across runs.
    pool_inductive_set, _, _ = mnist_regression_with_test(
        DATA_DIR, n_train=2_000, n_val=10, n_test=10, seed=999
    )
    pool_inductive = torch.stack([x for x, _ in pool_inductive_set])
    pixel_std = pool_inductive.view(pool_inductive.shape[0], -1).std(dim=0)

    # The test split is 1 000 samples disjoint from val. Both come from
    # the MNIST test set (10 000 total). val_set is also the transductive
    # pool for the transductive variants.
    _, val_set, test_set = mnist_regression_with_test(
        DATA_DIR, n_train=10, n_val=1_000, n_test=1_000, seed=0
    )
    pool_transductive = torch.stack([x for x, _ in val_set])

    # Sanity: val and test are disjoint
    assert not (set(val_set.indices) & set(test_set.indices))

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
        train_set, _, _ = mnist_regression_with_test(
            DATA_DIR, n_train=n_train, n_val=10, n_test=10, seed=0
        )
        train_loader, val_loader, test_loader = mnist_loaders_with_test(
            train_set, val_set, test_set, batch_size=BATCH_SIZE, seed=0
        )

        for seed in SEEDS:
            for variant, label, pool_kind, pool in plans:
                key = f"{label}@n{n_train}"
                runs.setdefault(key, [])

                kwargs = {}
                if variant.startswith(("v17_", "v18_")):
                    kwargs["noise_std"] = SIGMA
                    kwargs["input_noise_scale"] = pixel_std
                pool_for_run = pool if pool is not None else pool_inductive

                model = PDFRegressor()
                res = run_pdf_experiment(
                    model, train_loader, val_loader, pool_for_run,
                    variant=variant, epochs=EPOCHS, lr=LR, seed=seed,
                    test_loader=test_loader,
                    **kwargs,
                )
                save_run(OUT / f"{label}_n{n_train}_seed{seed}.json", res)
                runs[key].append(res)
                print(
                    f"  n={n_train} seed={seed} {label:18s} ({pool_kind:12s}): "
                    f"val={res.val_mse_best:.4f}  test@best_val={res.test_mse_at_best_val_epoch:.4f}  "
                    f"(epoch {res.val_mse_best_epoch}, {res.seconds:.1f}s)"
                )

    # --- aggregate ---
    def agg(key, attr):
        vals = [getattr(r, attr) for r in runs[key]]
        m = sum(vals) / len(vals)
        v = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)
        return m, v ** 0.5

    val_mean = {k: agg(k, "val_mse_best")[0] for k in runs}
    val_std = {k: agg(k, "val_mse_best")[1] for k in runs}
    test_mean = {k: agg(k, "test_mse_at_best_val_epoch")[0] for k in runs}
    test_std = {k: agg(k, "test_mse_at_best_val_epoch")[1] for k in runs}

    # The val−test gap diagnoses tailoring artifact for transductive and
    # standard model-selection bias for inductive.
    val_minus_test = {k: val_mean[k] - test_mean[k] for k in runs}

    summary = {
        "label_budgets": LABEL_BUDGETS,
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "sigma": SIGMA,
        "val_mse_best_mean": val_mean,
        "val_mse_best_std": val_std,
        "test_mse_at_best_val_epoch_mean": test_mean,
        "test_mse_at_best_val_epoch_std": test_std,
        "val_minus_test": val_minus_test,
    }
    with (OUT / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # --- plots ---
    import matplotlib.pyplot as plt

    style = {
        "v15":              dict(label="v15 (supervised)",        color="black",     marker="o", linestyle=":"),
        "v16":              dict(label="v16 (sup+reversal-aug)",  color="tab:gray",  marker="s", linestyle="--"),
        "v17_inductive":    dict(label="v17 inductive pool",      color="tab:orange", marker="^"),
        "v17_transductive": dict(label="v17 transductive (val)",  color="tab:red",   marker="v"),
        "v18_inductive":    dict(label="v18 inductive pool",      color="tab:green", marker="D"),
        "v18_transductive": dict(label="v18 transductive (val)",  color="tab:purple", marker="*"),
    }

    # Side-by-side: val curve and test curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, kind, mean, std in (
        (axes[0], "val (in-domain)", val_mean, val_std),
        (axes[1], "test (held-out, unbiased)", test_mean, test_std),
    ):
        for label, sty in style.items():
            ys = [mean[f"{label}@n{n}"] for n in LABEL_BUDGETS]
            es = [std[f"{label}@n{n}"] for n in LABEL_BUDGETS]
            ax.errorbar(LABEL_BUDGETS, ys, yerr=es, capsize=3, **sty)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n_train")
        ax.set_title(kind)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("best MSE (mean ± std over seeds)")
    fig.suptitle("exp1g — held-out test eval")
    fig.tight_layout()
    fig.savefig(OUT / "learning_curve.png", dpi=140)
    plt.close(fig)

    # The val−test gap plot — diagnostic for tailoring artifact
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, sty in style.items():
        ys = [val_minus_test[f"{label}@n{n}"] for n in LABEL_BUDGETS]
        ax.plot(LABEL_BUDGETS, ys, **sty)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("n_train")
    ax.set_ylabel("val − test  (negative = test harder, normal)")
    ax.set_title("exp1g — val/test gap per variant\n"
                 "transductive ≪ inductive at this gap = tailoring-artifact diagnosis")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "val_minus_test_gap.png", dpi=140)
    plt.close(fig)

    # transductive vs inductive on the unbiased test number
    fig, ax = plt.subplots(figsize=(8, 4.5))
    v17_diff = [
        test_mean[f"v17_inductive@n{n}"] - test_mean[f"v17_transductive@n{n}"]
        for n in LABEL_BUDGETS
    ]
    v18_diff = [
        test_mean[f"v18_inductive@n{n}"] - test_mean[f"v18_transductive@n{n}"]
        for n in LABEL_BUDGETS
    ]
    ax.plot(LABEL_BUDGETS, v17_diff, marker="o", color="tab:orange", label="v17")
    ax.plot(LABEL_BUDGETS, v18_diff, marker="s", color="tab:green", label="v18")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("n_train")
    ax.set_ylabel("inductive − transductive (test MSE, unbiased)")
    ax.set_title("exp1g — transductive vs inductive on held-out test set\n"
                 "(fair comparison: model never saw test inputs during training)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "transductive_vs_inductive_test.png", dpi=140)
    plt.close(fig)

    # --- print ---
    print("\n=== summary (mean ± std over seeds) ===")
    print(f"\n{'n':>5s}  ", end="")
    for lab in style:
        print(f"{lab:>22s}", end=" ")
    print()
    for n in LABEL_BUDGETS:
        print(f"{n:>5d}  ", end="")
        for lab in style:
            v = val_mean[f"{lab}@n{n}"]; t = test_mean[f"{lab}@n{n}"]
            print(f"{v:.3f}/{t:.3f}             ", end="")
        print()

    print("\n=== val − test gap (negative = test harder = normal model-selection bias) ===")
    for n in LABEL_BUDGETS:
        print(f"  n={n:>5d}: ", end="")
        for lab in style:
            print(f"{lab}={val_minus_test[f'{lab}@n{n}']:+.3f}  ", end="")
        print()

    print("\n=== transductive vs inductive ON HELD-OUT TEST ===")
    for n in LABEL_BUDGETS:
        v17_d = test_mean[f"v17_inductive@n{n}"] - test_mean[f"v17_transductive@n{n}"]
        v18_d = test_mean[f"v18_inductive@n{n}"] - test_mean[f"v18_transductive@n{n}"]
        print(f"  n={n:>5d}: v17 = {v17_d:+.4f}, v18 = {v18_d:+.4f}")

    print(f"\nWrote results to {OUT}")


if __name__ == "__main__":
    main()
