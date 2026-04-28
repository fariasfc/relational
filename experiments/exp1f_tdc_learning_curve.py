"""Experiment 1f-TDC — learning curve and inductive-vs-transductive on a real
molecular regression task.

Mirror of ``exp1f_learning_curve.py`` but with TDC Lipophilicity_AstraZeneca
and Mordred descriptors. The same loss machinery and runner are reused
unchanged — only the data layer and model differ. Reports both val
(in-domain, biased by selection) and held-out test (unbiased) MSE.

Variants per (n_labeled, seed):
* ``v15``               — supervised baseline.
* ``v16``               — sup + 2× reversal-augmentation.
* ``v18_inductive``     — v16 + plain Π-model with variance-scaled noise on the
                          unlabelled remainder of the scaffold-train set.
* ``v18_transductive``  — same loss, pool = val-set inputs (no labels).
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
OUT = ROOT / "results" / "exp1f_tdc"

LABEL_BUDGETS = [50, 100, 250, 500, 1_000, 2_000]
SEEDS = [0, 1, 2]
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
SIGMA = 0.5


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("[exp1f_tdc] Loading TDC Lipophilicity_AstraZeneca + Mordred features…")
    ds = load_tdc_lipophilicity()
    in_dim = ds.train.X.shape[1]
    print(
        f"  train={ds.train.X.shape[0]} val={ds.val.X.shape[0]} test={ds.test.X.shape[0]}  "
        f"feature dim={in_dim}"
    )

    # Pool tensors stacked as 4-D-friendly shape (n, 1, 1, d) is unnecessary
    # for tabular — the runner uses ``view(n, -1)`` which is a no-op for 2-D.
    pool_transductive = ds.val.X.clone()    # val-set inputs as transductive pool

    plans: list[tuple[str, str, str, torch.Tensor]] = [
        # (variant, label, pool_kind, pool tensor)
        ("v15",              "v15",              "none",         ds.train.X),
        ("v16",              "v16",              "inductive",    ds.train.X),
        ("v18_pi_input_var", "v18_inductive",    "inductive",    ds.train.X),  # placeholder; rebuilt per-budget
        ("v18_pi_input_var", "v18_transductive", "transductive", pool_transductive),
    ]

    runs: dict[str, list] = {}

    for n_train in LABEL_BUDGETS:
        print(f"\n=== label budget: {n_train} ===")
        for seed in SEEDS:
            labeled, unlabeled = subsample_train(ds.train, n_labeled=n_train, seed=seed)
            train_loader, val_loader, test_loader = tdc_loaders(
                labeled, ds.val, ds.test, batch_size=BATCH_SIZE, seed=0
            )

            # Inductive pool = the *unlabelled remainder* of the scaffold train set.
            # Falls back to the labelled set itself if n_train ≥ |train| (no
            # unlabelled rows left). This is the appropriate behaviour: at full
            # supervision the consistency loss has no novel inputs to learn from.
            pool_inductive = unlabeled.X if unlabeled.X.shape[0] > 0 else labeled.X

            for variant, label, pool_kind, _ in plans:
                pool = (
                    pool_inductive if pool_kind == "inductive"
                    else pool_transductive if pool_kind == "transductive"
                    else pool_inductive  # for v15, anything works (unused)
                )
                key = f"{label}@n{n_train}"
                runs.setdefault(key, [])

                kwargs = {}
                if variant.startswith(("v17_", "v18_")):
                    kwargs["noise_std"] = SIGMA
                    # Per-feature std on the labelled set itself — standardised
                    # data, so this is ≈ 1 across most features but captures any
                    # feature that happens to be near-constant in this fold.
                    kwargs["input_noise_scale"] = labeled.X.std(dim=0).clamp_min(1e-6)

                model = TabularMLP(in_dim=in_dim)
                res = run_pdf_experiment(
                    model, train_loader, val_loader, pool,
                    variant=variant, epochs=EPOCHS, lr=LR, seed=seed,
                    test_loader=test_loader,
                    **kwargs,
                )
                save_run(OUT / f"{label}_n{n_train}_seed{seed}.json", res)
                runs[key].append(res)
                print(
                    f"  n={n_train} seed={seed} {label:18s} ({pool_kind:12s}): "
                    f"val={res.val_mse_best:.4f} test@best_val={res.test_mse_at_best_val_epoch:.4f}  "
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

    summary = {
        "task": ds.name,
        "feature_dim": in_dim,
        "label_budgets": LABEL_BUDGETS,
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "sigma": SIGMA,
        "val_mse_best_mean": val_mean,
        "val_mse_best_std": val_std,
        "test_mse_at_best_val_epoch_mean": test_mean,
        "test_mse_at_best_val_epoch_std": test_std,
    }
    with (OUT / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # --- plots ---
    import matplotlib.pyplot as plt

    style = {
        "v15":              dict(label="v15 (supervised)",          color="black",     marker="o", linestyle=":"),
        "v16":              dict(label="v16 (sup+reversal-aug)",    color="tab:gray",  marker="s", linestyle="--"),
        "v18_inductive":    dict(label="v18 inductive pool",        color="tab:green", marker="D"),
        "v18_transductive": dict(label="v18 transductive (val)",    color="tab:purple", marker="*"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, kind, mean, std in (
        (axes[0], "val (in-domain)", val_mean, val_std),
        (axes[1], "test (held-out scaffold split)", test_mean, test_std),
    ):
        for label, sty in style.items():
            ys = [mean[f"{label}@n{n}"] for n in LABEL_BUDGETS]
            es = [std[f"{label}@n{n}"] for n in LABEL_BUDGETS]
            ax.errorbar(LABEL_BUDGETS, ys, yerr=es, capsize=3, **sty)
        ax.set_xscale("log")
        ax.set_xlabel("n_labeled")
        ax.set_title(kind)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("best MSE (mean ± std over seeds)")
    fig.suptitle(f"exp1f_tdc — {ds.name} learning curve, Mordred features")
    fig.tight_layout()
    fig.savefig(OUT / "learning_curve.png", dpi=140)
    plt.close(fig)

    # Δ vs supervised on test
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, sty in style.items():
        if label == "v15":
            continue
        deltas = [
            test_mean[f"v15@n{n}"] - test_mean[f"{label}@n{n}"]
            for n in LABEL_BUDGETS
        ]
        ax.plot(LABEL_BUDGETS, deltas, **sty)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("n_labeled")
    ax.set_ylabel("Δ test MSE vs v15  (positive = better than supervised)")
    ax.set_title(f"exp1f_tdc — absolute SSL benefit on held-out scaffold test")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "delta_vs_supervised_test.png", dpi=140)
    plt.close(fig)

    # transductive vs inductive on test
    fig, ax = plt.subplots(figsize=(8, 4.5))
    diffs = [
        test_mean[f"v18_inductive@n{n}"] - test_mean[f"v18_transductive@n{n}"]
        for n in LABEL_BUDGETS
    ]
    ax.plot(LABEL_BUDGETS, diffs, marker="o", color="tab:green")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("n_labeled")
    ax.set_ylabel("inductive − transductive  (positive = transductive wins)")
    ax.set_title("exp1f_tdc — transductive vs inductive (held-out scaffold test)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "transductive_vs_inductive_test.png", dpi=140)
    plt.close(fig)

    # --- print summary ---
    print(f"\n=== summary table (val / test MSE, mean ± std over {len(SEEDS)} seeds) ===")
    print(f"{'n':>5s}  ", end="")
    for lab in style:
        print(f"{lab:>22s}", end=" ")
    print()
    for n in LABEL_BUDGETS:
        print(f"{n:>5d}  ", end="")
        for lab in style:
            v = val_mean[f"{lab}@n{n}"]; t = test_mean[f"{lab}@n{n}"]
            print(f"{v:.3f}/{t:.3f}             ", end="")
        print()

    print(f"\nWrote results to {OUT}")


if __name__ == "__main__":
    main()
