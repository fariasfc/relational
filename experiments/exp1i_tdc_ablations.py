"""Experiment 1i-tdc — ablations after exp1f-tdc's negative result.

exp1f_tdc found that the MNIST recipe degrades held-out scaffold-test on
TDC Lipophilicity_AstraZeneca. Two follow-ups, both proposed in
SUMMARY.md:

1. **Drop reversal-augmentation.** v16 (sup + 2× reversal-aug) was
   already worse than v15 on Mordred. Test whether the v18 degradation
   is mostly the reversal trick or mostly the consistency loss itself.

2. **True-transductive.** Use scaffold-test inputs as the unlabelled
   pool (the natural setting if you have a known screening set) — but
   evaluate on a separate held-out half of scaffold-test so the reported
   number is unbiased.

Variants compared at each label budget:

* ``v15``                          — supervised baseline.
* ``v18_full_inductive``           — sup + reversal-aug + Π-model on the
                                     unlabelled remainder of scaffold-train.
                                     This is the "broken" recipe from
                                     exp1f_tdc, kept for reference.
* ``v18_pi_only_inductive``        — drop reversal-aug. Same Π-model term
                                     and inductive pool.
* ``v18_pi_only_true_transductive``— drop reversal-aug. Pool = first half
                                     of scaffold-test inputs. Eval on
                                     second half (disjoint).

The held-out eval split is shared by all variants so the comparison is
clean. We report ``test_mse_at_best_val_epoch`` on that 420-sample
held-out slice as the headline number.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from relational.data_tdc import (
    load_tdc_lipophilicity,
    split_test_for_transductive,
    subsample_train,
    tdc_loaders,
)
from relational.models import TabularMLP
from relational.train import run_pdf_experiment, save_run


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "exp1i_tdc"

LABEL_BUDGETS = [100, 250, 500, 1_000, 2_000]
SEEDS = [0, 1, 2]
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
SIGMA = 0.5


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("[exp1i_tdc] Loading TDC Lipophilicity_AstraZeneca + Mordred features…")
    ds = load_tdc_lipophilicity()
    in_dim = ds.train.X.shape[1]

    # Split scaffold-test 420 / 420: first half is the transductive pool
    # (inputs only), second half is the held-out eval set used by all
    # variants — so the number we report is on points NEVER seen during
    # training, even for the true-transductive variant.
    test_pool_split, test_eval = split_test_for_transductive(
        ds.test, n_pool=420, seed=0
    )

    print(
        f"  train={ds.train.X.shape[0]} val={ds.val.X.shape[0]}  "
        f"test_pool={test_pool_split.X.shape[0]} test_eval={test_eval.X.shape[0]}  d={in_dim}"
    )

    # Build a TDCSplit-shaped wrapper for the held-out eval set so the
    # existing tdc_loaders can produce a test_loader from it.
    held_out_test = test_eval

    plans: list[tuple[str, str, str]] = [
        # (variant, label, pool_kind)
        ("v15",                          "v15",                          "none"),
        ("v18_pi_input_var",             "v18_full_inductive",           "inductive"),
        ("v18_pi_only_input_var",        "v18_pi_only_inductive",        "inductive"),
        ("v18_pi_only_input_var",        "v18_pi_only_true_transductive","transductive"),
    ]

    runs: dict[str, list] = {}

    for n_train in LABEL_BUDGETS:
        print(f"\n=== label budget: {n_train} ===")
        for seed in SEEDS:
            labeled, unlabeled = subsample_train(ds.train, n_labeled=n_train, seed=seed)
            train_loader, val_loader, test_loader = tdc_loaders(
                labeled, ds.val, held_out_test, batch_size=BATCH_SIZE, seed=0
            )

            # Inductive pool: the unlabelled remainder of scaffold-train.
            pool_inductive = unlabeled.X if unlabeled.X.shape[0] > 0 else labeled.X
            pool_transductive = test_pool_split.X
            scale = labeled.X.std(dim=0).clamp_min(1e-6)

            for variant, label, pool_kind in plans:
                pool = (
                    pool_inductive   if pool_kind == "inductive"
                    else pool_transductive if pool_kind == "transductive"
                    else pool_inductive  # for v15, unused
                )
                key = f"{label}@n{n_train}"
                runs.setdefault(key, [])

                kwargs = {}
                if variant != "v15":
                    kwargs["noise_std"] = SIGMA
                    kwargs["input_noise_scale"] = scale

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
                    f"  n={n_train} seed={seed} {label:32s} ({pool_kind:12s}): "
                    f"val={res.val_mse_best:.4f}  "
                    f"held_out_test={res.test_mse_at_best_val_epoch:.4f}  "
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
        "label_budgets": LABEL_BUDGETS,
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "sigma": SIGMA,
        "val_mse_best_mean": val_mean,
        "val_mse_best_std": val_std,
        "held_out_test_mean": test_mean,
        "held_out_test_std": test_std,
        "test_pool_size": int(test_pool_split.X.shape[0]),
        "held_out_eval_size": int(test_eval.X.shape[0]),
    }
    with (OUT / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # --- plots ---
    import matplotlib.pyplot as plt

    style = {
        "v15":                            dict(label="v15 (supervised)",        color="black",     marker="o", linestyle=":"),
        "v18_full_inductive":             dict(label="v18 full (sup+rev+Π-model, inductive)",  color="tab:gray",  marker="s", linestyle="--"),
        "v18_pi_only_inductive":          dict(label="v18 Π-only (sup+Π-model, inductive)",     color="tab:green", marker="D"),
        "v18_pi_only_true_transductive":  dict(label="v18 Π-only, true transductive (test inputs)", color="tab:purple", marker="*"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
    for ax, kind, mean, std in (
        (axes[0], "val (in-domain)", val_mean, val_std),
        (axes[1], "held-out test (disjoint scaffold slice)", test_mean, test_std),
    ):
        for label, sty in style.items():
            ys = [mean[f"{label}@n{n}"] for n in LABEL_BUDGETS]
            es = [std[f"{label}@n{n}"] for n in LABEL_BUDGETS]
            ax.errorbar(LABEL_BUDGETS, ys, yerr=es, capsize=3, **sty)
        ax.set_xscale("log")
        ax.set_xlabel("n_labeled")
        ax.set_title(kind)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("best MSE (mean ± std over seeds)")
    fig.suptitle(
        f"exp1i_tdc — drop-reversal + true-transductive ablations on {ds.name}"
    )
    fig.tight_layout()
    fig.savefig(OUT / "ablations.png", dpi=140)
    plt.close(fig)

    # Δ vs v15 on the held-out test
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
    ax.set_ylabel("Δ held-out test MSE vs v15  (positive = better than supervised)")
    ax.set_title("exp1i_tdc — does any SSL variant beat supervised on scaffold-test?")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "delta_vs_v15_test.png", dpi=140)
    plt.close(fig)

    # --- print summary ---
    print(f"\n=== summary table (val / held-out test, mean ± std over {len(SEEDS)} seeds) ===")
    print(f"{'n':>5s}  ", end="")
    for lab in style:
        print(f"{lab[:30]:>32s}", end=" ")
    print()
    for n in LABEL_BUDGETS:
        print(f"{n:>5d}  ", end="")
        for lab in style:
            v = val_mean[f"{lab}@n{n}"]; t = test_mean[f"{lab}@n{n}"]
            print(f"{v:.3f}/{t:.3f}                 ", end="")
        print()

    print("\n=== Δ held-out test vs v15 (positive = SSL beats supervised) ===")
    for n in LABEL_BUDGETS:
        for lab in style:
            if lab == "v15":
                continue
            d = test_mean[f"v15@n{n}"] - test_mean[f"{lab}@n{n}"]
            print(f"  n={n:>5d} {lab:34s}: {d:+.4f}")
        print()

    print(f"\nWrote results to {OUT}")


if __name__ == "__main__":
    main()
