"""Experiment 0 — empirical falsification of the original PDF's transductive loss.

Reproduces ``version_15`` (regression baseline), ``version_16`` (PDF
"transductive" loss), and ``version_16_zeroed`` (same loss with the
transductive-sample forward replaced by zeros).

The Cancellation Theorem (docx Appendix A, Cor. 2) predicts v16 and
v16_zeroed compute *the same gradient* at any matched parameter state.
Two checks here:

1. **Gradient-equivalence (definitive).** With identical model init and
   identical data batches, the gradients ∇θ L_v16 and ∇θ L_v16_zeroed
   must agree to float32 precision (≤ 1e-5 abs). This is an exact
   algebraic prediction.

2. **Trajectory-similarity (illustrative).** Over many SGD steps,
   v16 and v16_zeroed loss curves track each other within seed-to-seed
   noise — but they are NOT bit-exact, because float32 rounding from the
   extra ``f(q)`` add+sub accumulates over ~1k optimiser steps. We report
   the discrepancy alongside the v15-vs-v16 gap so the magnitudes are
   directly comparable.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from relational.data import mnist_loaders, mnist_regression
from relational.losses import original_pdf_loss
from relational.models import PDFRegressor
from relational.plotting import save_curves
from relational.train import run_pdf_experiment, save_run, seed_everything


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT = ROOT / "results" / "exp0"


def gradient_equivalence_check(pool_x: torch.Tensor) -> dict:
    """Definitive cancellation test: compare ∇θ L_v16 and ∇θ L_v16_zeroed.

    With identical init and identical batch, these must match to float32
    precision (~1e-7 max abs diff). We report stats across several
    init/batch seeds.
    """
    rows = []
    for seed in range(5):
        seed_everything(seed)
        m_a = PDFRegressor()
        seed_everything(seed)
        m_b = PDFRegressor()  # same init

        g = torch.Generator().manual_seed(seed + 100)
        x = torch.randn(32, 1, 28, 28, generator=g)
        y = torch.rand(32, generator=g) * 9.0
        idx = torch.randint(0, pool_x.shape[0], (32,), generator=g)
        t = pool_x[idx]

        loss_a, _ = original_pdf_loss(m_a, x, y, transductive_x=t)
        loss_a.backward()
        g_a = torch.cat([p.grad.flatten() for p in m_a.parameters()])

        loss_b, _ = original_pdf_loss(
            m_b, x, y, transductive_x=t, zero_transductive=True
        )
        loss_b.backward()
        g_b = torch.cat([p.grad.flatten() for p in m_b.parameters()])

        cos = torch.nn.functional.cosine_similarity(
            g_a.unsqueeze(0), g_b.unsqueeze(0)
        ).item()
        rows.append(
            {
                "seed": seed,
                "loss_diff": abs(loss_a.item() - loss_b.item()),
                "grad_max_abs_diff": (g_a - g_b).abs().max().item(),
                "grad_norm_a": g_a.norm().item(),
                "grad_norm_b": g_b.norm().item(),
                "grad_cosine_similarity": cos,
            }
        )

    summary = {
        "max_loss_diff": max(r["loss_diff"] for r in rows),
        "max_grad_diff": max(r["grad_max_abs_diff"] for r in rows),
        "min_cosine": min(r["grad_cosine_similarity"] for r in rows),
        "tolerance_grad_abs": 1e-5,
        "passed": (
            max(r["loss_diff"] for r in rows) < 1e-5
            and max(r["grad_max_abs_diff"] for r in rows) < 1e-5
        ),
        "rows": rows,
    }
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # --- data ---
    train, val = mnist_regression(DATA_DIR, n_train=5_000, n_val=1_000, seed=0)
    train_loader, val_loader = mnist_loaders(train, val, batch_size=128, seed=0)

    # Transductive pool — separate slice of MNIST that the PDF code samples
    # from each training step. Anything will do; the cancellation result is
    # invariant to this content.
    pool, _ = mnist_regression(DATA_DIR, n_train=5_000, n_val=10, seed=999)
    pool_x = torch.stack([x for x, _ in pool])  # (N, 1, 28, 28)

    # --- (1) Definitive cancellation test: gradient equivalence ---
    grad_eq = gradient_equivalence_check(pool_x)
    with (OUT / "gradient_equivalence.json").open("w") as f:
        json.dump(grad_eq, f, indent=2)
    print(
        f"[exp0] gradient equivalence: max grad |Δ| = "
        f"{grad_eq['max_grad_diff']:.2e}, min cosine = "
        f"{grad_eq['min_cosine']:.10f}, PASS={grad_eq['passed']}"
    )

    seeds = [0, 1, 2]
    epochs = 30
    lr = 1e-3

    runs: dict[str, list] = {"v15": [], "v16": [], "v16_zeroed": []}

    for seed in seeds:
        for variant in ("v15", "v16", "v16_zeroed"):
            print(f"[exp0] seed={seed} variant={variant}")
            model = PDFRegressor()
            res = run_pdf_experiment(
                model,
                train_loader,
                val_loader,
                pool_x,
                variant=variant,
                epochs=epochs,
                lr=lr,
                seed=seed,
                # Step-level losses are how we cross-check v16 vs v16_zeroed.
                log_steps=(variant in ("v16", "v16_zeroed")),
            )
            save_run(OUT / f"{variant}_seed{seed}.json", res)
            runs[variant].append(res)
            print(
                f"  final val MSE = {res.val_mse[-1]:.4f}   "
                f"({res.seconds:.1f}s)"
            )

    # --- summary metrics ---
    def final_mean(variant):
        return sum(r.val_mse[-1] for r in runs[variant]) / len(runs[variant])

    summary = {
        "epochs": epochs,
        "seeds": seeds,
        "val_mse_mean": {v: final_mean(v) for v in runs},
        "v15_vs_v16_relative_change": (
            (final_mean("v16") - final_mean("v15")) / final_mean("v15")
        ),
    }

    # Cancellation check: v16 vs v16_zeroed step losses, same seed.
    diffs = []
    per_seed = {}
    for seed in seeds:
        a = next(r for r in runs["v16"] if r.seed == seed).step_losses
        b = next(r for r in runs["v16_zeroed"] if r.seed == seed).step_losses
        # Pairwise relative diff with a small epsilon.
        rel = [
            abs(x - y) / (abs(x) + 1e-8)
            for x, y in zip(a, b)
        ]
        max_rel = max(rel) if rel else 0.0
        max_abs = max((abs(x - y) for x, y in zip(a, b)), default=0.0)
        per_seed[str(seed)] = {
            "n_steps": len(a),
            "max_relative_diff": max_rel,
            "max_absolute_diff": max_abs,
            "mean_absolute_diff": (
                sum(abs(x - y) for x, y in zip(a, b)) / max(1, len(a))
            ),
        }
        diffs.append(max_rel)

    # Trajectory similarity: NOT a bit-exactness test (FP rounding accumulates
    # over ~1k optimiser steps). The right framing is "v16 and v16_zeroed
    # diverge by <<< the v15→v16 gap, AND end up indistinguishable within
    # seed-to-seed noise."
    v15_curve = mean_curve_now = sum(r.val_mse[-1] for r in runs["v15"]) / len(runs["v15"])
    v16_final = sum(r.val_mse[-1] for r in runs["v16"]) / len(runs["v16"])
    v16z_final = sum(r.val_mse[-1] for r in runs["v16_zeroed"]) / len(runs["v16_zeroed"])
    # std across seeds for v16 family combined
    pooled_v16 = [r.val_mse[-1] for r in runs["v16"]] + [
        r.val_mse[-1] for r in runs["v16_zeroed"]
    ]
    n = len(pooled_v16)
    pooled_mean = sum(pooled_v16) / n
    pooled_std = (sum((v - pooled_mean) ** 2 for v in pooled_v16) / max(1, n - 1)) ** 0.5

    cancellation = {
        "trajectory_max_relative_diff_overall": max(diffs) if diffs else 0.0,
        "trajectory_per_seed": per_seed,
        "v15_v16_gap": v15_curve - v16_final,
        "v16_v16zeroed_gap": v16_final - v16z_final,
        "pooled_v16_seed_std": pooled_std,
        # "Passed" here means: v16 and v16_zeroed agree to within seed noise,
        # AND the v15→v16 gap is much larger than the v16→v16_zeroed gap.
        "passed_within_seed_noise": (
            abs(v16_final - v16z_final) <= 2 * pooled_std
            and (v15_curve - v16_final) > 5 * abs(v16_final - v16z_final)
        ),
    }
    with (OUT / "cancellation_check.json").open("w") as f:
        json.dump(cancellation, f, indent=2)
    summary["gradient_equivalence_passed"] = grad_eq["passed"]
    summary["trajectory_equivalence_within_seed_noise"] = cancellation[
        "passed_within_seed_noise"
    ]
    with (OUT / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # --- plots ---
    # Average val-MSE curves across seeds for each variant.
    def mean_curve(variant):
        curves = [r.val_mse for r in runs[variant]]
        return [sum(c[i] for c in curves) / len(curves) for i in range(len(curves[0]))]

    save_curves(
        OUT / "curves.png",
        {
            "v15 (supervised)": mean_curve("v15"),
            "v16 (PDF transductive)": mean_curve("v16"),
            "v16_zeroed (f(q) := 0)": mean_curve("v16_zeroed"),
        },
        ylabel="val MSE",
        title="exp0 — MNIST-as-regression val MSE (mean of 3 seeds)",
    )

    print()
    print("Final val MSE (mean over seeds):")
    for v in ("v15", "v16", "v16_zeroed"):
        print(f"  {v}: {final_mean(v):.4f}")
    print()
    print(
        f"Definitive (gradient) cancellation:  max grad |Δ| = "
        f"{grad_eq['max_grad_diff']:.2e}, min cosine = "
        f"{grad_eq['min_cosine']:.10f}   PASS={grad_eq['passed']}"
    )
    print(
        f"Trajectory test (illustrative):     v15→v16 gap = "
        f"{cancellation['v15_v16_gap']:+.4f}, "
        f"v16→v16_zeroed gap = {cancellation['v16_v16zeroed_gap']:+.4f}, "
        f"pooled-seed σ = {cancellation['pooled_v16_seed_std']:.4f}   "
        f"PASS={cancellation['passed_within_seed_noise']}"
    )
    print(f"Wrote results to {OUT}")


if __name__ == "__main__":
    main()
