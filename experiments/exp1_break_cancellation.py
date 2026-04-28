"""Experiment 1 — break the cancellation symmetry with noisy transductive forwards.

The 2024 PDF's loss has ``(p + f(q)) - (p_rev + f(q))`` — the two ``f(q)``
terms cancel and the unlabeled data contributes zero gradient (proven in
exp0). Here we replace those evaluations with ``f(q + ε_a)`` and
``f(q + ε_b)`` where ε_a ≠ ε_b are independent samples per step, drawn
either in input space or at the bottleneck embedding.

Two questions:

1. **Gradient-flow probe.** Does ∇θ now differ between "real noisy
   forwards" and "transductive forwards := 0"? If yes, symmetry is broken
   and the unlabeled data is contributing.

2. **Does it actually help?** Compared to v16 (which is supervised + a
   reversal-augmentation regulariser, with no real transductive
   contribution), do v17_input or v17_emb deliver lower val MSE?

   Mathematically the symmetry-broken loss expands to
       reversal_aug(p, p_rev, y, y_rev) + E[(z_a − z_b)²] + cross_term
   where the second term is exactly Π-model-style consistency on q with
   coefficient 1, and the cross term has zero mean under symmetric noise.
   So we expect v17_* ≈ v16 + Π-model consistency.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from relational.data import mnist_loaders, mnist_regression
from relational.losses import noisy_pdf_loss, original_pdf_loss
from relational.models import PDFRegressor
from relational.plotting import save_curves
from relational.train import run_pdf_experiment, save_run, seed_everything


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT = ROOT / "results" / "exp1"

# σ values swept for both input- and embedding-space noise variants.
NOISE_STDS = [0.1, 0.5, 1.0, 2.0]
# Probe σ used for the up-front gradient-flow check.
PROBE_NOISE_STD = 0.5
# Floor: float-precision noise (exp0) was ~1e-6 ratio, so 1e-2 is 4 orders of
# magnitude above that — clearly genuine gradient flow.
FLOW_THRESHOLD = 1e-2


def gradient_flow_probe(pool_x: torch.Tensor) -> dict:
    """At matched init/batch, compare ∇θ for noisy v17 vs the same loss with
    transductive forwards zeroed. Big diff = unlabeled gradient is flowing."""
    rows = []
    for noise_space in ("input", "embedding"):
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

            # Use the SAME noise generator state for both forwards so that
            # any non-zero ∇θ diff is genuinely from the transductive
            # contribution, not from drawing different ε's.
            g_noise_a = torch.Generator().manual_seed(seed + 200)
            g_noise_b = torch.Generator().manual_seed(seed + 200)

            loss_a, _ = noisy_pdf_loss(
                m_a, x, y, transductive_x=t,
                noise_std=PROBE_NOISE_STD, noise_space=noise_space,
                generator=g_noise_a,
            )
            loss_a.backward()
            g_a = torch.cat([p.grad.flatten() for p in m_a.parameters()])

            loss_b, _ = noisy_pdf_loss(
                m_b, x, y, transductive_x=t,
                noise_std=PROBE_NOISE_STD, noise_space=noise_space,
                zero_transductive=True,
                generator=g_noise_b,
            )
            loss_b.backward()
            g_b = torch.cat([p.grad.flatten() for p in m_b.parameters()])

            cos = torch.nn.functional.cosine_similarity(
                g_a.unsqueeze(0), g_b.unsqueeze(0)
            ).item()
            rows.append({
                "noise_space": noise_space,
                "seed": seed,
                "loss_diff": abs(loss_a.item() - loss_b.item()),
                "grad_max_abs_diff": (g_a - g_b).abs().max().item(),
                "grad_l2_diff": (g_a - g_b).norm().item(),
                "grad_norm_a": g_a.norm().item(),
                "grad_norm_b": g_b.norm().item(),
                "grad_cosine_similarity": cos,
            })

    # Flow ratio: how big is the noisy-vs-zeroed diff relative to the
    # gradient norm? > 0.05 means the unlabeled data is meaningfully
    # contributing.
    summary = {"rows": rows}
    for ns in ("input", "embedding"):
        sub = [r for r in rows if r["noise_space"] == ns]
        ratios = [r["grad_l2_diff"] / r["grad_norm_a"] for r in sub]
        summary[ns] = {
            "mean_grad_diff_ratio": sum(ratios) / len(ratios),
            "min_grad_diff_ratio": min(ratios),
            "mean_cosine": sum(r["grad_cosine_similarity"] for r in sub) / len(sub),
        }
    # Pass = symmetry visibly broken (ratio well above the ~1e-6 float
    # noise floor measured in exp0).
    summary["threshold"] = FLOW_THRESHOLD
    summary["passed"] = all(
        r["grad_l2_diff"] / r["grad_norm_a"] > FLOW_THRESHOLD for r in rows
    )
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    train, val = mnist_regression(DATA_DIR, n_train=5_000, n_val=1_000, seed=0)
    train_loader, val_loader = mnist_loaders(train, val, batch_size=128, seed=0)

    pool, _ = mnist_regression(DATA_DIR, n_train=5_000, n_val=10, seed=999)
    pool_x = torch.stack([x for x, _ in pool])

    # --- (1) Gradient-flow probe ---
    flow = gradient_flow_probe(pool_x)
    with (OUT / "gradient_flow.json").open("w") as f:
        json.dump(flow, f, indent=2)
    print("[exp1] gradient-flow probe (noisy vs zeroed transductive):")
    for ns in ("input", "embedding"):
        s = flow[ns]
        print(
            f"  {ns:9s}: mean ‖Δgrad‖/‖grad‖ = {s['mean_grad_diff_ratio']:.3f}, "
            f"min = {s['min_grad_diff_ratio']:.3f}, "
            f"cosine = {s['mean_cosine']:.4f}"
        )
    print(f"  PASS (all ratios > 0.05): {flow['passed']}")

    # --- (2) Training comparison + σ sweep ---
    seeds = [0, 1, 2]
    epochs = 30
    lr = 1e-3

    # Anchors (no noise hyperparameter).
    anchor_variants = ["v15", "v16"]
    # Sweeps for each noise space.
    sweep_keys: list[tuple[str, float]] = []
    for ns in ("v17_input", "v17_emb"):
        for sigma in NOISE_STDS:
            sweep_keys.append((ns, sigma))

    runs: dict[str, list] = {}
    label_for: dict[str, str] = {}

    for seed in seeds:
        for variant in anchor_variants:
            key = variant
            label_for[key] = variant
            runs.setdefault(key, [])
            print(f"[exp1] seed={seed} variant={variant}")
            model = PDFRegressor()
            res = run_pdf_experiment(
                model, train_loader, val_loader, pool_x,
                variant=variant, epochs=epochs, lr=lr, seed=seed,
            )
            save_run(OUT / f"{variant}_seed{seed}.json", res)
            runs[key].append(res)
            print(f"  final val MSE = {res.val_mse[-1]:.4f}   ({res.seconds:.1f}s)")

        for variant, sigma in sweep_keys:
            key = f"{variant}_s{sigma}"
            label_for[key] = f"{variant} σ={sigma}"
            runs.setdefault(key, [])
            print(f"[exp1] seed={seed} variant={variant} σ={sigma}")
            model = PDFRegressor()
            res = run_pdf_experiment(
                model, train_loader, val_loader, pool_x,
                variant=variant, epochs=epochs, lr=lr, seed=seed,
                noise_std=sigma,
            )
            save_run(OUT / f"{key}_seed{seed}.json", res)
            runs[key].append(res)
            print(f"  final val MSE = {res.val_mse[-1]:.4f}   ({res.seconds:.1f}s)")

    def stats(name):
        vals = [r.val_mse[-1] for r in runs[name]]
        m = sum(vals) / len(vals)
        v = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)
        return m, v ** 0.5

    keys = list(runs.keys())
    means = {k: stats(k)[0] for k in keys}
    stds = {k: stats(k)[1] for k in keys}

    summary = {
        "probe_noise_std": PROBE_NOISE_STD,
        "noise_std_sweep": NOISE_STDS,
        "epochs": epochs,
        "seeds": seeds,
        "val_mse_mean": means,
        "val_mse_std": stds,
        "delta_vs_v16": {k: means["v16"] - means[k] for k in keys},
        "gradient_flow": {
            "input": flow["input"],
            "embedding": flow["embedding"],
            "passed": flow["passed"],
            "threshold": flow["threshold"],
        },
        "best": {
            "v17_input": min(
                (k for k in keys if k.startswith("v17_input_s")),
                key=lambda k: means[k],
            ),
            "v17_emb": min(
                (k for k in keys if k.startswith("v17_emb_s")),
                key=lambda k: means[k],
            ),
        },
    }
    with (OUT / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # --- Plots ---
    def mean_curve(key):
        curves = [r.val_mse for r in runs[key]]
        return [
            sum(c[i] for c in curves) / len(curves) for i in range(len(curves[0]))
        ]

    # Curves: v15, v16, plus best σ for each noisy variant.
    best_in = summary["best"]["v17_input"]
    best_em = summary["best"]["v17_emb"]
    save_curves(
        OUT / "curves.png",
        {
            "v15 (supervised)": mean_curve("v15"),
            "v16 (PDF, cancels)": mean_curve("v16"),
            f"v17_input — best σ ({label_for[best_in]})": mean_curve(best_in),
            f"v17_emb — best σ ({label_for[best_em]})": mean_curve(best_em),
        },
        ylabel="val MSE",
        title=f"exp1 — symmetry-broken transductive (mean of {len(seeds)} seeds)",
    )

    # σ-sweep summary plot: val MSE vs σ for each noise space.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for ns, marker in (("v17_input", "o"), ("v17_emb", "s")):
        xs = NOISE_STDS
        ys = [means[f"{ns}_s{s}"] for s in xs]
        es = [stds[f"{ns}_s{s}"] for s in xs]
        ax.errorbar(xs, ys, yerr=es, marker=marker, capsize=3, label=ns)
    ax.axhline(means["v16"], color="gray", linestyle="--", label="v16 (cancels)")
    ax.axhline(means["v15"], color="black", linestyle=":", label="v15 (supervised)")
    ax.set_xscale("log")
    ax.set_xlabel("noise σ")
    ax.set_ylabel("val MSE (mean ± std over seeds)")
    ax.set_title("exp1 — σ sweep for symmetry-broken transductive loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "sigma_sweep.png", dpi=140)
    plt.close(fig)

    print()
    print("Final val MSE (mean ± std over seeds):")
    for k in keys:
        print(f"  {label_for.get(k, k):24s}: {means[k]:.4f} ± {stds[k]:.4f}")
    print()
    print("Δ vs v16 (positive = better than the cancelling baseline):")
    for k in keys:
        print(f"  {label_for.get(k, k):24s}: {summary['delta_vs_v16'][k]:+.4f}")
    print(
        f"\nBest input-noise σ:     {summary['best']['v17_input']} "
        f"(val MSE {means[summary['best']['v17_input']]:.4f})"
    )
    print(
        f"Best embedding-noise σ: {summary['best']['v17_emb']} "
        f"(val MSE {means[summary['best']['v17_emb']]:.4f})"
    )
    print(f"\nWrote results to {OUT}")


if __name__ == "__main__":
    main()
