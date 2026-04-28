"""Experiment 1b — does breaking the cancellation symmetry help at low label budgets?

Two questions left open by exp1:

1. **Low labels.** At 5 000 MNIST labels, the symmetry-broken loss matched
   v16 within seed noise. Consistency-style regularisers earn their keep
   at low label budgets — does v17 visibly beat v16 at n_train ∈
   {100, 250, 1 000}?

2. **Variance-scaled noise.** Isotropic noise destroys the structure of
   the input (a pixel that is always black gets the same perturbation as
   a pixel in the digit). We replace ``ε ~ N(0, σ²·I)`` with
   ``ε = σ · per_dim_std · z``, ``z ~ N(0, I)``. ``σ`` is now a fraction
   of typical per-dimension variation, not an absolute magnitude. We
   want the *smallest* σ that still breaks the symmetry — preserving the
   semantics of the data point while still letting the unlabeled
   gradient flow.

   For input space, ``per_dim_std`` is the per-pixel std of the
   transductive pool (precomputed once). Many pixels (corners, borders)
   have near-zero variance and will get near-zero noise.

   For embedding space, the loss computes the per-coordinate std on-the-fly
   from the current batch's bottleneck.

The experiment compares variants at three label budgets, using 5 seeds.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from relational.data import mnist_loaders, mnist_regression
from relational.losses import noisy_pdf_loss
from relational.models import PDFRegressor
from relational.plotting import save_curves
from relational.train import run_pdf_experiment, save_run, seed_everything


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT = ROOT / "results" / "exp1b"

LABEL_BUDGETS = [100, 250, 1_000]
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3

# σ values explored for each noisy variant.
SIGMAS_ISO = [0.5]                  # exp1 anchor only
SIGMAS_VAR = [0.05, 0.1, 0.5]       # variance-scaled — small σ is the new option


def gradient_flow_probe(pool_x: torch.Tensor, pixel_std: torch.Tensor) -> dict:
    """Quantify how much each (variant, σ) actually breaks the symmetry."""
    rows = []
    for variant, sigma, scale in (
        [("v17_input", s, None) for s in SIGMAS_ISO]
        + [("v17_input_var", s, pixel_std) for s in SIGMAS_VAR]
        + [("v17_emb", s, None) for s in SIGMAS_ISO]
        + [("v17_emb_var", s, None) for s in SIGMAS_VAR]
    ):
        for seed in range(3):
            seed_everything(seed)
            m_a = PDFRegressor()
            seed_everything(seed)
            m_b = PDFRegressor()

            g = torch.Generator().manual_seed(seed + 100)
            x = torch.randn(32, 1, 28, 28, generator=g)
            y = torch.rand(32, generator=g) * 9.0
            idx = torch.randint(0, pool_x.shape[0], (32,), generator=g)
            t = pool_x[idx]

            g_a = torch.Generator().manual_seed(seed + 200)
            g_b = torch.Generator().manual_seed(seed + 200)

            space = "input" if "input" in variant else "embedding"

            la, _ = noisy_pdf_loss(
                m_a, x, y, transductive_x=t,
                noise_std=sigma, noise_space=space, noise_scale=scale,
                generator=g_a,
            )
            la.backward()
            ga = torch.cat([p.grad.flatten() for p in m_a.parameters()])

            lb, _ = noisy_pdf_loss(
                m_b, x, y, transductive_x=t,
                noise_std=sigma, noise_space=space, noise_scale=scale,
                zero_transductive=True, generator=g_b,
            )
            lb.backward()
            gb = torch.cat([p.grad.flatten() for p in m_b.parameters()])

            ratio = (ga - gb).norm().item() / ga.norm().item()
            rows.append(
                {"variant": variant, "sigma": sigma, "seed": seed, "flow_ratio": ratio}
            )

    summary: dict = {"rows": rows}
    by_key: dict[tuple[str, float], list[float]] = {}
    for r in rows:
        by_key.setdefault((r["variant"], r["sigma"]), []).append(r["flow_ratio"])
    summary["mean_flow_ratio"] = {
        f"{k[0]}@σ={k[1]}": sum(v) / len(v) for k, v in by_key.items()
    }
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Load a transductive pool — used both for symmetry-breaking and for
    # computing the per-pixel std that scales the variance-scaled noise.
    pool, _ = mnist_regression(DATA_DIR, n_train=5_000, n_val=10, seed=999)
    pool_x = torch.stack([x for x, _ in pool])
    pool_flat = pool_x.view(pool_x.shape[0], -1)
    # Per-pixel std across the unlabeled pool. After the standard MNIST
    # normalisation, background pixels are essentially constant.
    pixel_std = pool_flat.std(dim=0)
    n_zero = (pixel_std < 0.05).sum().item()
    print(
        f"[exp1b] pixel std stats: mean={pixel_std.mean():.3f}, "
        f"max={pixel_std.max():.3f}, "
        f"min={pixel_std.min():.3f}, "
        f"frac<0.05={n_zero / pixel_std.numel():.2%}"
    )

    # --- (1) Gradient-flow probe across all (variant, σ) we plan to train ---
    flow = gradient_flow_probe(pool_x, pixel_std)
    with (OUT / "gradient_flow.json").open("w") as f:
        json.dump(flow, f, indent=2)
    print("[exp1b] mean flow ratio across 3 seeds:")
    for k, v in flow["mean_flow_ratio"].items():
        print(f"  {k:30s}: {v:.4f}")

    # --- (2) Training sweep: label budget × variant ---
    val, _ = mnist_regression(DATA_DIR, n_train=10, n_val=1_000, seed=0)  # placeholder
    # We need a fixed val set across label budgets. Use a 1k subset.
    _, val_set = mnist_regression(DATA_DIR, n_train=10, n_val=1_000, seed=0)

    runs: dict[str, list] = {}
    label_for: dict[str, str] = {}

    def add_run(key, label, res):
        runs.setdefault(key, []).append(res)
        label_for[key] = label

    for n_train in LABEL_BUDGETS:
        print(f"\n=== label budget: {n_train} ===")
        train_set, _ = mnist_regression(
            DATA_DIR, n_train=n_train, n_val=10, seed=0
        )
        train_loader, val_loader = mnist_loaders(
            train_set, val_set, batch_size=BATCH_SIZE, seed=0
        )

        for seed in SEEDS:
            for variant in ("v15", "v16"):
                key = f"{variant}@n{n_train}"
                label = f"{variant} (n={n_train})"
                model = PDFRegressor()
                res = run_pdf_experiment(
                    model, train_loader, val_loader, pool_x,
                    variant=variant, epochs=EPOCHS, lr=LR, seed=seed,
                )
                save_run(OUT / f"{variant}_n{n_train}_seed{seed}.json", res)
                add_run(key, label, res)
                print(
                    f"  n={n_train} seed={seed} {variant:14s}: "
                    f"val_mse={res.val_mse[-1]:.4f} ({res.seconds:.1f}s)"
                )

            # Isotropic anchor (input-space, σ=0.5).
            for sigma in SIGMAS_ISO:
                variant = "v17_input"
                key = f"{variant}_iso_s{sigma}@n{n_train}"
                label = f"{variant} iso σ={sigma} (n={n_train})"
                model = PDFRegressor()
                res = run_pdf_experiment(
                    model, train_loader, val_loader, pool_x,
                    variant=variant, epochs=EPOCHS, lr=LR, seed=seed,
                    noise_std=sigma,
                )
                save_run(OUT / f"v17_input_iso_s{sigma}_n{n_train}_seed{seed}.json", res)
                add_run(key, label, res)
                print(
                    f"  n={n_train} seed={seed} v17_input_iso σ={sigma}: "
                    f"val_mse={res.val_mse[-1]:.4f} ({res.seconds:.1f}s)"
                )

            # Variance-scaled input.
            for sigma in SIGMAS_VAR:
                variant = "v17_input_var"
                key = f"{variant}_s{sigma}@n{n_train}"
                label = f"v17_input_var σ={sigma} (n={n_train})"
                model = PDFRegressor()
                res = run_pdf_experiment(
                    model, train_loader, val_loader, pool_x,
                    variant=variant, epochs=EPOCHS, lr=LR, seed=seed,
                    noise_std=sigma,
                    input_noise_scale=pixel_std,
                )
                save_run(OUT / f"v17_input_var_s{sigma}_n{n_train}_seed{seed}.json", res)
                add_run(key, label, res)
                print(
                    f"  n={n_train} seed={seed} v17_input_var σ={sigma}: "
                    f"val_mse={res.val_mse[-1]:.4f} ({res.seconds:.1f}s)"
                )

            # Variance-scaled embedding.
            for sigma in SIGMAS_VAR:
                variant = "v17_emb_var"
                key = f"{variant}_s{sigma}@n{n_train}"
                label = f"v17_emb_var σ={sigma} (n={n_train})"
                model = PDFRegressor()
                res = run_pdf_experiment(
                    model, train_loader, val_loader, pool_x,
                    variant=variant, epochs=EPOCHS, lr=LR, seed=seed,
                    noise_std=sigma,
                )
                save_run(OUT / f"v17_emb_var_s{sigma}_n{n_train}_seed{seed}.json", res)
                add_run(key, label, res)
                print(
                    f"  n={n_train} seed={seed} v17_emb_var σ={sigma}: "
                    f"val_mse={res.val_mse[-1]:.4f} ({res.seconds:.1f}s)"
                )

    # --- Summary ---
    def stats(key):
        vals = [r.val_mse[-1] for r in runs[key]]
        m = sum(vals) / len(vals)
        v = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)
        return m, v ** 0.5

    means = {k: stats(k)[0] for k in runs}
    stds = {k: stats(k)[1] for k in runs}

    summary = {
        "label_budgets": LABEL_BUDGETS,
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "sigmas_iso": SIGMAS_ISO,
        "sigmas_var": SIGMAS_VAR,
        "val_mse_mean": means,
        "val_mse_std": stds,
        "flow_ratios": flow["mean_flow_ratio"],
    }

    # Δ vs v16 per label budget
    delta_table: dict[str, dict[str, float]] = {}
    for n in LABEL_BUDGETS:
        v16_key = f"v16@n{n}"
        v16_mean = means[v16_key]
        delta_table[f"n={n}"] = {
            k: v16_mean - means[k]
            for k in runs if k.endswith(f"@n{n}")
        }
    summary["delta_vs_v16"] = delta_table
    with (OUT / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # --- Plots ---
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(LABEL_BUDGETS), figsize=(13, 4), sharey=False)
    for ax, n in zip(axes, LABEL_BUDGETS):
        keys_n = [k for k in runs if k.endswith(f"@n{n}")]
        keys_n_sorted = sorted(keys_n, key=lambda k: means[k])
        labels = [k.replace(f"@n{n}", "") for k in keys_n_sorted]
        ms = [means[k] for k in keys_n_sorted]
        es = [stds[k] for k in keys_n_sorted]
        xs = range(len(labels))
        ax.bar(xs, ms, yerr=es, capsize=3)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_title(f"n_train = {n}")
        ax.set_ylabel("val MSE")
    fig.suptitle(f"exp1b — low-label sweep (mean ± std over {len(SEEDS)} seeds)")
    fig.tight_layout()
    fig.savefig(OUT / "summary.png", dpi=140)
    plt.close(fig)

    # σ-sweep plot per label budget for variance-scaled variants
    fig, ax = plt.subplots(figsize=(8, 4))
    for n in LABEL_BUDGETS:
        xs = SIGMAS_VAR
        ys = [means[f"v17_input_var_s{s}@n{n}"] for s in xs]
        ax.plot(xs, ys, marker="o", label=f"v17_input_var, n={n}")
    ax.set_xscale("log")
    ax.set_xlabel("σ (fraction of per-pixel std)")
    ax.set_ylabel("val MSE (mean over seeds)")
    ax.set_title("exp1b — input-space variance-scaled σ sweep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "input_var_sigma_sweep.png", dpi=140)
    plt.close(fig)

    print("\n=== summary table ===")
    for n in LABEL_BUDGETS:
        print(f"\nn_train = {n}")
        v16_mean = means[f"v16@n{n}"]
        rows = sorted(
            [k for k in runs if k.endswith(f"@n{n}")],
            key=lambda k: means[k],
        )
        for k in rows:
            delta = v16_mean - means[k]
            mark = " *" if delta > 2 * stds[k] and k != f"v16@n{n}" else ""
            short = k.replace(f"@n{n}", "")
            print(f"  {short:30s}: {means[k]:.4f} ± {stds[k]:.4f}   Δ={delta:+.4f}{mark}")
    print(f"\nWrote results to {OUT}")


if __name__ == "__main__":
    main()
