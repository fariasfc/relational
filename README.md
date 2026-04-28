# Relational / ACR — CPU experiments

Quick falsification experiments for the Adaptive Consistency Regularization (ACR) research proposal. All experiments run on CPU in minutes.

## Setup

```
uv sync
```

## Run

```
uv run experiments/exp0_cancellation.py
uv run experiments/exp1_break_cancellation.py
uv run experiments/exp1b_low_labels.py
uv run experiments/exp1c_pi_model_ab.py
uv run experiments/exp1e_matched_steps.py
uv run experiments/exp1f_learning_curve.py
uv run experiments/exp1g_held_out_test.py
uv run experiments/exp1h_consistency_scaling.py
uv run experiments/exp2_synthetic_acr.py
```

Outputs land in `results/exp{0,1,1b,1c,1e,1f,1g,1h,2}/`. See [EXPERIMENTS.md](EXPERIMENTS.md) for the full design / observations / conclusions.

## What we are testing

- **exp0** — Reproduces the 2024-04-30 PDF's MNIST regression setup and tests the *Cancellation Theorem*: replacing `transductive_samples` with zeros must produce identical gradients to the PDF's "transductive" loss. If true, the original method's gain is from reversal augmentation, not transduction.
- **exp1** — Breaks the cancellation symmetry by replacing the two transductive forwards with `f(q + ε_a)` and `f(q + ε_b)` (independent draws), in either input or embedding space. Checks both that the unlabeled gradient now flows and whether val MSE actually improves over v16.
- **exp1b** — Low-label sweep (`n_train ∈ {100, 250, 1 000}`) with **variance-scaled noise** (`ε = σ · per_dim_std · z`) — the smallest σ that still breaks the symmetry while preserving the data semantics. Checks whether the SSL benefit appears at low labels and whether minimum-perturbation noise is sufficient.
- **exp1c** — Direct paired A/B between v17 (relational + noisy) and v18 (v16 + plain Π-model on q). Decides whether the relational wrapping adds anything beyond the standard SSL formulation.
- **exp1e** — Matched-optimiser-step fairness control: does v15 with 4× epochs match v17/v18 with 1× labelled epochs + 3× consistency-only steps? Tests whether the SSL win is genuine label-efficiency or a compute confound.
- **exp1f** — Learning curve over `n_train ∈ {50…5 000}` and head-to-head between *inductive* (separate unlabelled pool) and *transductive* (val-set inputs as pool).
- **exp1g** — Re-run of exp1f on a held-out test split disjoint from val. Reports both the in-domain val number and the unbiased test number; rules out tailoring artifact for transductive variants.
- **exp1h** — Sweep the consistency-step ratio `K ∈ {0, 1, 3, 10, 30, 100}` to find where extra unlabelled-only updates stop helping and start collapsing the model.
- **exp2** — On a 1D regression with 5× Lipschitz contrast (`y = sin(x)` on `[0, π]`, `y = 5·sin(x)` on `[π, 2π]`), compares supervised, fixed-σ Π-model, σ_jac, σ_dens, and oracle σ*(q). Pass: at least one adaptive variant closes ≥40% of the (fixed-σ → oracle) gap in-distribution and ≥60% under shift.
