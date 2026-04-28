# Relational / SSL — summary of findings

A 1-day CPU iteration that started from the 2024 PDF "Relational/Transductive Learning" idea and the 2026 ASCR research proposal, and produced a concrete, reproducible SSL recipe for MNIST regression along with three falsification results and several methodological controls. Full details, tables, and figures in [EXPERIMENTS.md](EXPERIMENTS.md). Code is all CPU-runnable in under 2.5 hours total.

---

## The story arc, one paragraph

The 2024 PDF claimed a 20 % MSE win on MNIST regression by adding a "transductive" loss involving unlabelled `q`. The 2026 docx proposed that loss is algebraically vacuous (Cancellation Theorem). We confirmed that empirically (exp0), then asked whether a small fix — replacing the cancelling `f(q)` with two independently-noised `f(q + ε_a)`, `f(q + ε_b)` — rescues anything (exp1). It does, but the relational form turns out to be just plain Π-model consistency wrapped in algebra (exp1c). Standard SSL components, with one user-contributed refinement (per-pixel-variance-scaled noise, exp1b), give a method that beats v15 by 20–32 % on MNIST regression at moderate label budgets and survives every methodological control we threw at it: matched optimiser steps (exp1e), held-out test eval (exp1g), and consistency-step-ratio K extrapolation (exp1h). One adjacent track — the docx's adaptive-σ ACR proposal — could not be tested on the current toy because the oracle isn't actually optimal there (exp2), so that work is gated on a toy redesign.

---

## The headline method

```
v18 = MSE(p, y)                                    # supervised
    + 2 · MSE(p − p_rev, y − y_rev)                # reversal augmentation
    + MSE(f(q + ε_a), f(q + ε_b))                  # plain Π-model on unlabelled q

ε_{a,b} = σ · per_pixel_std · z,    z ~ N(0, I),   σ = 0.5
```

* `p_rev = f(reverse(x))`, `y_rev = reverse(y)` (batch-flip reversal augmentation, the contributor of v16's win in exp0).
* Per-pixel std computed once over the unlabelled pool — gives near-zero noise to the 24.7 % of MNIST pixels that are always background. **The user's contribution and the part the literature does not normally do.**
* σ = 0.5 (or any "substantial" σ — a sigma below ~0.1 breaks the symmetry but doesn't deliver the SSL benefit).
* Train with K extra `MSE(z_a, z_b)`-only optimiser steps after each labelled-batch step. K is a regularisation hyperparameter and should be tuned per labelled budget. Always early-stop on val MSE.

---

## Falsifications and decisions, in one line each

| # | Tested | Outcome |
|---|---|---|
| **exp0** | 2024 PDF's "transductive" loss does anything | ❌ Falsified — gradient through `f(q)` is bit-exactly zero. The 18 % win came from reversal augmentation. |
| **exp1** | Symmetry-breaking with isotropic noise rescues the unlabelled gradient | ✅ Yes — but no measurable val-MSE gain at 5 000 labels. |
| **exp1b** | Variance-scaled noise + low labels | ✅ ~10 % MSE reduction over v16 at n ∈ {100, 250, 1 000}. Small "minimum" σ doesn't deliver the benefit even though it breaks symmetry. |
| **exp1c** | The relational wrapping in v17 adds anything beyond plain Π-model + reversal-aug (v18) | ❌ Cosmetic. v17 ≡ v18 within seed noise; at n = 250 v18 is even slightly better. |
| **exp1e** | Matched-optimiser-step v15 (4× epochs, all on labels) catches up to v17/v18_split | ❌ Doesn't catch up. SSL beats it by 12–25 σ — the win is genuine label-data efficiency, not compute. |
| **exp1f** | Inductive vs transductive (val-set inputs as pool) on val MSE | Indistinguishable on val MSE. Possible artifact — exp1g's job. |
| **exp1g** | Held-out test eval for the same grid | SSL benefit confirmed unbiased (~21–32 % reduction). Transductive is a clean +5 % win at n=100 only; otherwise indistinguishable. **No tailoring artifact.** |
| **exp1h** | Extrapolating K beyond 3 keeps paying | Up to a budget-dependent optimum, then collapse. K=10 wins at n=250 (additional 14 %); K=0 wins at n=1 000. |
| **exp2** | docx's adaptive σ(q) closes the gap between fixed-σ Π-model and oracle σ*(q) | ⚠️ Inconclusive — the toy lacks `oracle < fixed-σ` headroom; needs redesign before GPU spend. |

---

## What this means for label-expensive domains (e.g. molecular property prediction)

Three concrete takeaways:

1. **Pay the extra forwards.** When labels cost orders of magnitude more than compute, every optimiser step spent on `MSE(z_a, z_b)` over unlabelled molecules beats the same step spent re-processing the labelled set — by a wide margin (12–25 σ at matched compute, exp1e). v15 cannot be rescued by longer training.

2. **K is a real hyperparameter, and it depends on your labelled budget.** Picking your label budget on the SSL sweet spot (where supervised has just enough anchor to make consistency productive) and then sweeping K to the collapse point gives the best results. At the SSL sweet spot, the right K can give an extra 14 % on top of K = 0. With a lot of labels, the right K is 0 (consistency adds drift, not signal).

3. **Use the screening set as the unlabelled pool when labels are scarce.** Transductive training on val-set inputs gives a clean 5 % win at n = 100 (exp1g), washes out at n ≥ 200. It's free — you already know those inputs — so do it whenever you're in the low-label regime.

Things that surprised us:
* **The PDF's transductive loss was algebraically vacuous** — gradient through `f(q)` is bit-exactly zero. A future SSL loss can be checked the same way: replace unlabelled forwards with zeros and diff the gradients.
* **Variance scaling is a free semantic-preserving win at large σ but provides no rescue at small σ.** The intuition that "minimum perturbation that breaks symmetry should be enough" is empirically false: the consistency residual `~σ² · E[per_dim_std²]` must be large enough to bias the optimiser, regardless of whether the symmetry-breaking gradient flows.
* **K has a U-shaped optimum and a collapse mode.** Without best-epoch early stopping, K=100 looks catastrophic; with it, K=10 is the right answer at n=250.

---

## What's still open

| Track | Question | Cost to answer |
|---|---|---|
| **Pool distribution** | Does the unlabelled pool's distribution matter, or is consistency just generic regularisation? Compare v18 with real MNIST vs random Gaussian vs pixel-shuffled MNIST as the unlabelled pool. | ~5 min CPU. Cleanest single test of SSL utility. |
| **Toy redesign for ACR** | exp2's 1D `sin / 5·sin` toy doesn't have positive `oracle < fixed-σ` headroom, so adaptive σ(q) cannot be discriminated. Either dense labels with input-correlated noise, or piecewise Lipschitz with n_labeled ≥ 200. | ~30 min CPU to design and run a candidate. |
| **σ × K joint sweep** | Smaller σ probably allows larger K before drift, and vice versa. Map the Pareto front. | ~30 min CPU at one budget. |
| **Phase 1.5 (σ_jac dynamics)** | Track `‖∇f‖` distribution during training, compare the 4 mitigations from docx §4.5. Only worth doing on the redesigned toy. | ~1 hr CPU after toy is ready. |
| **Tier-1 cancellation note** | exp0 alone is publishable as a methodological note. Just needs writing-up. | Half a day of writing. |

---

## Reproducibility

```
uv sync                                          # one-time
uv run experiments/exp0_cancellation.py          # cancellation theorem
uv run experiments/exp1_break_cancellation.py    # symmetry-breaking
uv run experiments/exp1b_low_labels.py           # variance-scaled noise + low labels
uv run experiments/exp1c_pi_model_ab.py          # v17 vs v18 paired A/B
uv run experiments/exp1e_matched_steps.py        # matched-step fairness
uv run experiments/exp1f_learning_curve.py       # learning curve + transductive
uv run experiments/exp1g_held_out_test.py        # exp1f on held-out test
uv run experiments/exp1h_consistency_scaling.py  # K sweep
uv run experiments/exp2_synthetic_acr.py         # adaptive σ on 1D toy
```

Total wall-clock ~130 min on a single CPU. Every JSON in `results/` is regenerated from scratch on each run; matplotlib figures are deterministic given the JSON. Seeds are fixed.

If `uv sync` errors with TLS issues on macOS, prepend `UV_NATIVE_TLS=1`.
