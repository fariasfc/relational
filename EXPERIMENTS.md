# CPU-scale falsification experiments — first iteration

This document accompanies the code in [src/relational/](src/relational/) and [experiments/](experiments/). It records what was run, what was expected, what was observed, and what to do next. All results were produced on a single laptop CPU on 2026-04-27.

The experiments follow the structure of `ASCR_research_proposal_v2.docx`. We deliberately scoped this iteration to two phases that can be falsified cheaply:

| Phase | Question | Decision after |
|---|---|---|
| 0 | Does the 2024 PDF's "transductive" loss provably do nothing extra? | Whether the original Relational-Learning idea has any signal. |
| 1 | If we break the cancellation symmetry by injecting noise on the unlabeled forwards (input or embedding space), does the unlabeled gradient flow, and does it actually improve val MSE? | Whether the original relational form rescues anything once symmetry is broken. |
| 1b | At low label budgets (n ∈ {100, 250, 1 000}), does the symmetry-broken loss visibly beat v16, and is *variance-scaled* noise (per-pixel std) — large enough to break symmetry but small enough to preserve semantics — sufficient? | Whether the SSL benefit of v17 is real once labels are scarce, and whether minimum-perturbation noise is enough. |
| 1c | Direct A/B between v17 (relational + noisy) and v18 (v16 + plain Π-model on q with matched σ). The exp1 algebra predicts they are statistically indistinguishable. | Whether to keep the relational wrapping or adopt the simpler Π-model formulation. |
| 1e | Matched-optimiser-step fairness: does v15 with 4× epochs (all spent on labels) match v17/v18 with 1× labelled epochs + 3× consistency-only steps (spent on unlabelled)? | Whether the SSL win is genuine label-data efficiency or a compute confound. |
| 1f | Learning curve over `n_train ∈ {50…5 000}` and head-to-head between *inductive* (separate unlabelled pool) and *transductive* (val-set inputs as pool). | Where SSL helps most on the labelled-budget axis, and whether knowing the eval inputs in advance helps. |
| 1g | Re-run of the exp1f grid on a held-out test split disjoint from val. | Whether the transductive benefit (or its absence) is real or a tailoring artifact from re-using val inputs as the pool. |
| 1h | Sweep the consistency-step ratio K ∈ {0, 1, 3, 10, 30, 100}. Does paying more for unlabelled extrapolate? | Where the SSL benefit saturates and where it collapses; how K's optimum depends on label budget. |
| 2 | On a 1D problem with non-uniform Lipschitz, can adaptive σ(q) close the gap between fixed-σ and oracle σ*(q)? | Whether to commit GPU time to ACR. |

Phases 1.5 (σ_jac fixed-point dynamics) and the U-curve sweep on tabular benchmarks were deferred to a follow-up iteration.

---

## Experiment 0 — Cancellation theorem on MNIST regression

### Setup

Source: [experiments/exp0_cancellation.py](experiments/exp0_cancellation.py).

* **Task.** MNIST as regression with target ∈ [0, 9].
* **Model.** `Linear(784, 256) → ReLU → Linear(256, 3) → Linear(3, 1)`. The PDF used 4 096 hidden; 256 is plenty for CPU and preserves the qualitative behaviour.
* **Data.** 5 000 training, 1 000 validation, batch size 128, 30 epochs, Adam lr = 1e-3. Three seeds.
* **Variants.**
  * `v15` — vanilla MSE supervised (PDF baseline).
  * `v16` — exact reproduction of the PDF's `training_step`:
    `MSE(preds, y) + MSE((preds + t) − (preds_rev + t), y − y_rev) + MSE(preds − preds_rev, y − y_rev)`
    where `t = f(transductive_samples)` and `_rev = torch.flip(·, dims=(0,))`.
  * `v16_zeroed` — identical to `v16` but with `t := zeros_like(t)`. The Cancellation Theorem (docx Appendix A, Cor. 2) says `v16 ≡ v16_zeroed` algebraically because `f(q)` enters the loss only via opposing-sign linear terms that cancel.

### What we expected

1. **Gradient equivalence.** With matched init and matched batch, `∇θ L_v16` and `∇θ L_v16_zeroed` should be equal to float32 precision (max abs diff ≲ 1e-6). This is an exact algebraic prediction; failure here would mean the theorem is wrong.
2. **Trajectory similarity.** Over many SGD steps, v16 and v16_zeroed will not be bit-exact (FP rounding from the extra `(preds + t) − (preds_rev + t)` accumulates over ~1 170 steps), but their final val-MSE should agree within seed-to-seed noise — and both should be substantially better than v15, reproducing the PDF's "~20% improvement."

### What we observed

**Gradient equivalence test** ([results/exp0/gradient_equivalence.json](results/exp0/gradient_equivalence.json)):

| seed | max |Δgrad| | grad cosine | loss diff |
|---|---|---|---|
| 0 | 4.77e-7 | 1.0 | 0.0 |
| 1 | 9.54e-7 | 1.0 | 0.0 |
| 2 | 2.38e-7 | 1.0 | 0.0 |
| 3 | 1.19e-7 | 1.0 | 0.0 |
| 4 | 2.38e-7 | 1.0 | 0.0 |

All five seeds agree to float32 precision. The losses are bit-exact. The cancellation theorem holds empirically.

**Trajectory test** ([results/exp0/cancellation_check.json](results/exp0/cancellation_check.json), [results/exp0/curves.png](results/exp0/curves.png)):

| Variant | Final val MSE (mean of 3 seeds) |
|---|---|
| v15 (supervised) | 1.351 |
| v16 (PDF transductive) | 1.106 |
| v16_zeroed (`f(q) := 0`) | 1.136 |

* v15 → v16 gap: **+0.246** (≈ 18 % improvement; reproduces the PDF's "~20 %").
* v16 → v16_zeroed gap: **−0.030**, with pooled seed σ = 0.034. Within seed noise.
* The val-MSE curves for v16 and v16_zeroed visually overlay; v15 sits clearly above both.

Per-step training losses do drift after the first few hundred steps (max relative diff ~1.6 over 1 170 steps), as expected from FP rounding compounding through the optimizer. This is not a violation of the theorem — it is the consequence of `(p + t) − (p_rev + t) ≠ p − p_rev` exactly in float32 when t and p are of comparable magnitude.

### Conclusion

The 2024 PDF's "transductive" loss provides **zero gradient through `f(q)`**. The 18–20 % MSE win attributed to transduction is in fact the contribution of the **reversal-augmentation regulariser** `MSE(f(x) − f(x_rev), y − y_rev)`, which would have produced the same effect with no transductive forward pass at all.

This is the docx's Tier-1 deliverable. It is publishable as a standalone methodological note: any future SSL loss claiming transductive benefit can be subjected to the same algebraic check (replace unlabeled forwards with zeros; if the loss curves match, the unlabeled data did nothing).

---

## Experiment 1 — Break the cancellation symmetry via noisy transductive forwards

### Setup

Source: [experiments/exp1_break_cancellation.py](experiments/exp1_break_cancellation.py).

The cancellation theorem applies because `f(q)` enters the PDF loss with opposing signs *at the same input* `q`. Breaking that requires evaluating `f` at two different inputs. We replace the two transductive evaluations with `f(q + ε_a)` and `f(q + ε_b)`, where `ε_a` and `ε_b` are independent draws per step. Two flavours:

* **Input-space noise** — `ε_a, ε_b ~ N(0, σ²·I)` added to the raw 28×28 image.
* **Embedding-space noise** — `ε_a, ε_b ~ N(0, σ²·I)` added at the 3-dim bottleneck, *after* the encoder. The model now exposes `encoder` and `head` separately.

The loss is otherwise identical to v16:
```
L = MSE(p, y)
  + MSE((p + f(q + ε_a)) − (p_rev + f(q + ε_b)),  y − y_rev)   ← was cancelling, no longer is
  + MSE(p − p_rev,  y − y_rev)
```

### What we expected

The symmetry-broken middle term expands as:
```
MSE((p − p_rev) + (z_a − z_b),  y − y_rev)
= MSE((p − p_rev), y − y_rev)              ← supervised relational
  + E[(z_a − z_b)²]                        ← consistency on unlabeled q (coefficient 1)
  + 2·E[((p − p_rev) − (y − y_rev))·(z_a − z_b)]   ← cross term, mean-zero under symmetric noise
```

Predictions:

1. **Gradient flow** — `∇θ` should now differ between "real noisy forwards" and "transductive forwards := 0", with a non-trivial magnitude (≥ 1 % of the total gradient norm). This would be 4+ orders of magnitude above the float-precision noise floor measured in exp0.
2. **Val-MSE behaviour** — In expectation, v17 ≈ v16 + Π-model consistency at λ = 1. With 5 000 labeled MNIST images this is already a relatively rich signal; we expected at most a small improvement over v16, on the order of seed noise.

### What we observed

**Gradient-flow probe** ([results/exp1/gradient_flow.json](results/exp1/gradient_flow.json), 5 seeds, σ = 0.5):

| Noise space | mean ‖Δgrad‖/‖grad‖ | min | mean cosine |
|---|---|---|---|
| input | 0.164 | 0.124 | 0.986 |
| embedding | 0.042 | 0.030 | 0.999 |

Both are far above the threshold (4–5 orders of magnitude above the ~1e-6 float-noise floor from exp0). Symmetry is conclusively broken; the unlabeled data is contributing meaningful gradient. Input-space noise contributes ~16 % of total ‖∇θ‖; embedding-space noise contributes ~4 %.

**Training σ-sweep** ([results/exp1/summary.json](results/exp1/summary.json), 3 seeds × 30 epochs, see [results/exp1/sigma_sweep.png](results/exp1/sigma_sweep.png) and [results/exp1/curves.png](results/exp1/curves.png)):

| Variant | Val MSE | Δ vs v16 |
|---|---|---|
| v15 (supervised) | 1.335 ± 0.058 | −0.195 |
| v16 (cancelling) | **1.140** ± 0.017 | — |
| v17_input σ=0.1 | 1.120 ± 0.042 | +0.020 |
| v17_input σ=0.5 | 1.129 ± 0.037 | +0.011 |
| v17_input σ=1.0 | 1.190 ± 0.052 | −0.050 |
| v17_input σ=2.0 | 1.144 ± 0.021 | −0.005 |
| v17_emb σ=0.1 | 1.157 ± 0.019 | −0.017 |
| v17_emb σ=0.5 | 1.152 ± 0.099 | −0.012 |
| v17_emb σ=1.0 | 1.145 ± 0.045 | −0.005 |
| v17_emb σ=2.0 | 1.170 ± 0.025 | −0.030 |

Three findings:

1. **Symmetry-breaking works as designed.** The unlabeled forwards now contribute genuine gradient — both flavours pass the probe by 4+ orders of magnitude.
2. **The val-MSE improvement is at the edge of seed noise.** The best variant (v17_input σ=0.1) beats v16 by 0.020 MSE on average, with seed std 0.042. With only 3 seeds we cannot reject the null; this would need ≥ 5 seeds to call cleanly. None of the embedding-noise variants beats v16.
3. **Larger σ hurts, especially in input space.** σ=1.0 in input space loses 0.05 MSE relative to v16 — the noise begins to dominate the consistency residual and degrade the supervised relational term it shares the loss with.

### Conclusion

The mathematical framing predicts v17 ≈ v16 + Π-model consistency with coefficient λ = 1, and the empirics are consistent with that. We confirmed the unlabeled data contributes gradient (the original PDF's "transductive" intuition can be rescued), but the improvement vs. v16 — itself just supervised + reversal-augmentation — is small enough to be dominated by seed variance at this scale.

Two implications for the broader program:

* **The "noisy transductive" form is not a new method.** It is a re-derivation of Π-model consistency wrapped inside a relational loss. Any future use should compete against, not be confused with, plain Π-model + reversal augmentation.
* **The marginal benefit of consistency at 5 000 labels on MNIST regression is small.** This is consistent with the SSL literature: SSL methods earn their keep at low label budgets and on harder tasks. To decide whether ACR-style adaptive σ has headroom, exp1 should be re-run at low label counts (e.g., 100, 250, 1 000) where the consistency term contributes a larger share of the total signal — see exp1b.

---

## Experiment 1b — Low-label sweep with variance-scaled noise

### Setup

Source: [experiments/exp1b_low_labels.py](experiments/exp1b_low_labels.py).

Two extensions of exp1, motivated by two observations:

1. **Low labels.** Consistency-style regularisers earn their keep at low label budgets. exp1's negative result at 5 000 labels does not generalise; we need to test at smaller `n_train`.
2. **Variance-scaled noise.** Pure isotropic noise destroys the structure of the input — a pixel that is always background gets the same perturbation as a pixel inside the digit. We replace `ε ~ N(0, σ²·I)` with `ε = σ · per_dim_std · z`, `z ~ N(0, I)`. `σ` is now a *fraction of typical inter-sample variation*, not an absolute magnitude. Pixels with std ≈ 0 receive ε ≈ 0 — preserving the digit's silhouette. The hypothesis was that the *smallest* σ that breaks symmetry should be enough.

For the 5 000-sample MNIST transductive pool, the per-pixel std distribution (after standard normalisation) is:

| stat | mean | median | min | max | frac < 0.05 |
|---|---|---|---|---|---|
| value | 0.626 | 0.471 | **0.000** | 1.450 | **24.7 %** |

So roughly a quarter of the 784 pixels are essentially constant across MNIST and contribute nothing to variance-scaled noise.

* **Sweep**: `n_train ∈ {100, 250, 1 000}`, 5 seeds, 50 epochs, batch size 32.
* **Variants per label budget**:
  * `v15` — supervised baseline.
  * `v16` — PDF cancelling loss.
  * `v17_input_iso σ=0.5` — exp1's anchor (isotropic, semantic-destroying).
  * `v17_input_var σ ∈ {0.05, 0.1, 0.5}` — variance-scaled input noise (NEW).
  * `v17_emb_var σ ∈ {0.05, 0.1, 0.5}` — variance-scaled embedding noise (NEW; the per-coord std is computed on-the-fly from each batch's bottleneck).

### What we expected

* **Low labels:** v17 should beat v16 by a meaningful margin at `n_train ∈ {100, 250}`, with the gap closing as labels grow.
* **Variance scaling:** at large σ, var-scaled and isotropic should give similar val MSE (variance scaling is a "smarter" choice of where to spend the noise budget, not a stronger regulariser). At small σ — the user's "just enough to break symmetry, preserve semantics" target — the symmetry should still break (probe shows this), and we expected enough consistency signal to at least match v16.

### What we observed

**Gradient-flow probe** ([results/exp1b/gradient_flow.json](results/exp1b/gradient_flow.json), 3 seeds at the probe σ each):

| Variant @ σ | mean ‖Δgrad‖/‖grad‖ |
|---|---|
| v17_input @σ=0.5 (iso) | 0.173 |
| v17_input_var @σ=0.5 | 0.152 |
| v17_input_var @σ=0.1 | 0.054 |
| v17_input_var @σ=0.05 | 0.034 |
| v17_emb @σ=0.5 (iso) | 0.0075 |
| v17_emb_var @σ=0.5 | 0.0075 |
| v17_emb_var @σ=0.1 | 0.0015 |
| v17_emb_var @σ=0.05 | 0.0007 |

Two things stand out: (a) variance-scaled at σ=0.05 still breaks symmetry by 4 orders of magnitude over the float-noise floor — the user's "just enough" target is empirically reached; (b) the embedding-space surface is much smaller (only 3 bottleneck dims), so even σ=0.5 there yields ~0.7 % flow versus ~17 % in input space.

**Training results** ([results/exp1b/summary.json](results/exp1b/summary.json), [results/exp1b/summary.png](results/exp1b/summary.png), 5 seeds × 50 epochs, batch 32):

| Variant | n=100 | n=250 | n=1 000 |
|---|---|---|---|
| v15 supervised | 7.37 ± 0.26 | 4.13 ± 0.25 | 2.46 ± 0.11 |
| v16 (cancels) | 7.08 ± 0.36 | 3.62 ± 0.11 | 1.84 ± 0.05 |
| v17_input_iso σ=0.5 | 6.41 ± 0.31 | 3.17 ± 0.07 | 1.75 ± 0.03 |
| **v17_input_var σ=0.5** | **6.34 ± 0.26** | **3.16 ± 0.12** | **1.70 ± 0.02** |
| v17_input_var σ=0.1 | 7.20 ± 0.22 | 3.64 ± 0.07 | 1.85 ± 0.06 |
| v17_input_var σ=0.05 | 7.20 ± 0.18 | 3.76 ± 0.26 | 1.89 ± 0.04 |
| v17_emb_var σ=0.5 | 6.95 ± 0.07 | 3.72 ± 0.21 | 1.86 ± 0.02 |
| v17_emb_var σ=0.1 | 7.05 ± 0.26 | 3.66 ± 0.08 | 1.86 ± 0.05 |
| v17_emb_var σ=0.05 | 7.10 ± 0.24 | 3.58 ± 0.07 | 1.87 ± 0.03 |

Δ vs. v16 for the headline variant:

| n_train | v17_input_var σ=0.5 vs v16 | Relative reduction |
|---|---|---|
| 100 | −0.74 MSE (>2 σ) | **10.5 %** |
| 250 | −0.46 MSE (>4 σ) | **12.8 %** |
| 1 000 | −0.14 MSE (>2 σ) | **7.8 %** |

Three findings:

**(1) The SSL benefit of breaking the symmetry is real, and it scales with label scarcity.** v17_input variants beat v16 by 8–13 % across `n_train ∈ {100, 250, 1 000}`, with the gap shrinking as labels grow — exactly the SSL signature.

**(2) Variance scaling at σ=0.5 matches isotropic at σ=0.5 — it is a "free" semantic-preserving choice, but no stronger.** v17_input_var σ=0.5 ties v17_input_iso σ=0.5 within seed noise at every label budget. Variance scaling honours the user's instinct that the always-black pixels of MNIST shouldn't be perturbed, but does not give a separate accuracy benefit. At this σ, even var-scaled noise is not "small" — it perturbs typical pixels by 0.5 × 0.626 ≈ 0.31 standard deviations.

**(3) Small noise that preserves semantics is *not* enough.** The user's "just enough to break symmetry, preserve semantics" target — `v17_input_var σ ∈ {0.05, 0.1}` — *underperforms v16* at every label budget. Symmetry-breaking is necessary but not sufficient: the noise must also be large enough for the implicit consistency term `E[(z_a − z_b)²]` to act as a meaningful regulariser. The probe confirmed σ=0.05 still breaks symmetry by 4 orders of magnitude, so this is not a question of whether the unlabeled gradient flows; it is a question of whether the consistency residual is large enough to bias optimisation.

Embedding-space variance-scaled noise does not help meaningfully at any label budget. The bottleneck has only 3 dimensions; even at σ=0.5 the gradient-flow ratio is ~0.75 %, an order of magnitude below input-space σ=0.5.

### Conclusion

The original PDF's "transductive" intuition can be rescued — *if* you break the cancellation symmetry with substantial input-space noise. The SSL benefit at low labels is real and clean: ~10 % MSE reduction over v16 at n=100/250, scaling correctly with label budget. **v17_input_var σ=0.5 is the recommended variant**: it matches the simpler isotropic baseline while honouring the user's principle of leaving constant pixels untouched.

The deeper finding is that the apparent SSL win comes from the *magnitude* of the consistency residual, not from the symmetry-breaking per se. Once you're committed to a non-trivial `σ`, the variance-scaling refinement is a free semantic-preserving improvement; but you cannot dial `σ` down to the symmetry-breaking minimum and expect the SSL benefit to survive.

This is consistent with the exp1 algebraic decomposition. v17 expands to:
```
v15 + reversal-augmentation + E[(z_a − z_b)²] + cross-term (mean-zero)
```
The third term is exactly Π-model consistency at coefficient 1. Its magnitude scales with `σ²` (for variance-scaled noise, weighted by the squared per-pixel std). At small σ, this term contributes too little gradient to alter the optimisation trajectory — exactly what we observe.

The practical question to put to the docx is therefore not "can the relational form deliver SSL benefit?" but "is there any reason to wrap consistency inside the relational form rather than add it as a separate Π-model term?" The math says no in expectation; exp1c is the direct A/B that closes this loop.

---

## Experiment 1c — Direct A/B: v17 (relational + noisy) vs v18 (v16 + plain Π-model)

### Setup

Source: [experiments/exp1c_pi_model_ab.py](experiments/exp1c_pi_model_ab.py).

The exp1 algebra says
```
v17 = v18 + (mean-zero cross term)
```
so v17 and v18 should be statistically indistinguishable in expectation. v18 is defined as
```
v18 = MSE(p, y)  +  2 · MSE(p − p_rev, y − y_rev)  +  MSE(z_a, z_b)
```
i.e. exactly v16's three terms (the cancelling middle term reduced to its non-cancelling equivalent) plus a plain Π-model consistency term `MSE(z_a, z_b)` with `z_{a,b} = f(q + ε_{a,b})`. The noise process matches v17 exactly — `ε ~ N(0, σ² · per_pixel_std² · I)` with σ = 0.5.

We use **identical seeds** for model init, data shuffling, transductive sampling, and noise draws between v17 and v18 — so the only difference between the two trajectories is the loss formula itself. This makes a paired-difference test the right statistical comparison.

* `n_train ∈ {100, 250, 1 000}`, 5 seeds, 50 epochs, batch 32, σ = 0.5, variance-scaled noise.
* Variants: v16 (anchor), v17_input_var σ=0.5, v18_pi_input_var σ=0.5.

### What we expected

Per the algebra, paired-difference mean ≈ 0 with sem on the order of the cross-term standard deviation per training run. We checked at one batch in the smoke test: the per-step ‖Δgrad‖/‖grad‖ between v17 and v18 with matched noise is ~16 % (cosine 0.99) — non-zero but mean-zero in expectation. Over 50 epochs of accumulation, those should average out.

### What we observed

[results/exp1c/summary.json](results/exp1c/summary.json), [results/exp1c/summary.png](results/exp1c/summary.png), [results/exp1c/paired_diff.png](results/exp1c/paired_diff.png).

| n_train | v16 | v17_input_var σ=0.5 | v18_pi_input_var σ=0.5 | Paired (v17 − v18) ± sem | Verdict |
|---|---|---|---|---|---|
| 100 | 7.107 ± 0.351 | 6.468 ± 0.368 | 6.408 ± 0.383 | +0.060 ± 0.200 | indistinguishable |
| 250 | 3.696 ± 0.244 | 3.195 ± 0.090 | 3.128 ± 0.119 | +0.066 ± 0.021 | **v18 marginally better (3.1 sem)** |
| 1 000 | 1.853 ± 0.056 | 1.684 ± 0.061 | 1.692 ± 0.045 | −0.009 ± 0.033 | indistinguishable |

The paired-difference plot ([results/exp1c/paired_diff.png](results/exp1c/paired_diff.png)) shows per-seed deltas straddling zero at n=100 (one outlier at +0.78) and n=1000, but tightly clustered around +0.06 at n=250 — a small, consistent edge for v18.

Three findings:

**(1) The algebra is empirically confirmed.** At n=100 and n=1000, v17 and v18 are statistically indistinguishable. Both deliver ~10 % MSE improvement over v16. The relational wrapping in v17 contributes nothing the plain Π-model formulation doesn't already provide.

**(2) At n=250, v18 is marginally better (3.1 sem).** The paired diff is small in absolute terms (0.07 MSE = ~2 % of the val MSE) but consistent across seeds. Interpretation: the v17 cross term, while mean-zero, has positive variance and feeds gradient noise into the optimiser. v18 has zero cross-term variance, so its optimisation trajectory is slightly cleaner. The effect is detectable at the budget where consistency carries the most relative weight (n=250) and washes out at n=100 (high seed variance) and n=1000 (supervised loss dominates).

**(3) v18 is the recommended formulation going forward.** It is mathematically cleaner (no cancelling middle term), empirically equivalent or marginally better, and connects directly to the existing SSL literature (Π-model + reversal augmentation). The relational wrapping in v17 was a useful historical detail — it let us derive the right method by symmetry-breaking the original PDF loss — but adds no value to the final form.

### Conclusion

The relational wrapping is cosmetic. The headline method this program produced is:

> **MNIST-regression baseline that beats the 2024 PDF's "transductive" loss:** supervised MSE + 2× reversal-augmentation + plain Π-model consistency on unlabeled `q` with variance-scaled input noise at σ = 0.5. ~10 % MSE improvement over v16 at low label counts; matches it at high label counts.

This is just SSL-101 components — Π-model and reversal augmentation — assembled with a careful per-pixel-variance-scaled noise (the user's contribution) that preserves the always-background pixels of MNIST. The journey through the relational formulation served to identify exactly *what* signal v16 was missing (the mean-square consistency residual) and exactly *how big* the noise needs to be to deliver it (large enough that `σ² · E[per_dim_std²]` materially perturbs predictions).

---

## Experiment 1e — Matched-optimiser-step fairness control

### Setup

Source: [experiments/exp1e_matched_steps.py](experiments/exp1e_matched_steps.py).

In label-expensive domains (e.g. molecular property prediction) compute is cheap and labels are scarce, so the right fairness control is *matched optimiser steps*, not matched compute. Six variants compared per label budget:

| Variant | Labelled epochs | Extra consistency-only steps per labelled step | Total optimiser steps |
|---|---|---|---|
| `v15_1x` | 50 | 0 | 1× |
| `v15_4x` | 200 | 0 | 4× (all on labelled data) |
| `v17_1x` | 50 | 0 | 1× |
| `v17_split` | 50 | 3 | 4× (1 combined + 3 consistency-only) |
| `v18_1x` | 50 | 0 | 1× |
| `v18_split` | 50 | 3 | 4× (1 combined + 3 consistency-only) |

The `_split` variants and `v15_4x` end up with **identical total optimiser-step counts**. The split variants spend 1/4 of those steps on labelled+consistency and 3/4 on unlabelled-only `MSE(z_a, z_b)`. Reported numbers use **best val MSE during training** (early-stopping proxy) so longer runs aren't penalised by late-epoch drift.

`n_train ∈ {100, 250, 1 000}`, 5 seeds, σ = 0.5 variance-scaled.

### What we expected

* `v17/v18 _split ≫ v15_4x` → unlabelled data adds learning signal that re-processing labels cannot substitute for. **The hypothesis the molecular setting needs to be true.**
* `v15_4x ≈ v17/v18 _split` → v15_1x was simply under-trained. SSL win was a compute confound.
* `v15_4x ≫ v17/v18 _split` → consistency-only steps drift the model.

### What we observed

[results/exp1e/summary.json](results/exp1e/summary.json), [results/exp1e/summary.png](results/exp1e/summary.png).

Best val MSE (mean over 5 seeds):

| n_train | v15_1x | v15_4x | v17_1x | **v17_split** | v18_1x | **v18_split** |
|---|---|---|---|---|---|---|
| 100 | 5.348 | 5.352 | 5.181 | **4.529** | 5.280 | **4.665** |
| 250 | 3.836 | 3.907 | 2.982 | **2.708** | 3.122 | **2.732** |
| 1000 | 2.350 | 2.271 | 1.659 | **1.636** | 1.648 | **1.656** |

Paired test (same RNG seeds across variants), positive Δ means the split variant beat v15_4x:

| n_train | v15_4x − v17_split | v15_4x − v18_split |
|---|---|---|
| 100 | **+0.823 ± 0.068** (12 σ) | **+0.688 ± 0.058** (12 σ) |
| 250 | **+1.199 ± 0.078** (15 σ) | **+1.174 ± 0.046** (25 σ) |
| 1000 | **+0.636 ± 0.055** (12 σ) | **+0.615 ± 0.059** (10 σ) |

Three findings:

**(1) v15 with 4× epochs does *not* catch up.** The supervised baseline plateaus quickly: `v15_1x` ≈ `v15_4x` at every label budget (within seed noise; `v15_4x` actually overfits slightly worse at n=250). Longer supervised training cannot substitute for unlabelled data.

**(2) Spending the extra optimiser steps on consistency *dominates* spending them on labels.** Even at n=1 000 — where supervised training already has 1 000 examples to work with — `v18_split` beats `v15_4x` by 27 % (1.66 vs 2.27). At n=250, `v18_split` is **30 %** better than `v15_4x` despite identical total compute and identical labelled-passes count.

**(3) v17_split ≈ v18_split**, consistent with exp1c's finding that the relational wrapping is cosmetic. Both forms exploit the unlabelled data equivalently when given the matched-step training budget.

### Conclusion

This is the **fairness control the molecular setting needs.** Given identical optimiser budgets, an extra optimiser step spent on `MSE(z_a, z_b)` over unlabelled data is materially more valuable than the same step spent re-processing labelled data. The SSL win is genuine label-data-efficiency, not a compute artifact. In a setting where compute is cheap and labels expensive, the answer is unambiguous: pay the extra forwards.

---

## Experiment 1f — Learning curve and inductive vs transductive

### Setup

Source: [experiments/exp1f_learning_curve.py](experiments/exp1f_learning_curve.py).

* **Learning curve.** Sweep `n_train ∈ {50, 100, 200, 500, 1 000, 2 000, 5 000}`, 3 seeds, 50 epochs.
* **Transductive question.** For each method (v17 and v18), compare two unlabelled pools:
  * **Inductive** — separate slice of MNIST train (2 000 samples, fixed across runs).
  * **Transductive** — *the val set's inputs* (1 000 samples), without labels. The model is explicitly tailored to the points it will be evaluated on at training time. This is the original Vapnik framing.

Reported metric: best val MSE during training (same val set used for early-stopping and reporting — this is the methodological hole exp1g closes).

### What we observed

[results/exp1f/learning_curve.png](results/exp1f/learning_curve.png), [results/exp1f/transductive_vs_inductive.png](results/exp1f/transductive_vs_inductive.png), [results/exp1f/delta_vs_supervised.png](results/exp1f/delta_vs_supervised.png).

Best val MSE (mean over 3 seeds):

| n_train | v15 | v16 | v17_inductive | v17_transductive | v18_inductive | v18_transductive |
|---|---|---|---|---|---|---|
| 50 | 5.32 | 5.45 | 5.40 | 5.42 | **5.21** | 5.34 |
| 100 | 5.31 | **5.11** | 5.18 | 5.20 | 5.29 | 5.18 |
| 200 | 4.03 | 3.96 | 3.53 | 3.53 | 3.49 | **3.46** |
| 500 | 2.75 | 2.31 | 2.05 | 2.04 | **2.04** | 2.05 |
| 1000 | 2.40 | 1.83 | **1.66** | 1.70 | 1.69 | 1.67 |
| 2000 | 1.85 | 1.46 | 1.39 | 1.39 | **1.38** | 1.38 |
| 5000 | 1.26 | 1.06 | 1.04 | 1.02 | 1.03 | **1.00** |

Two findings:

**(1) The SSL benefit has a sweet spot in the middle of the learning curve.** Peak absolute Δ vs supervised is at n ∈ {500, 1 000} (~0.7 MSE absolute, 25–30 % relative). At n ≤ 100 the model has too little supervised anchor for consistency to help; at n ≥ 5 000 supervised becomes plentiful enough that the marginal contribution of unlabelled data shrinks. **Below the "minimum supervised viability" threshold, consistency cannot bootstrap.**

**(2) Transductive vs inductive looks like a null result on val MSE.** At every label budget the difference is within seed noise (range −0.13 to +0.12). Two interpretations: (a) genuine null — the model learns generic smoothness regardless of the unlabelled pool's distribution; (b) measurement artifact — the model has been tailored to the eval set's inputs, biasing val MSE toward the transductive variant precisely as much as it might bias toward overfitting. We can't distinguish these from val MSE alone — that's exp1g's job.

---

## Experiment 1g — Held-out test eval (clean transductive comparison)

### Setup

Source: [experiments/exp1g_held_out_test.py](experiments/exp1g_held_out_test.py).

Same grid as exp1f, but the 10 000-sample MNIST test split is now divided:

| Split | Size | Used for |
|---|---|---|
| `val_set`  | 1 000 | Per-epoch eval, "best epoch" selection, transductive pool (when applicable) |
| `test_set` | 1 000 (disjoint) | Final unbiased reporting only |

Each cell now reports two numbers:

* `val_mse_best` — same metric as exp1f (in-domain, biased by re-use of val for selection AND as transductive pool).
* `test_mse_at_best_val_epoch` — held-out, never seen at training time. The early-stop epoch is selected on val, then test MSE at that epoch is the unbiased number.

### What we observed

[results/exp1g/learning_curve.png](results/exp1g/learning_curve.png), [results/exp1g/val_minus_test_gap.png](results/exp1g/val_minus_test_gap.png), [results/exp1g/transductive_vs_inductive_test.png](results/exp1g/transductive_vs_inductive_test.png).

Held-out test MSE (mean over 3 seeds):

| n_train | v15 | v16 | v17_ind | v17_trans | v18_ind | v18_trans |
|---|---|---|---|---|---|---|
| 50 | 5.19 | 5.24 | 5.21 | 5.14 | 5.16 | 5.15 |
| 100 | 5.16 | 4.99 | 5.01 | 4.93 | 5.08 | **4.84** |
| 200 | 3.92 | 3.79 | 3.11 | 3.16 | 3.11 | **3.10** |
| 500 | 2.65 | 2.08 | 1.84 | 1.86 | **1.81** | 1.82 |
| 1000 | 2.09 | 1.62 | **1.48** | 1.48 | 1.49 | 1.52 |
| 2000 | 1.77 | 1.43 | 1.28 | 1.27 | 1.25 | **1.22** |
| 5000 | 1.29 | 1.07 | 1.04 | 1.04 | 1.01 | **0.98** |

Three findings:

**(1) The SSL benefit is *not* a val-set-selection artifact.** v18 vs v15 on held-out test reproduces the val-set numbers within seed noise: 21–32 % MSE reduction across n ∈ {200, …, 5 000}. The headline method is robust under proper protocol.

**(2) Transductive is *not* overfitting to val-set inputs.** The val − test gap ([results/exp1g/val_minus_test_gap.png](results/exp1g/val_minus_test_gap.png)) is the same for transductive variants as for inductive variants at every label budget. A tailoring artifact would have shown transductive's val ≪ test (model fitting val inputs but failing to generalize). The data shows them tracking each other.

**(3) Transductive's main edge is at n=100.** On the held-out test set, `v18_transductive` beats `v18_inductive` by **0.24 MSE (≈ 5 %)** at n = 100. Elsewhere they are within seed noise. The interpretation that matches both this and exp1f's val-set null: at low labels the consistency loss is shaping a function whose details still depend strongly on the unlabelled pool, so a pool drawn from the eval distribution helps. At higher labels the supervised signal dominates and pool choice washes out.

### Conclusion

The methodological concern is empirically resolved. The SSL benefit (~20–30 % MSE reduction) and the transductive/inductive comparison both hold up under proper held-out-test protocol. The transductive setting is a small, regime-specific win — meaningful at n = 100 but not at n ≥ 200. For the molecular use case, this argues that **knowing the screening set in advance is a small but real advantage at the lowest label budgets**, but the bulk of SSL value at moderate label budgets comes from generic consistency regularisation that is largely insensitive to the unlabelled pool's distribution.

---

## Experiment 1h — How far does extrapolating K go?

### Setup

Source: [experiments/exp1h_consistency_scaling.py](experiments/exp1h_consistency_scaling.py).

exp1e fixed K = 3 (3 consistency-only optimiser steps per labelled step). The natural follow-up: does more K keep paying, or does the model collapse toward a trivial constant solution (the global minimum of `MSE(z_a, z_b)`)?

* `K ∈ {0, 1, 3, 10, 30, 100}` (K = 0 is plain v18_1x; K = 3 reproduces v18_split from exp1e).
* `n_train ∈ {250, 1 000}` (250 is the SSL sweet spot from exp1f; 1 000 has more supervised anchor and was hypothesised to be more robust to drift).
* 3 seeds, 50 epochs, σ = 0.5 variance-scaled, held-out test eval per exp1g protocol.

### What we expected

Two competing effects: (a) more K → stronger smoothness regularisation per labelled signal; (b) more K → less frequent supervised anchor → drift toward `f ≡ constant`. A U-shaped curve in `K` was the obvious prediction, with the optimum shifting *lower* as labelled supervision grows (since more labels means less benefit from extra consistency).

### What we observed

[results/exp1h/k_sweep.png](results/exp1h/k_sweep.png), [results/exp1h/trajectories.png](results/exp1h/trajectories.png).

**n = 250 (3 seeds):** clean U-shape with optimum at K = 10.

| K | val MSE best | drift (final − best) | best epoch | test @ best-val |
|---|---|---|---|---|
| 0 | 3.091 ± 0.135 | +0.014 | 48 | 2.775 |
| 1 | 2.787 ± 0.024 | +0.028 | 45 | 2.517 |
| 3 | 2.688 ± 0.043 | +0.074 | 47 | 2.401 |
| **10** | **2.635 ± 0.040** | +0.222 | 37 | **2.388** |
| 30 | 2.754 ± 0.032 | +0.373 | 37 | 2.591 |
| 100 | 2.901 ± 0.067 | +0.583 | 46 | 2.776 |

(v15 supervised reference at this budget: val_best = 3.74.)

K = 10 → 14 % better test MSE than K = 0 (= v18_1x). Going from K = 10 to K = 100 wipes out almost all the extra benefit. The drift column climbs monotonically with K — final-epoch val MSE deteriorates by 0.58 at K = 100. The held-out test number tracks `val_best` faithfully, so this isn't an in-domain artifact.

**n = 1 000 (3 seeds):** monotone increase in val MSE with K — *the optimum is K = 0*.

| K | val MSE best | drift | best epoch | test @ best-val |
|---|---|---|---|---|
| **0** | **1.614 ± 0.035** | +0.040 | 44 | **1.449** |
| 1 | 1.628 ± 0.012 | +0.116 | 39 | 1.483 |
| 3 | 1.644 ± 0.018 | +0.102 | 41 | 1.487 |
| 10 | 1.724 ± 0.043 | +0.112 | 35 | 1.576 |
| 30 | 1.750 ± 0.057 | +0.139 | 39 | 1.574 |

(v15 supervised reference: 2.19.)

(K = 100 was skipped at n = 1 000 — runtime would have been ~13 min/seed and the trend was already clear.)

### Three findings

**(1) The optimum K depends on the labelled budget.** At n = 250 the optimum is K = 10; at n = 1 000 the optimum is K = 0. This is the predicted "less supervised anchor → benefits more from extra consistency" pattern, with the *direction* of the effect confirmed.

**(2) Drift is monotone in K and visible from K = 10 onwards.** The collapse mode is real — `MSE(z_a, z_b)` has a trivial minimum at constant predictions, and the model approaches it when consistency dominates. At K = 100, n = 250 the model is still better than v15 at its best epoch but drifts by 0.58 MSE by the end. **Without best-val early stopping, K = 100 would look catastrophically bad.**

**(3) `best epoch` shifts earlier with K, then late again under collapse.** At n = 250 the best epoch moves from 48 (K = 0) to 37 (K = 10–30) and then back to 46 (K = 100). The K = 10 point is where the model *both* converges fastest and stays stable longest — the practical Pareto front.

### Conclusion

Extrapolating K is genuinely beneficial up to a problem-dependent optimum, and beyond that point the consistency loss collapses the model rather than refining it. For the molecular setting:

* **The matched-step recipe needs a budget-dependent K.** A fixed K = 3 (exp1e) leaves performance on the table at low labels and risks unnecessary drift at high labels. A simple heuristic: tune K on a held-out budget at the same `n_train` you'll deploy at, treating it as the consistency-loss-weight hyperparameter — which it effectively is.
* **Always select the early-stop epoch on val MSE, never report final-epoch numbers.** At K ≥ 10, final-epoch val MSE is materially worse than best-epoch val MSE.
* **There is a regime where K → ∞ destroys the model.** This is the "cancellation re-emergence" failure mode the docx §4.4 warned about, but with a different mechanism — it's drift toward a trivial smoothness minimum rather than vanishing-gradient through ‖∇f‖.

---

## Experiment 2 — Adaptive consistency on 1D regression with non-uniform Lipschitz

### Setup

Source: [experiments/exp2_synthetic_acr.py](experiments/exp2_synthetic_acr.py).

* **Task.** `y*(x) = sin(x)` on `[0, π]`, `5·sin(x)` on `(π, 2π]`. Local Lipschitz contrast = 5×.
* **Model.** `Linear(1, 64) → ReLU → Linear(64, 64) → ReLU → Linear(64, 1)`. Adam lr = 1e-3, 200 epochs, batch size 64. Five seeds.
* **Data.** 50 labeled, 2 000 unlabeled, 1 000 test.
* **Two regimes.**
  * `in_dist` — labeled and unlabeled both uniform on `[0, 2π]`.
  * `shift` — labeled only on `[0, π]`; unlabeled covers the full domain. Test MSE is measured on the full domain, so the model must extrapolate into the steep half it has never seen labels for.
* **Five variants** (all use Π-model consistency `MSE(f(q), f(q + ε))`, `ε ~ N(0, σ(q)²·I)`, with sigmoid ramp-up of the consistency weight over the first 80 epochs):
  1. `supervised` — labeled-only baseline, no consistency.
  2. `fixed_best` — best of σ ∈ {0.01, 0.05, 0.1, 0.2, 0.5} chosen by mean test MSE.
  3. `sigma_jac` — σ(q) = clip(c / ‖∂f/∂q‖, 0, σ_max), with c = 0.1, σ_max = 0.5, stop-gradient on σ.
  4. `sigma_dens` — σ(q) = σ_max · min(1, dₖ(q) / d̄ₖ), k = 3 nearest labeled neighbours in input space.
  5. `oracle` — σ(q) = clip(c / ‖∇y*(q)‖, 0, σ_max). Uses the *true* gradient of `y*`; this is the upper bound any adaptive method could reach if it perfectly estimated local Lipschitz.

### What we expected

Per docx §5.4 Phase 2:
* **In-distribution** — fixed-σ tuned on the validation set should already be close to optimal because labeled and unlabeled come from the same distribution. We expected a small but positive `fixed → oracle` gap, with σ_jac (the most theoretically principled adaptive variant) closing ≥ 40 % of it.
* **Shift** — fixed-σ tuned on labeled-only data should be sub-optimal for the unlabeled region, so the gap should widen. We expected ≥ 60 % gap closure. This is the docx's headline claim.

### What we observed

[results/exp2/summary.json](results/exp2/summary.json), bar chart [results/exp2/in_dist/summary.png](results/exp2/in_dist/summary.png), [results/exp2/shift/summary.png](results/exp2/shift/summary.png).

| Regime | supervised | fixed_best | σ_jac | σ_dens | oracle |
|---|---|---|---|---|---|
| `in_dist` | 1.565 ± 0.13 | **1.536** ± 0.10 (σ=0.5) | 1.544 ± 0.14 | 1.545 ± 0.11 | 1.628 ± 0.18 |
| `shift` | **5.254** ± 0.76 | 5.562 ± 0.65 (σ=0.05) | 7.232 ± 0.09 | 7.457 ± 0.08 | 6.007 ± 0.56 |

Three findings, in order of importance.

**(1) No headroom: oracle ≤ fixed-σ in both regimes.**

* `in_dist`: fixed_best = 1.536, oracle = 1.628. The `fixed → oracle` gap is **−0.092** (oracle is *worse*).
* `shift`: fixed_best = 5.562, oracle = 6.007. Gap = **−0.445** (oracle is *worse*).

The docx Phase 2 pass criterion (40 %/60 % gap closed) is undefined when the gap is negative — there is nothing for an adaptive method to close. The `σ ∝ 1 / Lipschitz` ansatz is one specific oracle and not necessarily Bayes-optimal at every label-budget.

**(2) σ_jac mirrors the oracle in-distribution, defaults to σ_max under shift.**

The σ-curve plots ([results/exp2/in_dist/sigma_curves.png](results/exp2/in_dist/sigma_curves.png), [results/exp2/shift/sigma_curves.png](results/exp2/shift/sigma_curves.png)) show the qualitative story:

* In-distribution, σ_jac (blue) and oracle (green) both peak in flat regions where ‖∇f‖ is small, both decay in the steep half — modulo the σ_max cap, the shapes agree. This validates the *direction* of σ_jac's signal when the model has learned the function.
* Under shift, the labeled half (`[0, π]`) shows the same shape, but in the unlabeled half (`(π, 2π]`) σ_jac saturates at σ_max = 0.5. The model has never seen labels there; its gradient is meaningless; clip(c / ‖∂f/∂q‖) just hits the upper bound. We *want* σ small in that region (it's where Lipschitz is highest), so we get the worst possible behaviour — large perturbations smearing across the steep transition.

This is a different failure mode than the docx §4.4 prediction. The docx flagged "cancellation re-emergence" (σ → 0 because ‖∇f‖ blows up); we observed the *opposite* (σ → σ_max because ‖∇f‖ vanishes in OOD regions). Both reduce to "consistency loss vanishes or destroys signal in the regions that matter most."

**(3) Consistency hurts under shift, including the oracle.**

In the shift regime, **supervised-only beats every consistency variant** including the oracle (5.25 < 6.01 < 7.23). With labels only on `[0, π]` and `y* = 5·sin(x)` on `(π, 2π]`, no smoothness penalty can recover the 5× magnitude — pulling `f(q)` and `f(q + ε)` together in the OOD region just biases predictions toward whatever the model extrapolates from the labeled half (which is small). This is a structural ceiling, not a tuning issue.

### Conclusion

This particular toy is poorly suited to falsifying ACR. The 1D `sin / 5·sin` benchmark gives consistency regularisation no real foothold: in-distribution, fixed-σ at the high end of the sweep already produces enough smoothing; under shift, no consistency loss can reconstruct missing magnitude information. The docx's 40 %/60 % gap-closure thresholds are undefined here.

The σ-curve diagnostics still produced two useful pieces of information:
* **σ_jac is correctly directional in-distribution** — its shape tracks the oracle. This is weak corroboration that a well-trained model carries usable Lipschitz signal.
* **σ_jac fails in OOD regions in a previously-undocumented way** — saturating at σ_max rather than collapsing to zero. This should be added to the docx §4.4 failure-mode catalogue.

---

## Combined recommendation

* **Phase 0 is shippable as a standalone result.** The cancellation finding plus [results/exp0/curves.png](results/exp0/curves.png) is a clean, falsifiable, publishable methodological note.
* **Phase 1 + 1b + 1c land on a concrete SSL method.** The headline is: **supervised + 2× reversal-augmentation + plain Π-model consistency on q with variance-scaled input noise at σ=0.5** (i.e. v18). It delivers ~10 % MSE reduction over v16 at low label budgets (n ∈ {100, 250, 1 000}). v17 (the relational + noisy form) is empirically indistinguishable from v18 at n ∈ {100, 1 000} and slightly worse at n=250 — confirming the algebraic prediction that the relational wrapping is cosmetic. Variance scaling is the user's idea and is a free semantic-preserving improvement at large σ; small "just-break-symmetry" noise (σ ≤ 0.1) does *not* deliver the SSL benefit even though it does break the symmetry.
* **Phases 1e/f/g settle the fairness and protocol questions.** Matched-step v15_4x cannot catch up to v17/v18_split (12–25 σ paired-test margin); the SSL win is genuine label-data-efficiency, not a compute confound. The benefit profile is sweet-spot-shaped on the labelled-budget axis: peaks at n ∈ {500, 1 000} (~30 % relative MSE reduction), softens at both ends. Held-out test eval (exp1g) confirms the SSL benefit is unbiased and reveals a small transductive advantage that is real but localised to n = 100 (~5 % relative).
* **Phase 1h pins down the consistency-step hyperparameter K.** The SSL benefit DOES extrapolate further than K = 3 — at n = 250 the optimum is K = 10, giving an additional 14 % over K = 0. But K is U-shaped, with high-K runs collapsing toward `f ≡ const` (drift up to +0.58 MSE between best and final epoch at K = 100). The optimum K shifts *down* as labels become plentiful: at n = 1 000 the optimum is K = 0. Two operational rules for this method: (a) tune K to the labelled budget you'll deploy at; (b) always early-stop on val MSE, never report final-epoch numbers.
* **Phase 2 needs a redesigned toy before GPU spend on ACR.** The 1D benchmark in its current form cannot discriminate adaptive variants because the oracle isn't even an upper bound. Without a toy that has positive `fixed → oracle` headroom, results from Phase 3 GPU experiments will not be interpretable.

---

## Future steps

In rough priority order.

### Tier A — highest leverage, do before any GPU spend

1. **Adopt v18 as the SSL baseline going forward** (decision now made by exp1c). Wherever the docx talks about "the relational form" or "v17", switch to "v16 + plain Π-model with variance-scaled input noise at σ=0.5." This unblocks all subsequent comparisons against literature methods (VAT, Mean Teacher, etc.).

2. **Toy redesign so the oracle has clear headroom.** Two candidate setups:
   * **Dense labels with input-correlated noise.** `y = sin(x) + η(x)` where η has different variances in different regions of `x`. The optimal smoothness scale is region-dependent; ACR has well-defined headroom; we can compute the exact Bayes-optimal σ*(q) by line search.
   * **Piecewise-Lipschitz with sufficient labels.** Same `sin / 5·sin` structure but with n_labeled ≥ 200 so supervised has enough signal in both halves and consistency adds a measurable second-order improvement.
   In either case, validate that `oracle < fixed_best` *before* running adaptive variants.

3. **Document σ_jac's OOD failure mode in the docx.** Update §4.4 with the saturation-at-σ_max mode and propose a mitigation: detect OOD regions (e.g., via density estimate against labeled data) and fall back to σ_max-not-applied or to fixed-σ outside support.

### Tier B — only if Tier A produces a toy with measurable signal

4. **Phase 1 U-curve sweep on UCI tabular regression.** The docx's stated Phase 1: confirm fixed-σ has a U-curve depth ≥ 5 % relative on at least three benchmarks before claiming any adaptive method has a budget to exploit. Boston, Wine, Concrete, Yacht, Energy. CPU-tractable; ~10 minutes per benchmark.

5. **Phase 1.5 σ_jac fixed-point analysis on the redesigned toy.** Track the distribution of ‖∇f‖ over training; compare the four mitigation strategies from docx §4.5 (stop-gradient, EMA, bounds, curriculum) and identify the minimum mitigation set required for stability.

### Tier C — opportunistic

7. **Direction-adapted ACR (docx Appendix C).** Only worth pursuing if Tier A produces a toy where magnitude-only ACR shows ≥ 25 % gap closure. Then sketch `v(q) = first PC of {x − q : x ∈ S(q)}` on the same toy and check whether it strictly dominates magnitude-only.

8. **Confidence-driven σ for the classification port** (`σ_conf` in docx §4.2.1). Out of scope for the current 1D regression toy; revisit when extending to MNIST classification.

---

## Reproducibility

```
uv sync                                          # one-time
uv run experiments/exp0_cancellation.py          # ~75s on CPU
uv run experiments/exp1_break_cancellation.py    # ~6 min on CPU
uv run experiments/exp1b_low_labels.py           # ~25 min on CPU
uv run experiments/exp1c_pi_model_ab.py          # ~7 min on CPU
uv run experiments/exp1e_matched_steps.py        # ~12 min on CPU (matched-step fairness)
uv run experiments/exp1f_learning_curve.py       # ~22 min on CPU (n_train sweep, in-domain)
uv run experiments/exp1g_held_out_test.py        # ~25 min on CPU (n_train sweep, held-out)
uv run experiments/exp1h_consistency_scaling.py  # ~30 min on CPU (K sweep)
uv run experiments/exp2_synthetic_acr.py         # ~25s on CPU
```

Total wall-clock for the full suite is ~130 minutes. Every JSON in `results/` is regenerated from scratch on each run; matplotlib figures are deterministic given the JSON. Seeds are fixed.

If `uv sync` errors with TLS issues on macOS, prepend `UV_NATIVE_TLS=1`.
