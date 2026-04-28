# Relational / SSL — summary of findings

A 1-day CPU iteration that started from the 2024 PDF "Relational/Transductive Learning" idea and the 2026 ASCR research proposal, and produced a concrete, reproducible SSL recipe for MNIST regression along with three falsification results and several methodological controls. Full details, tables, and figures in [EXPERIMENTS.md](EXPERIMENTS.md). Code is all CPU-runnable in under 2.5 hours total.

---

## The story arc, one paragraph

The 2024 PDF claimed a 20 % MSE win on MNIST regression by adding a "transductive" loss involving unlabelled `q`. The 2026 docx proposed that loss is algebraically vacuous (Cancellation Theorem). We confirmed that empirically (exp0), then asked whether a small fix — replacing the cancelling `f(q)` with two independently-noised `f(q + ε_a)`, `f(q + ε_b)` — rescues anything (exp1). It does, but the relational form turns out to be just plain Π-model consistency wrapped in algebra (exp1c). Standard SSL components, with one user-contributed refinement (per-pixel-variance-scaled noise, exp1b), give a method that beats v15 by 20–32 % on MNIST regression at moderate label budgets and survives every methodological control we threw at it: matched optimiser steps (exp1e), held-out test eval (exp1g), and consistency-step-ratio K extrapolation (exp1h). **Then we ported it to TDC Lipophilicity_AstraZeneca + Mordred** (exp1f-tdc, exp1h-tdc) and the recipe failed — under scaffold split, v18 matches v15 on val but degrades by 25–55 % on held-out test, because SSL on the in-distribution unlabelled pool actively biases the model away from the OOD test scaffolds. The MNIST recipe was correct but specific to the in-distribution-eval regime. One adjacent track — the docx's adaptive-σ ACR proposal — could not be tested on the current toy because the oracle isn't actually optimal there (exp2), so that work is gated on a toy redesign.

---

## The headline method (qualified)

```
v18 = MSE(p, y)                                    # supervised
    + 2 · MSE(p − p_rev, y − y_rev)                # reversal augmentation (image-regression-specific)
    + MSE(f(q + ε_a), f(q + ε_b))                  # plain Π-model on unlabelled q

ε_{a,b} = σ · per_dim_std · z,    z ~ N(0, I),   σ = 0.5
```

* **Where it works** (exp1b/c/e/f/g/h on MNIST regression): 20–32 % MSE reduction over v15 at moderate label budgets, scaling correctly with `n_labeled` and surviving matched-step + held-out-test controls.
* **Where it fails** (exp1f-tdc, exp1h-tdc on TDC Lipophilicity_AstraZeneca + Mordred under scaffold split): degrades the held-out-test number at every n ≥ 500. Reversal augmentation specifically is image-domain-only — drop it on tabular features. The unlabelled-pool overfitting is fundamental: SSL on a pool that doesn't share the test distribution makes things worse.

For the molecular setting, the open follow-ups in the table below replace this recipe with something more conservative: drop reversal augmentation, use scaffold-test inputs as the unlabelled pool, or distribution-match the pool to the labelled set.

---

## Falsifications and decisions, in one line each

| # | Tested | Outcome |
|---|---|---|
| **exp0** | 2024 PDF's "transductive" loss does anything | ❌ Falsified — gradient through `f(q)` is bit-exactly zero. The 18 % win came from reversal augmentation. |
| **exp1** | Symmetry-breaking with isotropic noise rescues the unlabelled gradient | ✅ Yes — but no measurable val-MSE gain at 5 000 labels. |
| **exp1b** | Variance-scaled noise + low labels (MNIST) | ✅ ~10 % MSE reduction over v16 at n ∈ {100, 250, 1 000}. Small "minimum" σ doesn't deliver the benefit even though it breaks symmetry. |
| **exp1c** | The relational wrapping in v17 adds anything beyond plain Π-model + reversal-aug (v18) | ❌ Cosmetic. v17 ≡ v18 within seed noise; at n = 250 v18 is even slightly better. |
| **exp1e** | Matched-optimiser-step v15 (4× epochs, all on labels) catches up to v17/v18_split (MNIST) | ❌ Doesn't catch up. SSL beats it by 12–25 σ — the win is genuine label-data efficiency, not compute. |
| **exp1f** | Inductive vs transductive (val-set inputs as pool) on val MSE (MNIST) | Indistinguishable on val MSE. Possible artifact — exp1g's job. |
| **exp1g** | Held-out test eval for the same grid (MNIST) | SSL benefit confirmed unbiased (~21–32 % reduction). Transductive is a clean +5 % win at n=100 only; otherwise indistinguishable. **No tailoring artifact** under random val/test split. |
| **exp1h** | Extrapolating K beyond 3 keeps paying (MNIST) | Up to a budget-dependent optimum, then collapse. K=10 wins at n=250 (additional 14 %); K=0 wins at n=1 000. |
| **exp1f-tdc / 1h-tdc** | Port the recipe to TDC Lipophilicity_AstraZeneca + Mordred under scaffold split | ⚠️ **Does not transfer.** Same val-curve shape as MNIST, but on the held-out scaffold test SSL **degrades** by 0.3–0.6 MSE at n ≥ 1 000; transductive is the worst variant; reversal augmentation hurts. The MNIST recipe was specific to the in-distribution-eval regime. |
| **exp1i-tdc** | Drop reversal-augmentation + use scaffold-test inputs as the unlabelled pool (true transductive), evaluate on a disjoint held-out scaffold-test slice. | **Partial recovery.** Pure Π-model (no reversal-aug) beats v15 by 5–8 % on held-out test at n ∈ {100, 250}; ties at n=500; loses at n ≥ 1 000. True-transductive pool helps specifically at n=1 000 (−0.06 vs v15 instead of −0.17 with inductive pool). Reversal-augmentation was a substantial part of the exp1f-tdc failure. The high-n collapse is fundamental, not a recipe artefact. |
| **exp2** | docx's adaptive σ(q) closes the gap between fixed-σ Π-model and oracle σ*(q) | ⚠️ Inconclusive — the toy lacks `oracle < fixed-σ` headroom; needs redesign before GPU spend. |

---

## What this means for label-expensive domains (e.g. molecular property prediction)

The MNIST findings would have suggested a clean recipe to ship. The TDC port (exp1f-tdc, exp1h-tdc) walked that back. The exp1i-tdc ablations recovered part of it. The honest synthesis:

1. **The recipe works under in-distribution evaluation.** On both MNIST regression and TDC val MSE (in-domain split), v18 + variance-scaled noise + tuned K matches or beats the supervised baseline by ~10–30 % across the SSL sweet spot.

2. **Reversal-augmentation was a confound** (exp1i-tdc). Once dropped, **pure Π-model (`MSE(p, y) + MSE(z_a, z_b)`) does help on TDC scaffold test** — by 5–8 % at n ∈ {100, 250}, the realistic low-label molecular regime. It's a smaller win than on MNIST but real. The MNIST `v16` reversal trick was image-regression-specific.

3. **Using scaffold-test inputs as the unlabelled pool** (true transductive) gives an extra bump only at n = 1 000 — recovering ~0.11 MSE that the inductive pool loses at that budget. Below n = 500 the pool choice ties; above n = 1 000 nothing helps.

4. **The high-n SSL collapse is fundamental.** At n ≥ 1 000 every SSL variant on Mordred (with σ = 0.5 standardised noise) loses to the supervised baseline, and the gap *widens* with more labels. Two plausible reasons: (a) once the supervised signal is rich enough to learn nuanced 870-dim structure, σ = 0.5 perturbation actively destroys that nuance; (b) the consistency loss is enforcing smoothness toward an in-distribution unlabelled pool, which under scaffold split actively contradicts the test distribution.

5. **Implication for the molecular setting.** SSL on Mordred + MLP is worth trying *only* at the lowest label budgets and *only* without reversal-augmentation. Above n ≈ 500–1 000 the supervised baseline is the conservative choice. The rule from MNIST — "always pay the extra forwards" — is not safe in this domain.

What still holds even on TDC:

* **Best-epoch early-stopping is essential.** K-collapse and val-set drift are visible in both substrates.
* **The cancellation theorem (exp0) is substrate-agnostic** — it's algebraic. Any future "transductive" loss on Mordred + NN can be sanity-checked the same way.
* **Compute is genuinely cheap relative to labels.** exp1e showed v15_4x can't catch v17/v18_split on MNIST. We didn't re-run exp1e on TDC, but the underlying fact (extra optimiser steps on consistency are non-trivially different from extra steps on supervised) still applies — it's just that on TDC under scaffold split, those consistency steps push the model in the *wrong* direction.

Where to take this next, in the molecular setting (after exp1i-tdc closed the first two):

* ~~Drop reversal augmentation~~ ✅ done — recovers a 5–8 % win at low n on TDC.
* ~~True transductive~~ ✅ done — helps at n = 1 000, ties elsewhere.
* **σ × n_labeled joint sweep.** The high-n collapse may be specific to σ = 0.5; smaller σ might let the recipe scale further. Cheap to run.
* **Distribution-matched unlabelled pool.** For each labelled molecule, find its k nearest neighbours in scaffold space, use those as the consistency pool. This makes the SSL pool track wherever the labelled set lives — the most principled fix for the high-n collapse if it really is "consistency on the wrong distribution."
* **Re-run on a different ADMET task** (PPBR_AZ, VDss_Lombardo, Caco2). Whether the n=100/250 SSL benefit is task-general or Lipophilicity-specific.

Things that surprised us:
* **The PDF's transductive loss was algebraically vacuous** — gradient through `f(q)` is bit-exactly zero. A future SSL loss can be checked the same way: replace unlabelled forwards with zeros and diff the gradients.
* **Variance scaling is a free semantic-preserving win at large σ but provides no rescue at small σ.** The intuition that "minimum perturbation that breaks symmetry should be enough" is empirically false: the consistency residual `~σ² · E[per_dim_std²]` must be large enough to bias the optimiser, regardless of whether the symmetry-breaking gradient flows.
* **K has a U-shaped optimum and a collapse mode.** Without best-epoch early stopping, K=100 looks catastrophic; with it, K=10 is the right answer at n=250.

---

## What's still open

| Track | Question | Cost to answer |
|---|---|---|
| ~~TDC ablations~~ | ~~Drop reversal-aug from v18 — does the pure Π-model recover on scaffold test? Switch unlabelled pool to scaffold-test inputs (true transductive) — does that match the supervised baseline?~~ ✅ done in [exp1i-tdc](EXPERIMENTS.md#experiment-1i-tdc--drop-reversal-augmentation--true-transductive-on-tdc) — partial recovery at low n only. |
| **σ × n_labeled joint sweep on TDC** | Does smaller σ let the recipe scale past n = 500 without collapsing? | ~15 min CPU |
| **Distribution-matched pool** | Pick unlabelled molecules whose scaffold neighbours overlap with the labelled set; test whether SSL then preserves at high n. | ~30 min CPU |
| **Pool distribution** (MNIST) | Does the unlabelled pool's distribution matter, or is consistency just generic regularisation? Compare v18 with real MNIST vs random Gaussian vs pixel-shuffled MNIST as the unlabelled pool. | ~5 min CPU. Cleanest single test of MNIST SSL utility. |
| **Other TDC tasks** | Is the v15-beats-v18 result specific to Lipophilicity_AZ or general across ADMET regression? Try PPBR_AZ (~1 600), VDss_Lombardo (~1 130), Caco2 (~900). | ~10 min CPU per task once Mordred is cached |
| **Toy redesign for ACR** | exp2's 1D `sin / 5·sin` toy doesn't have positive `oracle < fixed-σ` headroom, so adaptive σ(q) cannot be discriminated. Either dense labels with input-correlated noise, or piecewise Lipschitz with n_labeled ≥ 200. | ~30 min CPU to design and run a candidate. |
| **σ × K joint sweep** | Smaller σ probably allows larger K before drift, and vice versa. Map the Pareto front. | ~30 min CPU at one budget. |
| **Phase 1.5 (σ_jac dynamics)** | Track `‖∇f‖` distribution during training, compare the 4 mitigations from docx §4.5. Only worth doing on the redesigned toy. | ~1 hr CPU after toy is ready. |
| **Tier-1 cancellation note** | exp0 alone is publishable as a methodological note. Just needs writing-up. | Half a day of writing. |

---

## Reproducibility

```
uv sync                                              # one-time, includes pyTDC + Mordred + RDKit
uv run experiments/exp0_cancellation.py              # cancellation theorem
uv run experiments/exp1_break_cancellation.py        # symmetry-breaking
uv run experiments/exp1b_low_labels.py               # variance-scaled noise + low labels
uv run experiments/exp1c_pi_model_ab.py              # v17 vs v18 paired A/B
uv run experiments/exp1e_matched_steps.py            # matched-step fairness
uv run experiments/exp1f_learning_curve.py           # learning curve + transductive (MNIST)
uv run experiments/exp1g_held_out_test.py            # exp1f on held-out test (MNIST)
uv run experiments/exp1h_consistency_scaling.py      # K sweep (MNIST)
uv run experiments/exp1f_tdc_learning_curve.py       # learning curve + held-out scaffold test (TDC)
uv run experiments/exp1h_tdc_consistency_scaling.py  # K sweep (TDC)
uv run experiments/exp1i_tdc_ablations.py            # drop reversal-aug + true-transductive (TDC)
uv run experiments/exp2_synthetic_acr.py             # adaptive σ on 1D toy
```

Total wall-clock ~150 min on a single CPU; the first TDC run computes Mordred descriptors for 4 200 molecules (~3 min, cached to disk). Every JSON in `results/` is regenerated from scratch on each run; matplotlib figures are deterministic given the JSON. Seeds are fixed.

If `uv sync` errors with TLS issues on macOS, prepend `UV_NATIVE_TLS=1`.
