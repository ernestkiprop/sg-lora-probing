# Design rationale

Concise version of the reasoning. See parent paper for the full theoretical
context (Jensen theorem, capacity-controller account, random-control test).

## Why probing

The parent paper rules out **gradient-magnitude** one-shot layer criteria
(SNIP, GN, Fisher) as placement oracles at layer granularity. Two related
follow-ups also failed:

| Method                          | Status      | What it broke         |
|---------------------------------|-------------|-----------------------|
| Activation patching (PatchMean) | Ran, failed | gradient-magnitude    |
| Cross-batch gradient agreement  | Ran, failed | gradient-magnitude    |

Both are still gradient-derived. Probing is **fundamentally non-gradient**:
it asks "how much task-relevant information is present in the layer
representation?", not "what is the magnitude of the gradient at this layer
at initialization?". A probe score is invariant to parameter count per layer
and depends only on what the frozen pre-trained representation can already
linearly decode.

## Three selection rules

It is not obvious which probe-derived rule should win. Three defensible
mechanisms:

- **Low-probe layers**: features are not yet linearly separable. LoRA's job
  is to *build* them. This is the "LoRA-as-feature-engineering" reading.
- **Mid-probe layers**: features are forming but not crystallized. LoRA can
  *refine* them with small updates. This is the "max marginal gain" reading.
- **High-probe layers**: features are present near the prediction surface.
  Small LoRA updates near the output have maximum *prediction leverage*.

All three are tested. The result that any rule beats Random would be
informative; the result that no rule beats Random would be informative in
a different direction (no-go theorem extends to information-theoretic
signals).

## Granularity choice

Per-layer probe accuracy gives one score per encoder block (12 layers).
The parent paper uses 72 candidate modules (12 layers × 6 modules each).
We map per-layer scores to module selection by including all 6 modules of
the selected layer. With `top_k_percent=0.20`:

- 2 of 12 layers selected → 12 modules → 16.7% of 72
- Compare to parent paper at k=20%: 14 modules → 19.4% of 72
- ~15% capacity mismatch (probing has slightly less)

The mismatch is acceptable because the **direction** of the mismatch is
against probing: if probing wins despite a small capacity disadvantage,
the placement signal is robust. If it ties Random, neither granularity
nor capacity were the issue.

## Probe implementation choices

- **Pooling**: CLS token at each layer. Matches RoBERTa's classification
  head convention. Mean pooling would be an alternative but introduces
  attention-mask dependence and risks contaminating later layers' scores
  with patterns from earlier ones.
- **Classifier**: `LogisticRegression` (lbfgs, max_iter=1000, C=1.0) for
  classification tasks; `Ridge(alpha=1.0)` for STS-B regression. Both have
  closed-form or near-closed-form solutions and have no hyperparameter
  sensitivity at the data scale used here (~770 train examples for SST-2,
  ~125 for RTE).
- **Eval**: Hold out the full GLUE validation split for probe scoring.
  Avoids the cross-validation bookkeeping of fitting on the 5% subset and
  evaluating within it. Probe scores are deterministic given the seed and
  the input split.

## What the parent paper's checklist requires

Any positive claim from this experiment must clear:

1. Beats Random at every k where it claims to win.
2. Beats LoRA-AllAttn at matched parameter count on ≥3 of 5 tasks.
3. Holm–Bonferroni survival with ≥5 seeds and Cohen's d_z.
4. Best-k reported alongside the headline k cell.
5. Validates on RoBERTa-large minimum.

This pilot covers (1) and (3) at k=20%; (4) and (5) are follow-ups gated on
positive (1)/(2)/(3) results.
