# Internal review, round 2 (critic role), 2026-10-23

Reviewed: the manuscript after revision 1, read as rendered PDF.

## Major

R2-1. **The SCC definition in §2.7 is wrong.**
- The text says correlations are computed on ranks. In HiCRep, and in our code, Pearson is computed on raw values; ranks enter only the weights.
- The weighting contrast is also misdescribed. The benchmark weights strata ∝ 1/N_d (long-range dominate). HiCRep weights ∝ N_d (short-range dominate). They point in opposite directions, which explains ρ = 0.54. The text should say so.

R2-2. **Unverified generalization.** The abstract and contribution 3 call the per-block-best comparison "widely used" / "common". It was *our own* original plan; there is no source for "widely used". Rephrase.

R2-3. **Missing outcome-neutral checks and exclusion rules.** These are expected in a Registered Report. State in advance:
- the conditions under which the data or the pipeline count as failed;
- what happens to NaN SCC windows;
- the positive controls (Akita reproduction, equivariance, beating the global mean).

R2-4. **H3 subsets are not reproducible as written** ("selected by contig"). The rule (seeded permutation, nested subsets) and the H3 test unit are missing.

## Minor

R2-5. **Overclaim.** "A map with wrong domains cannot have correct insulation" is false; different maps can share an insulation profile. Restate as: the loss is a function of the map itself.

R2-6. **"Equal compute"** for Struct vs. Base: only optimizer steps and data are equal. The structural terms add a little compute.

R2-7. **The λ pilot's starting point is not stated** (fork of Base seed 0 at step 52,560).

R2-8. **"Holds exactly"** for equivariance: exact by construction, verified numerically within tolerance.
