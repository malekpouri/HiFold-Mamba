# Internal review, round 1 (critic role), 2026-10-23

Reviewed: `paper/manuscript.md` (Stage 1 draft v1). Standard: a Stage 1 Registered Report at a Q1 venue. Would a reviewer accept the hypotheses, methods and analysis as specified enough to run without further decisions, and are the claims correctly bounded?

## Major

M1. **The H2 analysis is not executable as written.**
- "Block bootstrap applied to per-block AUROC contributions … recomputing AUROC on the resampled bin set" is ambiguous. It leaves open: per line or pooled; seed aggregation; and how a bin covered by several overlapping test windows is scored once.
- The same applies to loop pairs that fall in more than one window.
- There is no code for it.
- **Required:** an exact definition (deduplication across windows, seed aggregation, per-line averaging), an implementation and a test.

M2. **H2 checkpoint confound.** Checkpoints are selected by validation SCC, evaluated every 7,008 steps. The Struct arm could select a checkpoint from before step 52,560, i.e. one identical to Base, which makes H2 null by construction. Which arm is "HiFold-Mamba" for H1 and H3 is also undefined.

M3. **Power analysis ignores the per-block maximum over three baselines.** The maximum of three noisy baseline scores is biased upwards, so the simulated power (one baseline) is optimistic. Simulate with |ℬ| = 3.

M4. **Error control across hypotheses is not discussed.** H1, H2, H3 and N-M are each tested at α = 0.025 and each can change the title. The rationale (separate claims, separate conclusions) or a correction must be stated.

M5. **H3 may be trivially satisfied.** With 3,504 fine-tuning steps, a randomly initialized model is expected to lose to a pretrained one. A reference showing that adaptation does more than the pretrained model already does zero-shot is missing.

M6. **Comparability of the locus-shared vs. published Akita comparison rests on an unstated assumption:** that DNALongBench's published SCC was computed with its released `scc` implementation (which we use verbatim). Say so.

## Minor

m1. **Intro overgeneralizes.** "Sequence-to-map models are trained with pixel-wise losses" is verified only for Akita and the DNALongBench baselines. Restrict the statement.

m2. **Hi-Compass description is imprecise.** It uses sequence and a generalized CTCF profile, with accessibility as the only *cell-type-specific* input.

m3. **Block dependence.** Windows overlap across 1 Mb block borders. Add a sensitivity analysis with 3 Mb blocks.

m4. **The 0.01 margin is on the HiCRep scale** (reference ≈ 0.53), where it is relatively smaller than on the benchmark scale (≈ 0.23). Justify it as the smallest effect of interest and state the scale.

m5. **Akita reproduction criterion is undefined.** When is it "reproduced"?

m6. **Struct-arm λ selection uses the same boundary datasets** as the H2 evaluation (validation regions only). Acknowledge this.

m7. **Software environment** (versions, hardware) is missing from the reproducibility statement.

m8. **Duplicate figure labels** ("Figure 1: Fig. 1 |").

m9. **Boundary labels.** State that both Strong and Weak 4DN calls are positives.

## Deviations from the internal protocol that the revision introduces (user decisions to confirm)

D1. Caduceus/HyenaDNA are moved out of ℬ (published context only; compute).
D2. The loop loss is dropped from v1 (its positives would share a source with the loop evaluation).
D3. H2 compares final checkpoints; H1 and H3 use the Base arm (fix for M2).
D4. No cross-hypothesis multiplicity correction, with rationale (M4).
D5. Zero-shot reference added to H3 (secondary).
