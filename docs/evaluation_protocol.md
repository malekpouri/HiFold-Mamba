# Evaluation Protocol — v1 (megabase-scale)

Status: **locked.** G3, G4, G5 and G7 resolved by the user on 2026-10-21. G6, G8, G9 and G10 resolved on 2026-10-21 by Claude under the user's delegation (see `Roadmap.md`). G1 and G2 are decided provisionally from the papers and close when the DNALongBench files confirm them; until then they block the loss code.

Sources (checked against the publisher, PubMed/PMC or Crossref on 2026-10-21):
- HiCRep / SCC: Yang et al. 2017, *Genome Res* 27:1939–1949, doi:10.1101/gr.220640.117, PMC5668950
- DNALongBench: Cheng et al. 2025, *Nat Commun* 16, doi:10.1038/s41467-025-65077-4. Preprint: doi:10.1101/2025.01.06.631595, PMC11741265 (the task details below were read from the preprint)
- Akita: Fudenberg, Kelley & Pollard 2020, *Nat Methods* 17:1111–1117, doi:10.1038/s41592-020-0958-x, PMC8211359
- Caduceus: Schiff et al. 2024, ICML, PMLR 235:43632–43648, PMID 40567809, PMC12189541; code: https://github.com/kuleshov-group/caduceus
- ChiMamba: Zhang et al. 2024, *Advanced Intelligent Computing in Bioinformatics* (ICIC 2024), LNCS 14882, pp. 50–61, doi:10.1007/978-981-97-5692-6_5. Full text not yet read (paywalled)
- Removed as a source: arXiv 2606.04552 (LDARNet, learnable DNA tokenization; unrelated to bf16 stability).

Verified facts from these sources: the DNALongBench contact-map task has **9 cell lines** (Akita's five + HAP1, HeLa, HepG2, K562), 7,008/419/413 train/validation/test sequences, SCC and PCC, and an upper-triangle target with diagonal offset 2 (99,681 values). The split is random non-overlapping virtual contigs, 8:1:1. Akita's target: ICE → adaptive coarse-graining → distance normalization → natural log → clip (−2, 2) → linear interpolation of missing bins → 2D Gaussian smoothing (σ = 1). Augmentation: shifts of ±11 bp and RC.

## 1. Benchmark and splits

| Item | Decision |
|---|---|
| Primary benchmark | **DNALongBench contact-map task**: 1,048,576 bp, 2,048 bp bins, 448×448 |
| Primary split | **Official DNALongBench train/validation/test files** (region-based) |
| Robustness split | Chromosome held-out (test chr1, chr8, chr12; validation chr3, chr6). All baselines retrained. Reported separately |
| Cell lines (v1) | **Training: HFF, H1-hESC, GM12878, IMR-90, HCT116 only.** H3 held-out only: HAP1, HeLa, HepG2, K562 (the other 4 DNALongBench lines) |
| Pretraining data | **Only the official DNALongBench training split of the 5 training lines. No extra data in v1** |
| Baseline reproduction | Compared **per cell line** with published values for the 5 training lines. DNALongBench's 9-line averages are not comparable with our 5-line numbers |
| Forbidden | Random window-level splits |

## 2. Metrics

**SCC = stratum-adjusted correlation coefficient** (HiCRep): Pearson correlation per distance stratum, aggregated with stratum weights, computed per window. PCC is always reported. Spearman correlation is supplementary and labelled as such.

**Primary SCC (decided 2026-10-23):** the HiCRep weighting (Yang et al. 2017) is primary for H1–H3, checkpoint selection and the bootstrap. The DNALongBench `scripts/metrics.py::scc` variant (≈ uniform per-stratum weights; no smoothing) is reported alongside it as a secondary metric, for comparability with published DNALongBench numbers. The two rank windows differently (Spearman ρ = 0.54 on the locus-shared reference), so both are shown in every main table.

| Level | Primary | Supplementary |
|---|---|---|
| Contact map | SCC | PCC, Spearman correlation, MSE, MAE |
| TAD structure | Boundary AUROC on **independent** boundaries (§5 H2) | Insulation correlation, boundary F1, Directionality Index, insulation-derived boundaries (internal consistency only) |
| Loops | Average precision | Precision/recall within 1–2 bins, AUROC, Precision@k, Recall@k |
| Distance strata | SCC at 10–100 kb, 100–500 kb, 0.5–0.92 Mb (max separation in the output = 447 × 2,048 bp ≈ 915 kb) | ΔSCC per stratum |
| Cell-type generalization (H3) | $\Delta_{\text{H3}}$ | Performance-vs-data plot |
| Variants (H4) | ΔSCC, Δinsulation, Δloop score | Expression correlation (exploratory) |
| Efficiency | Inference time, peak memory, parameters | Only for 4–8 Mb inputs or 128 bp tokens |

Reporting: mean ± SD over 3 seeds.

## 3. Statistics (primary test)

1. SCC per test window, for each model and seed.
2. **Average SCC over the 3 seeds** per model and window, then average per non-overlapping **1 Mb block**.
3. Block-wise difference against the **best baseline in each block** (more conservative; prevents post-hoc baseline choice):
   $$\Delta_b = \mathrm{SCC}_{\text{ours},b} - \max_{m\in\mathcal{B}} \mathrm{SCC}_{m,b}$$
   $\mathcal{B}$ = the trained sequence models only: Akita, Caduceus-Ph/PS, matched Transformer, strong CNN, and any optional sequence baseline that is run (§4). The global cell-line mean and the locus-shared references are **not** in $\mathcal{B}$ (G7).
4. **One-sided** test, $H_0: \Delta \le 0$ vs. $H_1: \Delta > 0$, at **α = 0.025**, using 10,000 paired block-bootstrap resamples (G5).
5. Report the mean ΔSCC, the 95% percentile block-bootstrap CI and the one-sided p-value. The CI's lower limit is the 97.5% one-sided bound, so "lower limit > 0" and "p < 0.025" are the same decision. Per-seed ΔSCC goes in the appendix.

**Success criterion (H1):** mean block-wise ΔSCC ≥ **+0.01** relative to the per-block best baseline, with the 97.5% one-sided bound > 0.

The same bootstrap and one-sided α = 0.025 apply to every other pre-specified comparison (H2, H3, the mixer ablation for N-M, and the global-mean check in §7), each against its own stated comparator.

Per-window Wilcoxon: secondary only.

## 4. Baselines

| Baseline | Definition | Status |
|---|---|---|
| Global cell-line mean | Mean of training maps of the same cell line | Kept; not in ℬ; minimum bar for the §7 stopping rule |
| Locus-shared mean | Mean map of the **other** cell lines at the same locus (target cell excluded) | Kept; oracle-style reference for cell-specificity, not in ℬ, never a stopping condition (G7) |
| Locus-shared mean + cell offset | Locus mean + learned cell-line offset | Added; same status as locus-shared mean |
| Akita | Reference sequence → contact-map model | Kept |
| Caduceus-Ph/PS | Genomic Mamba model | Kept |
| Matched Transformer | Same parameters and budget | **Mandatory** |
| Strong CNN | Large receptive field. **Defined 2026-10-23:** HiFold with the Bi-Mamba mixer swapped for a dilated residual 1D CNN (dilation $2^i$), parameter-matched and RC-equivariant (`configs/baseline_conv.yaml`) | Kept |
| HyenaDNA | Only if checkpoint and pipeline are compatible | Optional |
| DeepC | Only with full retraining in our format | Optional |
| ChiMamba | Baseline, or at least a conceptual comparison table | Depends on its task (G9) |
| Distance-only | — | **Removed** (SCC almost eliminates it) |

Akita and Caduceus are reproduced on the official split, which provides per-window predictions for the bootstrap.

## 5. Hypotheses

### H1 — SCC improvement
On the official DNALongBench split, the model beats the per-block best baseline by ΔSCC ≥ +0.01 (§3).

### H2 — Structure constraint
$\mathcal{L}_{\text{struct}}$ (insulation + boundary + loop terms; `architecture.md`) improves TAD/loop reconstruction compared with the identical model without it.
- **Same compute:** same total optimizer steps. In a two-stage schedule, the Base model gets the same extra steps. The only difference is $\mathcal{L}_{\text{struct}}$.
- **Success:** independent-boundary AUROC ≥ +0.02 and loop AP ≥ +0.02, both with 97.5% one-sided bounds above 0, and no meaningful SCC loss (97.5% lower bound of ΔSCC > −0.005). All three conditions must hold, so no multiplicity correction is needed (intersection–union).

To avoid circularity:
- **Training:** $\hat{I}$ is computed from the predicted map, $I^{\text{true}}$ from the true map (same operator and space, see G2).
- **Preliminary evaluation:** insulation-derived boundaries, only to check that the constraint works.
- **Main evaluation:** **independent** boundaries: CTCF-associated boundaries, or TAD annotations independent of the model and of the training Hi-C (see G8).
- **Loops:** HiCCUPS / CTCF–CTCF loops from an independent annotation source, or at least from a different fold than the training labels.
- Without independent boundaries, the boundary metric is reported only as internal consistency, not as the main evidence for H2.

### H3 — Few-shot cell-type adaptation

| Item | Design |
|---|---|
| Main model | The 5 training lines |
| H3 held-out lines | **HAP1, HeLa, HepG2, K562**: the DNALongBench lines never used in (pre)training |
| H3 pretraining | The main 5-line model (DNALongBench training split) |
| H3 fine-tuning | Limited data from one held-out line at a time; a new FiLM embedding is learned and all weights are fine-tuned |
| H3 scratch arm | Same architecture, random initialization, same fine-tuning data and the same number of optimizer steps |
| H3 evaluation | **Official DNALongBench test regions** of the held-out line, not chromosome held-out (which overlaps the official training split) |
| Lines tested | All 4 held-out lines |
| H3 target smoothing | The DNALongBench files for the 4 held-out lines are **unsmoothed** (`kernel_stddev 0`), unlike the 5 training lines. Used unchanged (decided 2026-10-23); stated as a limitation. Both H3 arms see the same targets, so $\Delta_{\text{H3}}$ is unaffected, but absolute SCC on these lines isn't comparable with the training lines |

Leave-one-out among the five training lines is an optional secondary analysis. If run, the hold-out is removed from pretraining and the 4-line model retrained.

Data regimes (G4, resolved):

| Regime | Fine-tuning data |
|---|---|
| Limited 10% / 25% / 50% | Contiguous-block subsets of the held-out line's official DNALongBench training regions (blocks, not random windows; same subsets for both arms) |
| Full | All official training regions of that line |
| Read-depth-limited | **Moved to v2** (needs reprocessing from read pairs, which conflicts with "no extra data in v1") |

$$\Delta_{\text{H3}} = \mathrm{SCC}_{\text{pretrain+few-shot}} - \mathrm{SCC}_{\text{scratch}}$$

Both arms get exactly the same compute and fine-tuning data. The FiLM embedding for the new cell line is learned during fine-tuning. $\Delta_{\text{H3}}$ is tested per held-out line and regime with the §3 block bootstrap, against the scratch arm.

**H3 success (used by N-F in §7):** in **at least 2 of the 4 held-out lines**, $\Delta_{\text{H3}} > 0$ with a 97.5% one-sided bound > 0 in at least one limited regime, Holm-corrected over the three limited regimes within each line. The full regime is reported but does not count towards N-F.

### H4 — Structural variants
Only public SVs with Hi-C before and after the variant. **If there are fewer than 10 valid cases, H4 is limited to H4-sim**: explicit in-silico labels and a limited claim.

## 6. Ablations, interpretability, case study

- Ablations: Bi-Mamba → matched Transformer, RC equivariance, $\mathcal{L}_{\text{struct}}$, multi-scale mixer.
- Interpretability: CTCF motifs, CTCF orientation, TAD-boundary regions.
- Case study: a known deletion/inversion vs. existing Hi-C.

## 7. Decision rules and titles

| Trigger | Action |
|---|---|
| SCC below Akita/Caduceus after 2–3 months | Check data, normalization and split first; then pivot the claim to whichever narrative below can still hold |
| $\mathcal{L}_{\text{struct}}$ gives no significant benefit (H2 not met) | No "structure/insulation-constrained" claim, including in the title |
| Doesn't beat the **global cell-line mean** (§3 bootstrap against that baseline alone; 97.5% bound > 0) | The project fails in its current form. The locus-shared references are oracle-style context only and never a stopping condition (G7) |
| Not enough variant data | Drop the mutation claim (H4-sim at most) |
| Too much compute | Shorter pretraining, frozen encoder, less depth. Never a random split |
| **None** of N-I, N-M, N-F holds | Hard stop (below) |

**Narratives and their pre-specified criteria.**

| ID | Narrative | Criterion |
|---|---|---|
| N-I | Insulation-constrained | H2 met (§5) |
| N-M | Multi-scale | H1 met **and** the mixer ablation is significant: ΔSCC(full − no mixer), §3 block bootstrap, 97.5% one-sided bound > 0 |
| N-F | Few-shot cell-type adaptation | H3 success (§5) |

**Titles**, fixed in advance:

| Narratives met | Title |
|---|---|
| N-I and N-M | *Insulation-Constrained Multi-Scale State-Space Modeling for Megabase-Scale Prediction of 3D Chromatin Folding* (working title) |
| N-I only | *Insulation-Constrained Sequence-to-Contact-Map Prediction at Megabase Scale* |
| N-M only | *Multi-Scale State-Space Modeling of Chromatin Contact Maps from DNA Sequence* |
| N-F only | *Few-Shot Cell-Type Adaptation for Sequence-Based Prediction of 3D Chromatin Folding* |

If N-F holds together with N-I or N-M, the title follows N-I/N-M and N-F is reported as a secondary result. No title contains "physics" or "energy". No title rests on "RC-equivariant Bi-Mamba" alone, because Caduceus has the same core.

**Hard stop.** If none of N-I, N-M or N-F holds, the project is **stopped and the idea redesigned**. It does not become a benchmark paper, a negative-result paper or an "RC-equivariant Bi-Mamba" paper.

## 8. Open items

**G1 — Target space. Provisionally resolved 2026-10-21; closes in Execution step 1.** Decision: train and evaluate in the DNALongBench target space, unchanged. Per the Akita paper this is natural-log O/E, clipped to (−2, 2), with interpolated missing bins and Gaussian smoothing, stored as the upper triangle with diagonal offset 2. Step 1 must confirm in the files: (i) value range and log base, (ii) whether any mask or NaN is shipped (if not, all 99,681 entries count as valid, since missing bins were interpolated upstream), (iii) the offset, (iv) the processing of the 4 DNALongBench-added lines. Original note:

**G1 (original) — Target space: contradiction with the benchmark.** The proposal "log1p on contact counts after normalization; all losses in this space" conflicts with:
- **The primary benchmark.** v1 trains and evaluates on the **DNALongBench files**, whose target is fixed by the benchmark (as discussed, a distance-normalized log O/E-like map `[verify against the DNALongBench data/code]`). Our predictions and SCC must be in the benchmark's own target space, or the numbers aren't comparable with the reproduced Akita/Caduceus runs.
- **F1.** The softplus step assumes $\hat{C}$ is in a log/O/E space that can be negative. With a log1p target, $\hat{C} \ge 0$ and no softplus is needed.
- **The distance issue (earlier D1).** log1p of balanced counts is dominated by the distance decay.

**Proposal:** the training target = the DNALongBench target, unchanged. log1p(balanced counts) is not used in v1. First step in Phase 0: inspect the DNALongBench files and document the exact target transform.

**G2 — Insulation transform. Provisionally resolved 2026-10-21:** $g(x) = \exp\!\big(2\tanh(x/2)\big)$, applied identically to $\hat{C}$ and $C^{\text{target}}$. It is smooth, bounded to $[e^{-2}, e^{2}]$ (the clip range of the target), and ≈ exp near 0. Because the same $g$ is used on both sides, its compression near ±2 introduces no scale bias between prediction and target. The same $d_{\min} = 2$ applies, which $a < i < b$ already enforces. Closes with G1. Original note:

**G2 (original) — Insulation: predicted and true maps in different spaces.** $\hat{I}$ is computed from $\operatorname{softplus}(\hat{C})$, and $I^{\text{true}}$ from $C^{\text{raw}}$. These are different scales:
- $C^{\text{raw}}$ (balanced counts) contains the full distance decay. $\operatorname{softplus}$ of a log-O/E value doesn't.
- softplus is not the inverse of log, so $\operatorname{softplus}(\log x) \ne x$.

$\mathcal{L}_{\text{ins}}$ would therefore force the map towards a systematically wrong scale. Also, v1 has only the DNALongBench files, so there is no $C^{\text{raw}}$ unless coolers are reprocessed.

**Fix:** apply **the same transform $g$ to both maps**: $\tilde{C} = g(\hat{C})$ and $\tilde{C}^{\text{true}} = g(C^{\text{target}})$, with the same operator, $w$ and mask.
- With $g = \exp$: if the target is log(O/E), $\exp$ is the exact inverse and gives O/E-scale insulation. This is safe because the target is clipped `[verify the range]`.
- With $g = \operatorname{softplus}$: a smooth approximation, but then on both sides.

Recommendation: $g = \exp$ if the target is natural-log O/E (else use the matching base), with a clamp on $\hat{C}$ for numerical safety. Additional notes:
- With $w = 40$ (changed from 20 on 2026-10-23 after the validation positive control, manuscript §3.4), insulation is defined only for bins $i \in [40, 408)$ of the 448-bin output.
- Masked bins are excluded from the sum, and the $1/w^2$ normalization becomes "divide by the number of valid pairs".
- Pairs inside the benchmark's excluded near-diagonal band ($b - a < d_{\min}$) are also excluded, with $d_{\min}$ taken from the DNALongBench files/code when G1 is documented.

**G3 — $\lambda_{\text{sym}}$ is redundant. Resolved 2026-10-21: dropped.** $\hat{C} = \mathrm{symmetrize}(\cdot)$ makes $\lVert\hat{C}-\hat{C}^\top\rVert_1 \equiv 0$. Symmetry is architectural. Equivalently, it can be listed as a structural prior, but not as a loss.

**G4 — H3 data modes. Resolved 2026-10-21:** contiguous-block subsets 10/25/50% + full; read-depth-limited moved to v2 (§5 H3). Original note:
- "Region-limited = official training regions of that line" and "Full = all training regions" are the same thing. Region-limited must be a **subset**, e.g. 10%, 25% and 50% of the training regions. Sample contiguous blocks rather than random windows, to avoid neighbouring windows giving away information.
- Read-limited needs the read pairs or contact counts of the held-out line, reprocessed with the **exact DNALongBench pipeline**. That is extra data processing in v1, which conflicts with "no extra data in v1". Either accept it as an exception (validated by reproducing the released targets at 100% reads), or move read-limited to v2.

**G5 — Significance level. Resolved 2026-10-21: one-sided α = 0.025, reported with the 95% percentile CI (§3).** "If two-sided is requested, report with α = 0.025" has it the wrong way round. A one-sided test at α = 0.05 corresponds to a two-sided test at α = 0.10. A one-sided test equivalent to two-sided α = 0.05 uses **one-sided α = 0.025**. Recommendation: **pre-register one-sided α = 0.025** from the start (equivalently, a 97.5% one-sided CI bound > 0). That needs no later adjustment.

**G6 — Equivariance unit test conditions. Resolved 2026-10-21:** fp32 exact test (atol 1e-4, rtol 1e-5); bf16 check (rtol 1e-2); eval mode, plus train mode with dropout 0 or identical RNG per call; with and without gradient checkpointing.
- With bf16, `atol=1e-4` will fail through rounding alone. Run the exact test in **fp32**, and add a looser bf16 check (e.g. relative tolerance ~1e-2).
- **Train mode:** dropout (and stochastic layers) give different masks for $x$ and $\mathrm{RC}(x)$, so equality can't hold. Run the train-mode test with dropout disabled or the RNG seeded identically per call, or test in eval mode only plus a deterministic train-mode variant.
- The window crop (32 bins per side) is symmetric, so the 180° rotation stays valid after cropping.

**G7 — Locus-shared mean is an oracle-style reference. Resolved 2026-10-21:** kept out of ℬ, reported as context only, never a stopping condition. The §7 stopping bar is the global cell-line mean. It uses the **measured** test-region maps of the other cell types. That makes it a strong, legitimate reference for cell-specificity, but not a sequence-only predictor. Keep it **out of $\mathcal{B}$** in the H1 max (otherwise H1 compares a sequence model with measured data) and report it separately. The same applies to "locus-shared mean + cell offset".

**G8 — Independent boundaries: which source?** Published TAD annotations for these cell lines are usually derived from the **same** Hi-C/Micro-C datasets (often the same data behind the Akita/DNALongBench targets), so they aren't independent. Candidates:
- CTCF ChIP-seq (ENCODE)-bound sites with convergent orientation at insulation minima
- boundaries from a **different replicate or study** of the same cell line (e.g. 4DN)

**Resolved 2026-10-21:**
- **Primary:** boundaries from an **independent** Hi-C/Micro-C dataset of the same cell line (a different study or biological replicate from the one behind the DNALongBench target), called with a caller and parameters fixed before evaluation.
- **Secondary:** ENCODE CTCF ChIP-seq peaks containing a CTCF motif.
- **Lines without an independent contact dataset:** only the secondary set is reported, and this is stated per line.
- **Loops:** taken from an independent dataset. Loop calls derived from the training data (e.g. Rao et al. 2014 for GM12878/IMR-90) are excluded.
- **Accessions:** chosen in Execution step 1 and recorded in `docs/data_manifest.md` before any evaluation.
- **Pre-registered 2026-10-23:** loop positives = CTCF ChIA-PET clusters with PET count ≥ 10 (ENCODE; none for HFF). Boundaries = the shipped 4DN boundary calls (insulation pipeline), accepted by the user. Their method family matches our insulation operator; this is stated as a limitation.
- **Source table approved 2026-10-23** (`notes/2026-10-23_g8-candidates.md`): one boundary and one loop source per line, from a different lab wherever 4DN has one. IMR-90 has no independent loop dataset on 4DN, so it gets the secondary CTCF set only for loops.

**G9 — Citation links don't match their sources. Resolved 2026-10-21:** all sources are re-checked and corrected (see Sources above). ChiMamba is positioned as related work (sequence + accessibility, per its abstract) unless the full chapter shows a comparable sequence-only setting. Original note: In your list, the ChiMamba claim links to the Caduceus GitHub repository, and the arXiv-2606.04552 note links to a Springer book. The Springer book (978-981-97-5692-6) is presumably the ChiMamba source. In the reference manager, attach each claim to its own source. ChiMamba details (input, output format, resolution, window, split) are still to be extracted.

**G10 — Compute fallback changes the model. Resolved 2026-10-21:** the paper configuration is the 24–40 GB model. The 8 GB fallback is for development only and is never reported. Original note: The fallback for 8 GB (4 Bi-Mamba layers, hidden 128) is a different model from the 24–40 GB configuration. Decide **before** training which configuration is the paper model. Baselines (especially the matched Transformer) must be matched to that configuration. Don't mix results from the two configurations.
