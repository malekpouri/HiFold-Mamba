# Evaluation Protocol — v1 (megabase-scale)

Status: **locked by the user, except the open items in §8 (G1–G10).** G1 and G2 block coding.

Sources:
- HiCRep / SCC: Yang et al. 2017, *Genome Research* — https://genome.cshlp.org/content/early/2017/08/30/gr.220640.117.full.pdf
- DNALongBench: https://pmc.ncbi.nlm.nih.gov/articles/PMC11741265/
- Akita: Fudenberg et al. 2020 — https://pmc.ncbi.nlm.nih.gov/articles/PMC8211359/
- Caduceus: *Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling*, ICML 2024 — https://pubmed.ncbi.nlm.nih.gov/40567809/; code: https://github.com/kuleshov-group/caduceus (confirmed by the user)
- ChiMamba: *ChiMamba: Predicting Chromatin Interactions Based on Mamba*, chapter in a Springer book — presumably https://link.springer.com/book/10.1007/978-981-97-5692-6 (see G9). Input/output/split details still to extract from the full text.
- Removed as a source: arXiv 2606.04552 (it is about tokenization / BiMamba-2, not bf16 stability).

## 1. Benchmark and splits

| Item | Decision |
|---|---|
| Primary benchmark | **DNALongBench contact-map task**: 1,048,576 bp, 2,048 bp bins, 448×448 |
| Primary split | **Official DNALongBench train/validation/test files** (region-based) |
| Robustness split | Chromosome held-out (test chr1, chr8, chr12; validation chr3, chr6). All baselines retrained. Reported separately |
| Cell lines (v1) | **HFF, H1-hESC, GM12878, IMR-90, HCT116 only** |
| Pretraining data | **Only the official DNALongBench training split. No extra data in v1** |
| Forbidden | Random window-level splits |

## 2. Metrics

**SCC = stratum-adjusted correlation coefficient** (HiCRep): Pearson correlation per distance stratum, aggregated with stratum weights, computed per window. PCC is always reported. Spearman correlation is supplementary and labelled as such.

| Level | Primary | Supplementary |
|---|---|---|
| Contact map | SCC | PCC, Spearman correlation, MSE, MAE |
| TAD structure | Boundary AUROC on **independent** boundaries (§5 H2) | Insulation correlation, boundary F1, Directionality Index, insulation-derived boundaries (internal consistency only) |
| Loops | Average precision | Precision/recall within 1–2 bins, AUROC, Precision@k, Recall@k |
| Distance strata | SCC at 10–100 kb, 100–500 kb, 0.5–~0.9 Mb | ΔSCC per stratum |
| Cell-type generalization (H3) | $\Delta_{\text{H3}}$ | Performance-vs-data plot |
| Variants (H4) | ΔSCC, Δinsulation, Δloop score | Expression correlation (exploratory) |
| Efficiency | Inference time, peak memory, parameters | Only for 4–8 Mb inputs or 128 bp tokens |

Reporting: mean ± SD over 3 seeds.

## 3. Statistics (primary test)

1. SCC per test window, for each model and seed.
2. **Average SCC over the 3 seeds** per model and window, then average per non-overlapping **1 Mb block**.
3. Block-wise difference against the **best baseline in each block** (more conservative; prevents post-hoc baseline choice):
   $$\Delta_b = \mathrm{SCC}_{\text{ours},b} - \max_{m\in\mathcal{B}} \mathrm{SCC}_{m,b}$$
   $\mathcal{B}$ = the trained sequence models only (see G7).
4. **One-sided** test, $H_0: \Delta \le 0$ vs. $H_1: \Delta > 0$, using 10,000 paired block-bootstrap resamples.
5. Report the mean ΔSCC, 95% block-bootstrap CI and p-value. Per-seed ΔSCC goes in the appendix.

**Success criterion (H1):** mean block-wise ΔSCC ≥ **+0.01** relative to the per-block best baseline, with a positive one-sided CI. For the significance level, see G5.

Per-window Wilcoxon: secondary only.

## 4. Baselines

| Baseline | Definition | Status |
|---|---|---|
| Global cell-type mean | Mean of training maps of the same cell type | Kept |
| Locus-shared mean | Mean map of the **other** cell types at the same locus (target cell excluded) | Kept; measures cell-specificity (see G7) |
| Locus-shared mean + cell offset | Locus mean + learned cell-type offset | Added |
| Akita | Reference sequence → contact-map model | Kept |
| Caduceus-Ph/PS | Genomic Mamba model | Kept |
| Matched Transformer | Same parameters and budget | **Mandatory** |
| Strong CNN | Large receptive field | Kept |
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
- **Success:** independent-boundary AUROC +0.02 and loop AP +0.02, both with CIs above 0, and no meaningful SCC loss (lower CI bound of ΔSCC > −0.005).

To avoid circularity:
- **Training:** $\hat{I}$ is computed from the predicted map, $I^{\text{true}}$ from the true map (same operator and space, see G2).
- **Preliminary evaluation:** insulation-derived boundaries, only to check that the constraint works.
- **Main evaluation:** **independent** boundaries: CTCF-associated boundaries, or TAD annotations independent of the model and of the training Hi-C (see G8).
- **Loops:** HiCCUPS / CTCF–CTCF loops from an independent annotation source, or at least from a different fold than the training labels.
- Without independent boundaries, the boundary metric is reported only as internal consistency, not as the main evidence for H2.

### H3 — Few-shot cell-type adaptation

| Item | Design |
|---|---|
| Main model | All 5 cell lines |
| H3 hold-out | One line (e.g. IMR-90), removed from pretraining |
| H3 pretraining | The remaining 4 lines (DNALongBench training split) |
| H3 fine-tuning | Limited data from the held-out line |
| H3 evaluation | **Official DNALongBench test regions** of the held-out line, not chromosome held-out (which overlaps the official training split) |
| Rotation | Ideally each of the 5 lines held out once; at least 2 |

Data modes (see G4):

| Mode | Fine-tuning data |
|---|---|
| Read-limited | 10%, 25%, 50% downsampled reads |
| Region-limited | Official DNALongBench training regions of that line *(to be redefined as a subset, G4)* |
| Full | All training regions of that line |

$$\Delta_{\text{H3}} = \mathrm{SCC}_{\text{pretrain+few-shot}} - \mathrm{SCC}_{\text{scratch}}$$

Both arms get exactly the same compute and fine-tuning data. The FiLM embedding for the new cell line is learned during fine-tuning.

### H4 — Structural variants
Only public SVs with Hi-C before and after the variant. **If there are fewer than 10 valid cases, H4 is limited to H4-sim**: explicit in-silico labels and a limited claim.

## 6. Ablations, interpretability, case study

- Ablations: Bi-Mamba → matched Transformer, RC equivariance, $\mathcal{L}_{\text{struct}}$, multi-scale mixer.
- Interpretability: CTCF motifs, CTCF orientation, TAD-boundary regions.
- Case study: a known deletion/inversion vs. existing Hi-C.

## 7. Decision rules and titles

| Trigger | Action |
|---|---|
| SCC below Akita/Caduceus after 2–3 months | Check data, normalization and split first; then pivot the claim |
| $\mathcal{L}_{\text{struct}}$ gives no significant benefit | No "structure/insulation-constrained" claim |
| Doesn't beat the mean-map baselines | The project fails in its current form |
| Not enough variant data | Drop the mutation claim (H4-sim at most) |
| Too much compute | Shorter pretraining, frozen encoder, less depth. Never a random split |

**Titles.** No "physics" and no "energy head". Use *structure-constrained* or *insulation-constrained*. A bare "RC-equivariant Bi-Mamba" title is excluded because Caduceus has the same core.

Narratives prepared in case the hard stop triggers:
1. Structure-aware: *Insulation-Constrained Sequence-to-Contact-Map Prediction at Megabase Scale*
2. Cell-type generalization: *Few-Shot Cell-Type Adaptation for Sequence-Based Prediction of 3D Chromatin Folding*
3. Multi-scale: *Multi-Scale State-Space Modeling of Chromatin Contact Maps from DNA Sequence*

If none of the three gives a significant benefit, the project is **stopped and the idea redesigned**. It does not become a benchmark or negative-result paper.

## 8. Open items

**G1 — Target space: contradiction with the benchmark. Blocking.** The proposal "log1p on contact counts after normalization; all losses in this space" conflicts with:
- **The primary benchmark.** v1 trains and evaluates on the **DNALongBench files**, whose target is fixed by the benchmark (as discussed, a distance-normalized log O/E-like map `[verify against the DNALongBench data/code]`). Our predictions and SCC must be in the benchmark's own target space, or the numbers aren't comparable with the reproduced Akita/Caduceus runs.
- **F1.** The softplus step assumes $\hat{C}$ is in a log/O/E space that can be negative. With a log1p target, $\hat{C} \ge 0$ and no softplus is needed.
- **The distance issue (earlier D1).** log1p of balanced counts is dominated by the distance decay.

**Proposal:** the training target = the DNALongBench target, unchanged. log1p(balanced counts) is not used in v1. First step in Phase 0: inspect the DNALongBench files and document the exact target transform.

**G2 — Insulation: predicted and true maps in different spaces. Blocking.** $\hat{I}$ is computed from $\operatorname{softplus}(\hat{C})$, and $I^{\text{true}}$ from $C^{\text{raw}}$. These are different scales:
- $C^{\text{raw}}$ (balanced counts) contains the full distance decay. $\operatorname{softplus}$ of a log-O/E value doesn't.
- softplus is not the inverse of log, so $\operatorname{softplus}(\log x) \ne x$.

$\mathcal{L}_{\text{ins}}$ would therefore force the map towards a systematically wrong scale. Also, v1 has only the DNALongBench files, so there is no $C^{\text{raw}}$ unless coolers are reprocessed.

**Fix:** apply **the same transform $g$ to both maps**: $\tilde{C} = g(\hat{C})$ and $\tilde{C}^{\text{true}} = g(C^{\text{target}})$, with the same operator, $w$ and mask.
- With $g = \exp$: if the target is log(O/E), $\exp$ is the exact inverse and gives O/E-scale insulation. This is safe because the target is clipped `[verify the range]`.
- With $g = \operatorname{softplus}$: a smooth approximation, but then on both sides.

Recommendation: $g = \exp$ if the target is natural-log O/E (else use the matching base), with a clamp on $\hat{C}$ for numerical safety. Additional notes:
- With $w = 20$, insulation is defined only for bins $i \in [20, 428)$ of the 448-bin output.
- Masked bins are excluded from the sum, and the $1/w^2$ normalization becomes "divide by the number of valid pairs".

**G3 — $\lambda_{\text{sym}}$ is redundant.** $\hat{C} = \mathrm{symmetrize}(\cdot)$ makes $\lVert\hat{C}-\hat{C}^\top\rVert_1 \equiv 0$. Drop the term: symmetry is architectural. Equivalently, it can be listed as a structural prior, but not as a loss.

**G4 — H3 data modes.**
- "Region-limited = official training regions of that line" and "Full = all training regions" are the same thing. Region-limited must be a **subset**, e.g. 10%, 25% and 50% of the training regions. Sample contiguous blocks rather than random windows, to avoid neighbouring windows giving away information.
- Read-limited needs the read pairs or contact counts of the held-out line, reprocessed with the **exact DNALongBench pipeline**. That is extra data processing in v1, which conflicts with "no extra data in v1". Either accept it as an exception (validated by reproducing the released targets at 100% reads), or move read-limited to v2.

**G5 — Significance level.** "If two-sided is requested, report with α = 0.025" has it the wrong way round. A one-sided test at α = 0.05 corresponds to a two-sided test at α = 0.10. A one-sided test equivalent to two-sided α = 0.05 uses **one-sided α = 0.025**. Recommendation: **pre-register one-sided α = 0.025** from the start (equivalently, a 97.5% one-sided CI bound > 0). That needs no later adjustment.

**G6 — Equivariance unit test conditions.**
- With bf16, `atol=1e-4` will fail through rounding alone. Run the exact test in **fp32**, and add a looser bf16 check (e.g. relative tolerance ~1e-2).
- **Train mode:** dropout (and stochastic layers) give different masks for $x$ and $\mathrm{RC}(x)$, so equality can't hold. Run the train-mode test with dropout disabled or the RNG seeded identically per call, or test in eval mode only plus a deterministic train-mode variant.
- The window crop (32 bins per side) is symmetric, so the 180° rotation stays valid after cropping.

**G7 — Locus-shared mean is an oracle-style reference.** It uses the **measured** test-region maps of the other cell types. That makes it a strong, legitimate reference for cell-specificity, but not a sequence-only predictor. Keep it **out of $\mathcal{B}$** in the H1 max (otherwise H1 compares a sequence model with measured data) and report it separately. The same applies to "locus-shared mean + cell offset".

**G8 — Independent boundaries: which source?** Published TAD annotations for these cell lines are usually derived from the **same** Hi-C/Micro-C datasets (often the same data behind the Akita/DNALongBench targets), so they aren't independent. Candidates:
- CTCF ChIP-seq (ENCODE)-bound sites with convergent orientation at insulation minima
- boundaries from a **different replicate or study** of the same cell line (e.g. 4DN)

Choose the source in Phase 0.

**G9 — Citation links don't match their sources.** In your list, the ChiMamba claim links to the Caduceus GitHub repository, and the arXiv-2606.04552 note links to a Springer book. The Springer book (978-981-97-5692-6) is presumably the ChiMamba source. In the reference manager, attach each claim to its own source. ChiMamba details (input, output format, resolution, window, split) are still to be extracted.

**G10 — Compute fallback changes the model.** The fallback for 8 GB (4 Bi-Mamba layers, hidden 128) is a different model from the 24–40 GB configuration. Decide **before** training which configuration is the paper model. Baselines (especially the matched Transformer) must be matched to that configuration. Don't mix results from the two configurations.
