# Data Sources and Dataset Construction — v1

Status: **largely superseded for v1.** v1 uses the DNALongBench contact-map data and official split directly, for the 5 Akita cell lines only. Pretraining uses only the DNALongBench training split. This document stays relevant for:
- the chromosome-held-out robustness analysis
- H4 SV data
- v2 (extra cell lines, epigenomic inputs, and the H3 read-depth-limited regime, which moved to v2 on 2026-10-21)

Resolved by the final decisions: D1 (v1 has no extra cell lines; v2 only same assay/resolution/normalization, with Hi-C and Micro-C flagged and analysed separately), D2 (pretraining on the DNALongBench training split only) and D4 (RC equivariance by architecture). Numbers marked `[UNVERIFIED]` must be checked against the source.

**Local reference genome.** A UCSC hg38 FASTA is already on disk at `/home/alan/Downloads/Human genomes Dataset/new_idea/datasets/reference_hg38/hg38.fa` (+ `.fa.gz`), documented in that folder's `MANIFEST.md` (Part B, §7.4). It includes alt/unplaced contigs, chrY and chrM, which this project excludes. Whether DNALongBench needs an external FASTA at all, and if so which hg38 build it expects, is `[UNVERIFIED]`. Check this against the DNALongBench release before using the file.

## 1. Data sources

| Data | Role | Source |
|---|---|---|
| Processed Akita Hi-C / Micro-C | Contact-map labels (5 cell lines: HFF, H1-hESC, GM12878, IMR-90, HCT116) | DNALongBench and Akita datasets |
| DNALongBench contact-map task | Ready-to-use benchmark: 9 cell lines; 7,008 train / 419 validation / 413 test sequences (verified from the DNALongBench and Akita papers) | Direct download. Fair comparison with Akita, CNN, HyenaDNA and Caduceus |
| Additional Hi-C cell lines: HAP1, HeLa, HepG2, K562 | **v1: H3 held-out lines only**, taken from the DNALongBench files (no in-house processing) | DNALongBench. Whether its processing matches Akita's exactly is `[UNVERIFIED]`; check in Execution step 1 |
| ENCODE CTCF ChIP-seq, ATAC-seq, H3K27ac | Optional input (v2) and mechanistic analysis (v1, e.g. CTCF orientation at predicted boundaries/loops) | ENCODE |
| 4D Nucleome | Hi-C, Micro-C, independent validation data | 4DN Data Portal |
| SVs / perturbation Hi-C | Variant generalization (H4) | Public studies pairing an engineered or naturally occurring structural rearrangement with Hi-C before and after. If too few: ClinVar / gnomAD-SV for **in-silico analysis only (H4-sim)**, with no claim of experimental validation |

For every dataset, record in `docs/data_manifest.md` (when created): accession, assay (Hi-C / Micro-C), protocol, read depth, replicate, genome build, license / data-use terms, download date.

## 2. Pretraining dataset construction (as proposed)

1. **Window extraction.** 1 Mb (1,048,576 bp) windows with a 250 kb or 500 kb stride along each chromosome. Exclude windows with high N content, assembly gaps, or very low Hi-C coverage.
2. **Binning.** 512 bins of 2,048 bp per window. Crop 32 bins on each side, so the cropped bins act as sequence context only → 448×448 output (DNALongBench-compatible).
3. **Label processing (as proposed; see D1).** KR or ICE normalization of the raw counts, then log1p. Mask low-quality rows and columns.
4. **Split (as proposed; see D2).** Chromosome held-out, e.g. test chr1, chr8, chr12 and validation chr3, chr6. Windows from one chromosome never appear in both training and test.
5. **Multi-cell batching.** For each window, the samples from all cell types in the corpus (5 in v1; the original proposal had 9) are grouped together, with a cell-type embedding in the batch. Missing cell types for a window are masked in the loss.
6. **Augmentation.** Only limited window shifting (small or bin-aligned). The v1 network is RC-equivariant by construction, so RC augmentation is redundant for it. It would only matter for a non-equivariant ablation or baseline, and then sequence and map are transformed together (D4). No independent rotations or flips of the contact map: the map has a genomic orientation (e.g. CTCF-orientation-dependent asymmetry).
7. **Independent validation data.** At least one cell type or one Hi-C study is excluded from all training. In v1 it is used for **few-shot** evaluation only. Zero-shot evaluation is v2.

## 3. Review notes

**D1 — Label processing must match DNALongBench/Akita exactly. Blocking.** "KR/ICE + log1p" is not the Akita/DNALongBench target. Per the Akita paper (verified 2026-10-21), the target is: ICE balancing → adaptive coarse-graining → distance normalization → natural log → clip to (−2, 2) → linear interpolation of missing bins → 2D Gaussian smoothing (σ = 1), i.e. a log(observed/expected) map. Still to be confirmed against the DNALongBench files (G1). Consequences:
- If the 4 extra cell lines use log1p(balanced) labels, they're on a different scale and with different content from the 5 Akita cell types. They can't share heads, losses or metrics with them, and H1 numbers wouldn't be comparable.
- log1p of balanced contact frequencies (small values) is dominated by the distance decay. The model then mostly learns P(s), and PCC in particular rewards that. (A distance-only baseline would look deceptively strong; it has since been removed, protocol §4.)
- **Proposal:** for any cell line processed in-house (v2), use the exact Akita/DNALongBench processing code. Validate it by reprocessing one of the 5 Akita cell types from its cooler and comparing with the released targets. Keep log1p(balanced) only as a secondary representation, if at all.
- (An earlier note tied this to the energy head; the energy head has since been removed, F2.)

**D2 — Split for the pretraining corpus vs. the primary DNALongBench split. Blocking.** The protocol now uses the **official DNALongBench split as the primary benchmark**, with chromosome held-out as robustness only. This data plan makes chromosome held-out the default for the pretraining corpus. If a model is pretrained on the chromosome-split corpus and then evaluated on the DNALongBench test set, any DNALongBench test/validation windows on the corpus's training chromosomes have been seen in training (in the same or highly similar cell types). That is **leakage into H1**. Options:
- (a) Two separate corpora and training runs: one following the DNALongBench split (for H1/H2 headline numbers) and one following the chromosome split (robustness, H3).
- (b) One corpus whose training set excludes **both** the held-out chromosomes **and** all DNALongBench validation/test regions (plus a buffer of ≥ 1 window). Usable for both evaluations, at the cost of less training data.

I'd recommend (a) for clarity, or (b) if compute is limited. Either way, write it down before training.

**D3 — Stride and overlap.** With 1 Mb windows and a 250 kb stride, adjacent training windows overlap by 75%. That's fine within the training set, but:
- the effective sample count is much lower than the window count, which matters for the bootstrap and for reporting dataset size
- for the robustness split, use non-overlapping (or fixed-stride, documented) windows for validation/test, so each locus is weighted equally
- define the coverage/N-content exclusion thresholds numerically, and use DNALongBench's criteria where they exist.

**D4 — Reverse-complement augmentation does flip the map.** Reverse-complementing the sequence reverses the bin order, so the target must be **reversed along both axes**: $C'_{ij} = C_{n-1-i,\,n-1-j}$, i.e. the 180° rotation $\mathrm{rot}_{180}$ used in `architecture.md`. For a symmetric map this equals the anti-diagonal flip $C_{n-1-j,\,n-1-i}$. That *is* a map transform, but a coupled one, and it's correct. The rule should read: "no flip or rotation of the map **independent of** the sequence; RC is applied to sequence and map together." Window shifts must either be small relative to the 2 kb bin (Akita uses shifts of ±11 bp, verified from the paper) or be bin-aligned with the target recomputed.

**D5 — Combining Hi-C and Micro-C.** HFF and H1-hESC are Micro-C (Krietenstein et al. 2020); GM12878 and IMR-90 are Hi-C (Rao et al. 2014); HCT116 is Hi-C (Rao et al. 2017). All verified from the Akita paper's Table 1. The assay of the 4 H3 lines is `[UNVERIFIED]`. In v1 the benchmark processing is inherited as-is and assay effects are not modelled (manuscript §7, limitation 8). For v2, whether the assay becomes a covariate (embedding) or is analysed separately is still open. Also record depth, and consider depth-matched downsampling for the v2 read-depth-limited H3 regime.

**D6 — Independent validation cell type/study.** In v1 these are the 4 H3 held-out lines, HAP1, HeLa, HepG2 and K562 (protocol §5 H3). For H3 the evaluation regions must also be absent from pretraining for all cell types (CLAUDE.md, v1 scope; protocol §5 H3).

**D7 — ClinVar / gnomAD SVs.** These give realistic SV *positions and sizes* but no observed contact maps, so they only support H4-sim. VCF coordinates are 1-based. Convert explicitly, and handle sequence-length changes (deletions/insertions) by a documented padding/cropping and map-alignment scheme.
