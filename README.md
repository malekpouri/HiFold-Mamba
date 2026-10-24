# HiFold-Mamba

Sequence-based prediction of 3D chromatin folding at megabase scale.

*Working title, conditional on pre-specified outcomes (see [Stopping rules](#stopping-rules)): "Insulation-Constrained Multi-Scale State-Space Modeling for Megabase-Scale Prediction of 3D Chromatin Folding".*

> **Status: protocol stage — no results yet.** The analysis plan, metrics, baselines, statistical tests and stopping rules in `docs/` were fixed **before** any training, and no model has been trained. The manuscript's Results section is deliberately empty. Nothing here reports a measured outcome.

## What this project does

Predicts Hi-C / Micro-C contact maps from DNA sequence alone: a 1,048,576 bp window in, a 448×448 map at 2,048 bp resolution out, on the DNALongBench contact-map task in hg38.

The goal is not only a high overall correlation, but recovery of the structure that matters biologically — TAD boundaries and insulation, CTCF-anchored loops, and the changes structural variants cause.

## Approach

Replacing a convolutional or attention encoder with a state-space model is not, on its own, a contribution: Caduceus already pairs a bidirectional Mamba with reverse-complement equivariance for DNA. At 512 bins per window, attention is also cheap, so **no efficiency advantage is claimed at this scale**. The contribution is instead:

1. **An insulation constraint on the predicted map.** Insulation and boundary scores are computed *from the predicted contact map*, with the same operator applied to the observed map, so the structural gradient reaches the map itself. An auxiliary structure head can be right while the map is wrong; this cannot.
2. **Network-wide RC equivariance** — `f(RC(x)) = rot180(f(x))` for the whole network, not just the state-space blocks, enforced by a unit test that must pass before training starts.
3. **Multi-scale state mixing** across shallow, intermediate and deep encoder states, for domain-scale and loop-scale structure together.
4. **Few-shot cell-type adaptation** via a FiLM cell-type embedding, measured by leave-one-cell-line-out transfer against training from scratch under equal compute.
5. **A pre-specified analysis plan** with per-block testing against the strongest baseline in each block, oracle-style reference baselines, non-circular structural evaluation, and stopping rules that end the project if none of its pre-specified claims holds.

## Architecture

| Module | Role |
|---|---|
| Conv1D encoder | 1,048,576 bp → 512 bins of 2 kb; RC-paired kernels |
| Bi-Mamba encoder | 8–12 blocks over bins; forward and RC paths share parameters (MambaDNA / Caduceus-PS) |
| Multi-scale mixer | Pyramidal pooling over shallow/intermediate/deep states |
| Pairwise head | Symmetric biaffine `sym(UVᵀ/√d + b_|i−j|)`, cropped to 448×448 |
| Structure terms | Insulation, boundaries and loops derived from the predicted map — no separate heads, no energy head |
| Cell-type adapter | FiLM from a learned per-cell-line embedding |

## Protocol summary

| Item | Choice |
|---|---|
| Benchmark | DNALongBench contact-map task, hg38 |
| Cell lines | Training: HFF, H1-hESC, GM12878, IMR-90, HCT116. Few-shot held-out only: HAP1, HeLa, HepG2, K562 (the benchmark's other four lines) |
| Split | Official DNALongBench files (region-based); chromosome-held-out as separate robustness. **Never** a window-level random split |
| Primary metric | **SCC** — stratum-adjusted correlation coefficient (HiCRep). PCC always reported alongside; Spearman, where used, is named in full |
| Statistics | SCC per window → averaged over 3 seeds → 1 Mb blocks → ΔSCC vs. the per-block best baseline → one-sided paired block bootstrap, 10,000 resamples, α = 0.025 (reported with the 95% percentile CI, whose lower limit is the one-sided bound) |
| Success | Mean block-wise ΔSCC ≥ +0.01 with a positive 97.5% one-sided bound |
| Baselines | Akita, Caduceus-Ph/PS, matched Transformer, strong CNN; mean-map references reported separately |
| Precision | bf16 mixed precision |

Blocks rather than windows are the resampling unit because neighbouring windows overlap and are not independent. The per-block maximum prevents post-hoc baseline selection. The global cell-line mean and the locus-shared references are excluded from the primary comparison set. The locus-shared references use measured maps from evaluation regions of *other* cell lines, so they are context only.

## Hypotheses

- **H1** — exceeds the strongest trained baseline on contact-map accuracy.
- **H2** — the insulation constraint improves TAD and loop recovery under equal compute, evaluated on **independent** boundaries (insulation-derived boundaries are internal consistency only, since training supervises insulation).
- **H3** — pretraining on the five training lines plus few-shot adaptation beats training from scratch on the same limited data, for at least two of the four held-out lines; evaluated on official test regions never seen in training.
- **H4** — tracks the direction of structural-variant effects, **only** where observed before/after maps exist; otherwise restricted to a labelled in-silico analysis.

## Stopping rules

Fixed in advance (`docs/evaluation_protocol.md` §7). Three narratives each have a pre-specified criterion: **insulation-constrained** (H2 met), **multi-scale** (H1 met and the mixer ablation significant), and **few-shot adaptation** (H3 met in at least two held-out lines). The title follows whichever holds. If none holds, the project is stopped and redesigned, not reframed as a benchmark or negative-result paper. If the model does not beat the global cell-line mean, the project fails in its current form. If the structural loss shows no benefit, every structure-constrained claim is removed, including from the title.

## Repository layout

```
CLAUDE.md                      working rules and conventions
Roadmap.md                     decisions, phases, open items
docs/evaluation_protocol.md    benchmark, metrics, statistics, baselines, hypotheses
docs/architecture.md           model, training, loss, ablations
docs/data_processing.md        data sources and dataset construction
paper/manuscript.md            manuscript draft (Results empty)
```

Created when first needed: `configs/` (one config per run), `src/` (model, training, evaluation), `notebooks/` (exploration and sanity checks), `notes/` (experiment log, including negative results), `docs/data_manifest.md` (data provenance).

## Open items before training

All protocol items are decided. Two are provisional until the DNALongBench files are inspected, because they determine where the gradient goes:

1. **Target space.** The papers document natural-log observed/expected, clipped to (−2, 2), interpolated and smoothed, with diagonal offset 2. The released files must confirm this before any loss is written.
2. **Insulation transform.** `g(x) = exp(2·tanh(x/2))`, applied identically to prediction and target.

Also outstanding: reading the ChiMamba chapter in full (closest architectural precedent), and selecting the accessions of the independent boundary and loop datasets.

## Reproducibility

Every reported number must be reproducible from a committed config and recorded commit, with 3 seeds per model and logged seeds, hardware and versions. Negative and null results are reported with the same prominence as positive ones.

## Data

All data are public: DNALongBench for training and evaluation; 4D Nucleome Hi-C/Micro-C and ENCODE CTCF ChIP-seq and ATAC-seq for structural evaluation. Histone marks and other epigenomic inputs are v2 only. Accessions, assay, depth, build, licence and access date will be recorded in `docs/data_manifest.md` as each dataset is obtained. No individual-level human genotype data are used — reference sequence and cell-line data only.

## Citation

Not yet published. Cite the repository if you use the protocol or code.

## Authors

- **Mohammad Malekpouri** (corresponding) — [0009-0001-6278-6163](https://orcid.org/0009-0001-6278-6163)
- **Somayeh Lotfi** — [0000-0002-7218-5388](https://orcid.org/0000-0002-7218-5388)

Department of Computer Engineering, BA.C., Islamic Azad University, Bandar Abbas, Iran

## AI assistance

Claude (Anthropic) was used to generate code and to draft and edit the manuscript text. The authors made or approved every design decision and take full responsibility for the content. Items not yet verified are marked `[UNVERIFIED]` in the documents.

## License

MIT — see [LICENSE](LICENSE).
