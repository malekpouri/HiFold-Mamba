# CLAUDE.md — 3D Chromatin Folding Prediction

## Project in one paragraph

We predict Hi-C / Micro-C contact maps from DNA sequence (epigenomic inputs only in v2). Success means more than a high overall correlation. The model must reproduce TAD boundaries/insulation, loops, CTCF-mediated features and structural-variant effects. The contribution is **a multi-scale sequence-to-structure architecture with structural inductive bias**, built on an RC-equivariant bidirectional Mamba (HiFold-Mamba). It is not "a new Mamba".

**Titles are conditional** (see protocol §7). Never "physics-guided" or "energy head". "Structure/insulation-constrained" only if $\mathcal{L}_{\text{struct}}$ shows a significant benefit. Never a bare "RC-equivariant Bi-Mamba" claim (that is Caduceus' core). v1 is **megabase-scale**. Never call v1 results "multi-megabase".

## Binding documents (follow exactly; propose changes, never make them silently)

- `Roadmap.md`: final decisions, execution plan, open items.
- `docs/evaluation_protocol.md`: benchmark, split, metrics, statistics, baselines, H1–H4, decision rules, open items G1–G10 (F1–F9 resolved).
- `docs/architecture.md`: HiFold-Mamba, training, loss, ablations.
- `docs/data_processing.md`: data sources; relevant mainly for the robustness split, H3 low-read and v2.

## v1 scope (fixed)

- Human, hg38. **DNALongBench contact-map task**: 1,048,576 bp input, 2,048 bp bins, 448×448 output.
- **Training cell lines: HFF, H1-hESC, GM12878, IMR-90, HCT116 only.** The DNALongBench contact-map task has 9 lines. The other 4 (HAP1, HeLa, HepG2, K562) are used **only** as H3 held-out lines, never in (pre)training.
- **Primary split: official DNALongBench files.** Pretraining uses the DNALongBench training split only. Chromosome held-out is a separate robustness analysis.
- H3 is few-shot only (FiLM cell-type embedding), on held-out cell lines, with evaluation regions excluded from pretraining.

## Current phase

**Phase 0, nearly closed: all G-items are decided (G1/G2 provisionally, pending inspection of the DNALongBench files). Remaining: read the ChiMamba chapter, verify the target against the files. Do not write code until the user explicitly asks for it.**

## Working rules for Claude

- **Verify before you claim.** Check papers, model specifications, accessions and numbers against the primary source. Otherwise mark them `[UNVERIFIED]`. Never invent citations, DOIs or numbers.
- Prior-art claims must cite a specific paper or benchmark.
- Ask before settling anything listed as open.
- Small verifiable steps. Sanity checks (heatmap, P(s), coverage) after every data step.
- Report all results honestly, including negative ablations, in `notes/`. If a decision-rule trigger fires (protocol §7), report it to the user. Don't work around it.

## Scientific conventions

| Item | Convention |
|---|---|
| Genome build | hg38 only |
| Chromosomes | `chr` prefix. chrY, chrM, alt and unplaced contigs excluded |
| Coordinates | 0-based, half-open. Be explicit when converting from 1-based (VCF, GFF) |
| Target | DNALongBench processed target, unchanged. Documented (Akita paper) as natural-log O/E, clipped to [−2, 2], missing bins interpolated, Gaussian-smoothed; upper triangle with diagonal offset 2 (99,681 values per map). **Can be negative.** Verify against the files before any loss is written (G1). No log of the target or of predictions. Any transform used in a loss (e.g. for insulation) is applied **identically** to prediction and target |
| Masking | NaN / masked bins never count in losses or metrics |
| Symmetry | Built into the model (symmetrized pairwise head) |
| Diagonal | Excluded as in DNALongBench/Akita |
| Splits | **Never a random window-level split.** No validation/test region ever in (pre)training. H3 evaluation regions excluded from pretraining |
| Augmentation / RC | The network is RC-equivariant by construction: $f(\mathrm{RC}(x)) = \mathrm{rot}_{180}(f(x))$. Shifts only if small or bin-aligned. No map transforms independent of the sequence |
| Seeds | 3 seeds per model, logged. Every number reproducible from config + commit |

## Evaluation (details in the protocol)

- **SCC = stratum-adjusted correlation coefficient** (HiCRep), computed per window. Never "similarity" or "Spearman". PCC always reported. Spearman correlation is supplementary and labelled as such.
- Checkpoint selection: mean validation HiCRep SCC.
- Primary test: average SCC over 3 seeds per window → 1 Mb blocks → ΔSCC vs. the **per-block best** trained baseline → one-sided paired block bootstrap, 10,000 resamples. One-sided α = 0.025. Success: ΔSCC ≥ +0.01 with a positive 97.5% one-sided bound (= lower limit of the 95% percentile CI).
- Hard stop only if none of the three narratives (insulation, multi-scale, few-shot) meets its criterion (protocol §7). Not beating the global cell-line mean = failure; locus-shared references are context only.
- Main H2 evidence uses **independent** boundaries/loops. Insulation-derived boundaries are internal consistency only.
- Insulation and boundaries are computed **from the predicted map**, with the same operator as for the true map.
- Efficiency claims never at 512 bins. Only for 4–8 Mb inputs or 128 bp tokens.

## Hardware and environment

- Local GPU: **NVIDIA GeForce RTX 5050, 8 GB VRAM** (Blackwell), for development. Full runs probably need cloud/HPC (budget open).
- **bf16 mixed precision only** + gradient checkpointing. Batch from profiling (1/2/4), effective batch 8–16 via accumulation.
- Check `mamba-ssm` / `causal-conv1d` kernel support for this GPU before relying on them.
- Environment manager and data path: TBD. Never commit data, coolers or checkpoints.

## Directory layout (create only when needed)

```
CLAUDE.md, Roadmap.md
docs/        protocol, architecture, data processing, literature notes
configs/     one config per run
src/         code (later)
notebooks/   exploration and sanity checks
notes/       experiment log, including negative results
paper/       manuscript and figures
```

## Paper and AI-use notes

- Track AI assistance for the journal's disclosure statement.
- All references in `paper/` checked against the actual publication.
