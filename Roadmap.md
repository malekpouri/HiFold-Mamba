# Roadmap — Megabase-scale prediction of 3D chromatin folding

**Title (conditional):** no "physics" and no "energy head". Use *structure-constrained* / *insulation-constrained* only if $\mathcal{L}_{\text{struct}}$ helps significantly (H2 met). A bare "RC-equivariant Bi-Mamba" title is excluded because Caduceus has the same core. The three narratives, each with a pre-specified criterion, are in protocol §7:
- insulation-constrained prediction (N-I: H2 met)
- multi-scale state-space modeling (N-M: H1 met **and** the mixer ablation is significant)
- few-shot cell-type adaptation (N-F: H3 bound > 0 in ≥ 2 held-out lines)

If none of them holds, stop and redesign. "Multi-megabase" only if the 4–8 Mb phase is done.

## Research question

Can an RC-equivariant bidirectional Mamba architecture with structural priors predict megabase-scale contact maps more accurately than Akita and Caduceus, and model structural-variant effects better?

## Core claim

A multi-scale sequence-to-structure architecture with structural inductive bias, not "a new Mamba".

## Binding documents

| Document | Content |
|---|---|
| `docs/evaluation_protocol.md` | Benchmark, split, metrics, statistics, baselines, H1–H4, decision rules, titles, **open items G1–G10** (F1–F9 resolved) |
| `docs/architecture.md` | HiFold-Mamba, training, loss, ablations |
| `docs/data_processing.md` | Data sources and dataset construction (v1 now uses the DNALongBench data directly) |

## Final decisions (v1)

1. Benchmark: DNALongBench contact-map task (1,048,576 bp, 2 kb bins, 448×448); metrics SCC (stratum-adjusted) + PCC.
2. Primary split: official DNALongBench files. Chromosome held-out only as robustness, with all baselines retrained.
3. Cell lines: training **only HFF, H1-hESC, GM12878, IMR-90 and HCT116** (the Akita five). The 4 lines DNALongBench added (HAP1, HeLa, HepG2, K562) are H3 held-out lines only.
4. Pretraining: only the DNALongBench training split.
5. Statistics: SCC per window → average over 3 seeds → 1 Mb blocks → paired block bootstrap (10,000 resamples), **one-sided test at α = 0.025**. Reported: mean ΔSCC, 95% percentile CI (its lower limit is the 97.5% one-sided bound), one-sided p-value. Wilcoxon secondary only.
6. Success threshold: ΔSCC ≥ +0.01 over the **per-block best** baseline in ℬ, with the one-sided 97.5% bound above 0.
7. Baselines: trained sequence models in ℬ: Akita, Caduceus-Ph/PS, matched Transformer, strong CNN (optional: HyenaDNA; DeepC only if fully retrained; ChiMamba as a baseline or in a comparison table). References outside ℬ: global cell-line mean; locus-shared mean and locus-shared mean + cell offset (oracle-style, G7). The distance-only baseline is removed.
8. Architecture: RC-equivariant Bi-Mamba (Caduceus-PS style), mandatory FiLM cell-type adapter, insulation/boundaries computed from the predicted map.
9. Training: bf16, batch from memory profiling, effective batch 8–16, EMA, early stopping on validation HiCRep SCC.
10. H2: Base vs. Base + $\mathcal{L}_{\text{struct}}$ under identical optimizer steps.
11. H3: few-shot only. Held-out lines = HAP1, HeLa, HepG2, K562 (never in pretraining); the pretrained model is the main 5-line model; evaluation regions excluded from pretraining. Limited-data regimes = contiguous-block subsets (10/25/50%) of the held-out line's training regions, plus 100%. Read-depth-limited → v2.
12. H4: experimental SVs, or H4-sim with explicit in-silico labels.
13. Efficiency claims only for 4–8 Mb inputs or 128 bp tokens.
14. Hard stop: the project is stopped and redesigned only if **none** of the three narratives N-I, N-M, N-F meets its pre-specified criterion (protocol §7). Independently, failing to beat the global cell-line mean means the project fails in its current form. There is no fallback to an "RC-equivariant Bi-Mamba" paper.

## Execution plan

| Step | Task | Hypothesis |
|---|---|---|
| 0 | **Phase 0 (closing):** G1–G10 decided (G1/G2 provisional). Sample counts 7,008/419/413, crop, SCC/PCC, 9 cell lines and diagonal offset 2 verified from the papers. Remaining: read the ChiMamba chapter. Freeze the protocol | — |
| 1 | Download DNALongBench (record in `docs/data_manifest.md`); **first: confirm the target transform, value range, masks and diagonal offset in the files (closes G1/G2)**; reproduce Akita and Caduceus on the official split, per cell line for the 5 training lines; memory profiling on the local GPU | Baselines |
| 2 | Implement RC-equivariant Bi-Mamba + cell-type adapter (+ equivariance unit test) | — |
| 3 | Train the base model on 5 cell lines, 3 seeds | H1 |
| 4 | Add $\mathcal{L}_{\text{struct}}$ with the same training budget | H2 |
| 5 | Full ablation: Bi-Mamba, RC equivariance, structure loss, multi-scale mixer, matched Transformer | H1/H2 |
| 6 | H3 few-shot on the 4 held-out DNALongBench lines, and chromosome-held-out robustness | H3 |
| 7 | H4 on experimental SVs, or H4-sim | H4 |
| 8 | Block bootstrap, final tables, manuscript, AI-use disclosure | — |
| v2 | Epigenomic inputs (CTCF/ATAC/H3K27ac), zero-shot cell types, extra cell lines, 4–8 Mb inputs, 128 bp tokens | — |

Go/no-go checkpoints: after step 1 (baselines reproduced within tolerance) and about 2–3 months into steps 3–5 (decision rules in protocol §7).

## Prior art

| Model | Note |
|---|---|
Full, verified citations are in `paper/manuscript.md`, References.

| Model | Note |
|---|---|
| Akita (Fudenberg et al. 2020, *Nat Methods*) | Primary baseline; defines the 1 Mb / 2 kb / 448×448 setting |
| Caduceus-Ph/PS (Schiff et al. 2024, ICML) | Bidirectional, RC-equivariant Mamba. Our backbone follows its PS design, so the novelty must come from the structure terms, multi-scale mixer, cell-type adapter and evaluation |
| HyenaDNA (Nguyen et al. 2023, NeurIPS) | DNALongBench baseline; optional |
| DNALongBench (Cheng et al. 2025, *Nat Commun*) | Primary benchmark. Preprint: Akita 0.230 vs. HyenaDNA 0.115, Caduceus-Ph 0.133, Caduceus-PS 0.127 (average contact-map score, 9 lines) |
| DeepC (Schwessinger et al. 2020, *Nat Methods*) | Only if fully retrained |
| ChiMamba (Zhang et al. 2024, LNCS 14882) | Bi-Mamba + chromatin accessibility, six cell lines (from the abstract). Positioned as related work unless the full chapter shows a comparable sequence-only setting (G9). **Read before training** |
| Orca (Zhou 2022, *Nat Genet*) | Competitor for the multi-Mb phase |
| C.Origami (Tan et al. 2023, *Nat Biotechnol*) | Competitor for v2 (epigenomic inputs) |
| AlphaGenome (Avsec et al. 2026, *Nature*) | Optional |

## Resolved in the last round (F1–F9)

- **F1:** crossing-square insulation computed from the map; $w$ = 20 bins (sensitivity 10/20/40).
- **F2:** energy head removed.
- **F3:** H3 evaluated on the official test regions of held-out lines (originally leave-one-out among the five, rotated over ≥ 2 lines; superseded on 2026-10-21 by the 4 DNALongBench-added lines).
- **F4:** global RC equivariance + a mandatory unit test.
- **F5:** per-block max, average over seeds first, one-sided test.
- **F6:** baseline table revised.
- **F7:** independent boundaries are the main H2 evidence.
- **F8:** three fallback narratives; otherwise stop.
- **F9:** Caduceus confirmed; ChiMamba source identified as a Springer chapter (chapter not yet read, G9); arXiv 2606.04552 removed.
- SVs: < 10 valid cases → H4-sim. Compute: 24–40 GB assumed, with an 8 GB fallback model.

## Resolved 2026-10-21 (user decisions)

- **Hard stop:** three-narrative rule with pre-specified criteria (protocol §7). Old decision 14 ("narrow to RC-equivariant Bi-Mamba") deleted.
- **G3:** $\lambda_{\text{sym}}$ dropped; symmetry is architectural.
- **G4:** H3 regimes = contiguous-block subsets 10/25/50% + 100% of the held-out line's training regions. Read-depth-limited moved to v2.
- **G5:** one-sided α = 0.025; report the 95% percentile CI, whose lower limit is the 97.5% one-sided bound.
- **G7 / mean-map stopping rule:** locus-shared references stay out of ℬ and are context only. The stopping bar is the global cell-line mean.

## Resolved 2026-10-21 (delegated by the user to Claude; review welcome)

The user asked for "whatever approach is best". Each choice is recorded here so it can be reversed before training.

- **Cell lines:** train on the Akita five; the 4 DNALongBench-added lines are H3 held-out lines. Why: keeps the v1 scope, uses only official benchmark files, gives H3 four held-out lines instead of a rotation, and needs no 4-line re-pretraining.
- **G1 (provisional):** target = natural-log O/E, clipped [−2, 2], interpolated, smoothed; upper triangle, diagonal offset 2. Source: the Akita and DNALongBench papers. Closed only when the files confirm it (Execution step 1).
- **G2 (provisional):** $g(x) = \exp(2\tanh(x/2))$ on both maps. Smooth, bounded to $[e^{-2}, e^{2}]$, ≈ exp near 0, and identical on both sides.
- **G6:** fp32 test (atol 1e-4, rtol 1e-5); bf16 check (rtol 1e-2); eval mode, plus train mode with dropout 0 or identical RNG.
- **G8:** primary = boundaries called with a pre-fixed caller from an **independent** Hi-C/Micro-C dataset of the same line (different study or replicate from the training target); secondary = ENCODE CTCF ChIP-seq peaks with a CTCF motif. Loops likewise from an independent dataset. The Rao et al. 2014 GM12878/IMR-90 loop calls are excluded, because they come from the training data. Accessions to be chosen in step 1 and recorded in the manifest.
- **G9:** citations corrected; ChiMamba = related work pending the full chapter.
- **G10:** the paper configuration is the 24–40 GB model; the 8 GB fallback is for development only and never reported.

## Still open

| # | Item | Status |
|---|---|---|
| G1/G2 | Confirm target transform, range, masks and offset in the DNALongBench files | Step 1, **blocks loss code** |
| G9 | Read the ChiMamba chapter (paywalled) | Before training |
| — | Compute environment and budget (cloud/HPC) | Open |
| — | LICENSE copyright holder(s) | Authors' decision |
| — | Target venue | Open |
| — | SV data sources for H4 | Open |
| — | Hyperparameters, conv stack, $\mathcal{L}_{\text{bnd}}$ form | Before coding |

## Risks

- **Novelty:** the backbone follows Caduceus-PS, and ChiMamba may overlap. The contribution has to come from the map-derived structure terms, multi-scale mixing, cell-type transfer and variant evaluation.
- **H1 margin:** +0.01 SCC over Akita/Caduceus may be hard to reach. The hard stop is predefined (protocol §7).
- **SV ground truth is scarce**, so H4-sim or the Discussion are possible outcomes.
- **H3 held-out lines** come from DNALongBench's own processing, which is not verified to match Akita's pipeline exactly; the assay per line is to be confirmed.
- **Compute:** local RTX 5050 (8 GB) for development only. `mamba-ssm` kernel support on Blackwell needs to be checked.
