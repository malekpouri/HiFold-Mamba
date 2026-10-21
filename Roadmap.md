# Roadmap — Megabase-scale prediction of 3D chromatin folding

**Title (conditional):** no "physics" and no "energy head". Use *structure-constrained* / *insulation-constrained* only if $\mathcal{L}_{\text{struct}}$ helps significantly. A bare "RC-equivariant Bi-Mamba" title is excluded because Caduceus has the same core. The hard-stop narratives are listed in protocol §7:
- insulation-constrained prediction
- few-shot cell-type adaptation
- multi-scale state-space modeling

If none of them holds, stop and redesign. "Multi-megabase" only if the 4–8 Mb phase is done.

## Research question

Can an RC-equivariant bidirectional Mamba architecture with structural priors predict megabase-scale contact maps more accurately than Akita and Caduceus, and model structural-variant effects better?

## Core claim

A multi-scale sequence-to-structure architecture with structural inductive bias, not "a new Mamba".

## Binding documents

| Document | Content |
|---|---|
| `docs/evaluation_protocol.md` | Benchmark, split, metrics, statistics, baselines, H1–H4, decision rules, titles, **open items F1–F9** |
| `docs/architecture.md` | HiFold-Mamba, training, loss, ablations |
| `docs/data_processing.md` | Data sources and dataset construction (v1 now uses the DNALongBench data directly) |

## Final decisions (v1)

1. Benchmark: DNALongBench contact-map task (1,048,576 bp, 2 kb bins, 448×448); metrics SCC (stratum-adjusted) + PCC.
2. Primary split: official DNALongBench files. Chromosome held-out only as robustness, with all baselines retrained.
3. Cell lines: **only HFF, H1-hESC, GM12878, IMR-90 and HCT116.**
4. Pretraining: only the DNALongBench training split.
5. Statistics: SCC per window → 1 Mb blocks → paired block bootstrap (10,000 resamples), 3 seeds, 95% CI, two-sided p-value. Wilcoxon secondary only.
6. Success threshold: ΔSCC ≥ +0.01 over the best baseline with the CI above 0.
7. Baselines: global mean map, locus-shared mean, cell-type prior, Akita, Caduceus-Ph/PS, matched Transformer. DeepC only if fully retrained. ChiMamba as a baseline or in a comparison table. The distance-only baseline is removed.
8. Architecture: RC-equivariant Bi-Mamba (Caduceus-PS style), mandatory FiLM cell-type adapter, insulation/boundaries computed from the predicted map.
9. Training: bf16, batch from memory profiling, effective batch 8–16, EMA, early stopping on validation HiCRep SCC.
10. H2: Base vs. physics-guided under identical optimizer steps.
11. H3: few-shot only, held-out cell lines, evaluation regions excluded from pretraining.
12. H4: experimental SVs, or H4-sim with explicit in-silico labels.
13. Efficiency claims only for 4–8 Mb inputs or 128 bp tokens.
14. Hard stop: ΔSCC < +0.01 and no structural gain → narrow to "RC-equivariant Bi-Mamba for megabase-scale chromatin folding".

## Execution plan

| Step | Task | Hypothesis |
|---|---|---|
| 0 | **Phase 0 (current):** resolve G1–G10. **Inspect the DNALongBench files and document the exact target transform (G1)**. Extract the ChiMamba details. Verify DNALongBench cell types, SCC settings, split files and sample counts (7,008/419/413). Freeze the protocol | — |
| 1 | Download DNALongBench; reproduce Akita and Caduceus on the official split; memory profiling on the local GPU | Baselines |
| 2 | Implement RC-equivariant Bi-Mamba + cell-type adapter (+ equivariance unit test) | — |
| 3 | Train the base model on 5 cell lines, 3 seeds | H1 |
| 4 | Add $\mathcal{L}_{\text{struct}}$ with the same training budget | H2 |
| 5 | Full ablation: Bi-Mamba, RC equivariance, physics loss, multi-scale mixer, matched Transformer | H1/H2 |
| 6 | H3 few-shot (leave-one-cell-type-out, F3) and chromosome-held-out robustness | H3 |
| 7 | H4 on experimental SVs, or H4-sim | H4 |
| 8 | Block bootstrap, final tables, manuscript, AI-use disclosure | — |
| v2 | Epigenomic inputs (CTCF/ATAC/H3K27ac), zero-shot cell types, extra cell lines, 4–8 Mb inputs, 128 bp tokens | — |

Go/no-go checkpoints: after step 1 (baselines reproduced within tolerance) and about 2–3 months into steps 3–5 (decision rules in protocol §7).

## Prior art

| Model | Note |
|---|---|
| Akita (Fudenberg et al. 2020) | Primary baseline; defines the 1 Mb / 2 kb / 448×448 setting |
| Caduceus-Ph/PS (Schiff et al. 2024) | Bidirectional, RC-equivariant Mamba. Our backbone follows its PS design, so the novelty must come from the structure terms, multi-scale mixer, cell-type adapter and evaluation |
| HyenaDNA | DNALongBench baseline (whether to keep it is open, F6) |
| DNALongBench (2025) | Primary benchmark |
| DeepC (Schwessinger et al. 2020) | Only if fully retrained |
| ChiMamba `[UNVERIFIED]` | Bi-Mamba + accessibility for chromatin interactions. **Check before training** |
| Orca (Zhou 2022) | Competitor for the multi-Mb phase |
| C.Origami (Tan et al. 2023) | Competitor for v2 (epigenomic inputs) |
| AlphaGenome (2025) | Optional |

## Resolved in the last round (F1–F9)

- **F1:** crossing-square insulation computed from the map; $w$ = 20 bins (sensitivity 10/20/40).
- **F2:** energy head removed.
- **F3:** H3 = leave-one-cell-line-out, evaluated on the official test regions, rotated over ≥ 2 lines.
- **F4:** global RC equivariance + a mandatory unit test.
- **F5:** per-block max, average over seeds first, one-sided test.
- **F6:** baseline table revised.
- **F7:** independent boundaries are the main H2 evidence.
- **F8:** three fallback narratives; otherwise stop.
- **F9:** Caduceus confirmed, ChiMamba confirmed as a Springer chapter, arXiv 2606.04552 removed.
- SVs: < 10 valid cases → H4-sim. Compute: 24–40 GB assumed, with an 8 GB fallback model.

## Still open

| # | Item | Status |
|---|---|---|
| G1 | Target space: the proposed "log1p of normalized counts" conflicts with the DNALongBench target and with F1 | **Blocking** |
| G2 | Insulation: softplus(Ĉ) vs. $C^{\text{raw}}$ are different spaces. Apply the same transform to both | **Blocking** |
| G3 | $\lambda_{\text{sym}}$ term is identically 0 (map is symmetrized): drop it | Open (trivial) |
| G4 | H3: "region-limited" = "full" as defined; read-limited needs reprocessing (an exception to "no extra data") | Open |
| G5 | Significance: one-sided α = 0.025 ≡ two-sided 0.05 (the stated mapping is the wrong way round) | Open |
| G6 | Equivariance test: fp32 for exact tolerance; dropout in train mode | Open |
| G7 | Locus-shared means are oracle-style references, so keep them out of the H1 max set | Open |
| G8 | Source for independent TAD boundaries | Open |
| G9 | Citation links attached to the wrong sources; extract ChiMamba details | Phase 0 |
| G10 | Choose the paper configuration (24–40 GB vs. 8 GB fallback) before training | Open |
| — | Target venue | Open |
| — | SV data sources for H4 | Open |
| — | Hyperparameters, conv stack, $\mathcal{L}_{\text{bnd}}$ form | Before coding |

## Risks

- **Novelty:** the backbone follows Caduceus-PS, and ChiMamba may overlap. The contribution has to come from the map-derived structure terms, multi-scale mixing, cell-type transfer and variant evaluation.
- **H1 margin:** +0.01 SCC over Akita/Caduceus may be hard to reach. The hard stop is predefined.
- **SV ground truth is scarce**, so H4-sim or the Discussion are possible outcomes.
- **H3 has only 5 cell lines**, so few held-out cases.
- **Compute:** local RTX 5050 (8 GB) for development only. `mamba-ssm` kernel support on Blackwell needs to be checked.
