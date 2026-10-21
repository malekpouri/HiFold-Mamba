# HiFold-Mamba: Insulation-Constrained Multi-Scale State-Space Modeling for Megabase-Scale Prediction of 3D Chromatin Folding

> **Draft status — protocol / registered-report stage.** Methods and the analysis plan are complete and pre-specified. **Results and Discussion are deliberately empty**: no experiment has been run. Every table in §5 is a placeholder with explicit `[TBD]` cells. Nothing in this file may be filled in by estimation, extrapolation or expectation — only from logged experimental runs with a config hash and a commit id. Title is conditional (see §8).

**Authors**

Mohammad Malekpouri^a,\*^ · Somayeh Lotfi^a^

^a^ Department of Computer Engineering, BA.C., Islamic Azad University, Bandar Abbas, Iran

\* Corresponding author. Mohammad Malekpouri — ORCID [0009-0001-6278-6163](https://orcid.org/0009-0001-6278-6163)
Somayeh Lotfi — ORCID [0000-0002-7218-5388](https://orcid.org/0000-0002-7218-5388)

**Code availability:** https://github.com/malekpouri/HiFold-Mamba

---

## Abstract

*(Written last. Results sentences stay blank until §5 is filled.)*

**Background.** Sequence-to-contact-map models such as Akita and DeepC predict Hi-C and Micro-C maps at roughly megabase scale with convolutional architectures, while general long-range DNA models (HyenaDNA, Caduceus) underperform task-specific models on contact-map prediction. The open problem is not the choice of sequence backbone but the combination of an efficient long-range encoder, structural inductive bias, and generalization across cell types and sequence variants.

**Methods.** We present HiFold-Mamba, a sequence-only model that predicts 448×448 contact maps from 1,048,576 bp of DNA at 2,048 bp resolution on the DNALongBench contact-map task. It combines (i) a reverse-complement-equivariant bidirectional state-space (Mamba) encoder over 512 bins, (ii) a multi-scale state mixer, (iii) a symmetric biaffine pairwise head with a learned distance bias, (iv) a **differentiable insulation constraint computed from the predicted map itself**, rather than from an auxiliary head, and (v) a FiLM cell-type adapter enabling few-shot adaptation to a held-out cell line. All hypotheses, metrics, baselines, statistical tests and stopping rules were pre-specified before training.

**Results.** `[TBD — not yet run]`

**Conclusions.** `[TBD]`

**Keywords:** 3D genome; chromatin folding; Hi-C; Micro-C; topologically associating domains; state-space models; Mamba; reverse-complement equivariance; insulation score.

---

## 1. Introduction

The spatial organization of the genome into compartments, topologically associating domains (TADs) and CTCF-anchored loops shapes transcriptional regulation, and its disruption by structural variants causes disease. Hi-C and Micro-C measure this organization genome-wide but are costly per cell type and condition, which motivates models that predict contact maps directly from DNA sequence.

Two lines of work exist. **Task-specific models** — Akita, DeepC, Orca, C.Origami and, most recently, AlphaGenome — map sequence (optionally with epigenomic tracks) to contact maps using convolutional or hybrid convolution–attention encoders. **General long-range DNA models** — HyenaDNA and Caduceus among them — learn sequence representations at up to hundreds of kilobases of context and are fine-tuned for downstream tasks, but reportedly do not match task-specific models on contact-map prediction `[verify against the DNALongBench results]`.

This gap is usually framed as a backbone problem, and the obvious response — replace convolutions or attention with a state-space model — is not by itself a contribution. Caduceus already pairs a bidirectional Mamba with reverse-complement (RC) equivariance for DNA, and ChiMamba reportedly applies a bidirectional Mamba to chromatin-interaction prediction using sequence together with chromatin accessibility `[details to be extracted from the published chapter]`. A paper whose claim is "Mamba instead of attention" is therefore already answered.

We take a different position. At the resolution of this task — 512 bins per megabase window — attention over bins is computationally cheap, so a state-space encoder offers **no efficiency advantage at this scale**, and we make no such claim. What a long-range encoder can offer is a representation in which **structural priors are expressible and enforceable**. Our contribution is accordingly:

1. **An insulation constraint that acts on the predicted map.** Insulation and TAD-boundary scores are computed from the predicted contact map with the same operator applied to the observed map, so the gradient of the structural loss reaches the map itself. A wrong map necessarily yields wrong insulation. This differs from auxiliary structure heads, whose outputs can be correct while the map is not.
2. **Network-wide RC equivariance.** We require f(RC(x)) = rot₁₈₀(f(x)) for the complete network — convolutional encoder, state-space blocks, multi-scale mixer, pairwise head and cell-type adapter — not only for the state-space blocks, and we verify it with a numerical unit test as a precondition for training.
3. **Multi-scale state mixing**, combining shallow, intermediate and deep encoder states so that domain-scale and loop-scale structure are modelled jointly.
4. **Few-shot cell-type adaptation** through a FiLM cell-type embedding, evaluated by leave-one-cell-line-out transfer against training from scratch under an identical compute budget.
5. **A pre-specified analysis plan**, including per-block statistical testing against the strongest baseline in each genomic block, oracle-style reference baselines that quantify how much of a map is cell-type-shared, independent (non-circular) structural evaluation, and explicit stopping rules under which the central claim is withdrawn.

### 1.1 Relation to prior work

| Model | Encoder | Inputs | Window / resolution | Relation to this work |
|---|---|---|---|---|
| Akita (Fudenberg et al. 2020) | CNN + 2D head | Sequence | 1,048,576 bp, 2,048 bp bins, 448×448 | Defines our task setting; primary baseline |
| DeepC (Schwessinger et al. 2020) | CNN | Sequence | Different window/resolution | Included only if fully retrained in our format |
| Orca (Zhou 2022) | Multi-scale CNN | Sequence | Up to chromosome scale, multi-resolution | Already covers multi-megabase scales; we therefore claim megabase scale only |
| C.Origami (Tan et al. 2023) | CNN + Transformer | Sequence + CTCF + ATAC | ~2 Mb, ~8 kb | Cell-type-aware; comparison deferred to the epigenomic-input phase |
| AlphaGenome (2025) | U-Net + Transformer | Sequence | ~1 Mb input, many output tracks | Optional strong baseline |
| HyenaDNA; Caduceus-Ph/PS (Schiff et al. 2024) | Hyena; bidirectional RC-equivariant Mamba | Sequence | Long context | Caduceus supplies the RC-equivariant BiMamba design we build on; it is a baseline, not our novelty |
| ChiMamba | Bidirectional Mamba | Sequence + accessibility | `[to extract]` | Closest architectural precedent; resolved in §2.1 before training |

`[All rows to be verified against primary sources during the literature pass; see §10.]`

---

## 2. Task and data

### 2.1 Prior-art resolution (precondition)

Before any training, the ChiMamba chapter is read in full and its task, inputs, output format, resolution, window size and split are tabulated. If ChiMamba performs sequence-only prediction of contact maps at comparable resolution with a bidirectional Mamba, contribution (1)–(5) above is re-scoped and ChiMamba becomes a required baseline. Otherwise it is positioned as concurrent work on a different task. The outcome is reported either way.

### 2.2 Benchmark

| Item | Specification |
|---|---|
| Benchmark | DNALongBench contact-map task |
| Genome build | hg38 (GRCh38) |
| Input | 1,048,576 bp of DNA sequence, one-hot encoded; sequence only |
| Binning | 512 bins of 2,048 bp; 32 bins cropped from each edge, which serve as sequence context only |
| Output | 448×448 symmetric contact map |
| Cell lines | HFF, H1-hESC, GM12878, IMR-90, HCT116 |
| Split | Official DNALongBench train / validation / test files (region-based) |
| Sample counts | `[verify: reported as 7,008 / 419 / 413]` |
| Target transform | The DNALongBench processed target, used unchanged. Its exact transform is documented from the released files and code before training `[open item]` |

No cell lines beyond these five are used. Any additional line would require matched assay, sequencing depth and processing pipeline, and is deferred.

**Target space.** The target is believed to be a distance-normalized, log-scaled observed/expected map that can take negative values; this is verified directly from the released files before any loss is implemented. The model is trained and evaluated in the benchmark's own target space so that its scores are commensurable with reproduced baselines. No alternative normalization (for example log1p of balanced counts) is substituted, because that would both break comparability and reintroduce distance-decay dominance.

### 2.3 Robustness split

A chromosome-held-out split (test chr1, chr8, chr12; validation chr3, chr6) is reported as a separate robustness analysis with all baselines retrained or fine-tuned. Its numbers are never pooled with the primary benchmark numbers. Window-level random splitting is never used, in any setting, because overlapping windows leak structure between partitions.

---

## 3. Model

Let x ∈ {A,C,G,T}^L with L = 1,048,576, and let c index the cell line. The model outputs Ĉ ∈ ℝ^(448×448), symmetric.

### 3.1 Local sequence encoder

A stack of 1D convolutions (kernel width 5–9) with pooling reduces base-pair resolution to 512 bins of 2,048 bp. Kernels are RC-paired, so complementing and reversing the input permutes and reverses the encoder output rather than changing it. `[Exact channel widths, depth and pooling schedule to be specified.]`

### 3.2 RC-equivariant bidirectional state-space encoder

Eight to twelve blocks operate over the 512-bin sequence. Each block applies a selective state-space (Mamba) scan in the forward direction and in the RC direction **with shared parameters**, following the MambaDNA/Caduceus-PS construction; the RC-path output is mapped back to the original orientation before combination. Each block carries normalization and a feed-forward sublayer.

### 3.3 Multi-scale state mixer

Shallow, intermediate and deep block outputs are combined by pyramidal pooling and returned to 512-bin resolution. Pooling and combination are symmetric under RC. This supplies domain-scale context and loop-scale detail to the pairwise head simultaneously.

### 3.4 Pairwise head

From the mixed representation Z ∈ ℝ^(512×d), two projections U, V ∈ ℝ^(512×d) give

$$\hat{C} = \mathrm{crop}_{32}\!\left(\mathrm{sym}\!\left(\frac{UV^{\top}}{\sqrt{d}} + b_{|i-j|}\right)\right), \qquad \mathrm{sym}(M) = \tfrac{1}{2}(M + M^{\top})$$

with b a learned function of genomic separation. Symmetry is architectural, so no symmetry penalty is required; U and V are RC-paired so that RC of the input yields a 180° rotation of Ĉ.

### 3.5 Cell-type adapter

A learned embedding e_c per cell line produces FiLM parameters γ_c, β_c = MLP(e_c), applied in every state-space block as Z ← γ_c ⊙ Z + β_c. The parameters depend on cell line only, never on strand, and respect the channel pairing that RC equivariance imposes. This adapter is required, not optional: it carries both multi-cell-line training and few-shot adaptation.

### 3.6 Equivariance as a precondition

The full network must satisfy f(RC(x), c) = rot₁₈₀(f(x, c)), where rot₁₈₀(C)_ij = C_{n−1−i, n−1−j}. A unit test asserts this in fp32 (tight tolerance) and in bf16 (loose tolerance), on random and real sequences, in evaluation mode and in a deterministic training mode, with and without gradient checkpointing. **Training does not begin until it passes.** The symmetric edge crop preserves the identity.

---

## 4. Training and analysis plan

### 4.1 Objective

$$\mathcal{L} = \mathcal{L}_{\text{contact}} + \lambda_{\text{ins}}\mathcal{L}_{\text{ins}} + \lambda_{\text{bnd}}\mathcal{L}_{\text{bnd}} + \lambda_{\text{loop}}\mathcal{L}_{\text{loop}}$$

ℒ_contact is a masked error on the benchmark target; masked, blacklisted and unbalanced bins and the excluded near-diagonal band never contribute to any loss or metric.

**Differentiable insulation.** A single differentiable transform g maps both the prediction and the target to a contact-like scale, **applied identically to both** so that the two insulation values are commensurable:

$$\tilde{C} = g(\hat{C}), \qquad \tilde{C}^{\text{obs}} = g(C^{\text{target}})$$

For each bin i, insulation averages contacts that **cross** i — the upstream flank against the downstream flank, excluding the diagonal:

$$\hat{I}_i = \frac{1}{|\Omega_i|}\sum_{(a,b)\in\Omega_i} \tilde{C}_{ab}, \qquad \Omega_i = \{(a,b) : i-w \le a < i < b \le i+w,\ a,b \text{ unmasked}\}$$

with I_i^obs the same operator on C̃^obs, and ℒ_ins = mean_i (Î_i − I_i^obs)². Default w = 20 bins (40 kb), with sensitivity reported at w ∈ {10, 20, 40}; insulation is defined for i ∈ [w, 448−w). ℒ_bnd acts on boundary scores derived from Î `[exact form to be specified]`. ℒ_loop uses called loops as positives against distance-matched negatives, with supervision labels drawn only from training regions.

Because g is a smooth function of Ĉ and the operator is linear in C̃, the structural gradient flows into the map. No auxiliary energy head is used: in a log-ratio target space an "energy" is a relabelling of the map and adds no information.

### 4.2 Optimization

AdamW (weight decay 10⁻²), learning rate 3×10⁻⁴ with 1,000-step warmup and cosine decay; bf16 mixed precision, chosen for kernel compatibility and practical stability rather than on any theoretical stability claim; EMA weights; gradient checkpointing. Batch size is set by memory profiling at 1, 2 and 4, with gradient accumulation to an effective 8–16 windows. Checkpoints are selected on mean validation SCC. Training uses the DNALongBench training split only; no validation or test region enters training or pretraining at any stage.

The structural terms are introduced after ℒ_contact has converged. The no-structure control receives **the same total number of optimizer steps**, so that the comparison isolates the constraint rather than additional training.

A single model configuration is designated the paper configuration before training, and all baselines are matched to it. Results from a reduced fallback configuration are never pooled with it.

### 4.3 Metrics

The primary metric is **SCC, the stratum-adjusted correlation coefficient** of HiCRep: a Pearson correlation computed within each genomic-distance stratum and aggregated with stratum weights, which removes the inflation that distance decay produces in a naive whole-map correlation. SCC is computed per window with the same implementation and parameters as the benchmark. PCC is always reported alongside it. Spearman correlation, where reported, is named as such and never abbreviated to SCC.

Secondary: MSE and MAE; SCC within distance strata 10–100 kb, 100–500 kb and 0.5–≈0.9 Mb (the maximum separation inside a 448-bin window is ≈896 kb); insulation correlation, boundary F1 and AUROC, and the directionality index; loop average precision, AUROC, and precision/recall within a 1–2 bin radius; and, for the long-context phase only, inference time, peak memory and parameter count.

### 4.4 Baselines

| Baseline | Definition | Role |
|---|---|---|
| Global cell-line mean | Mean training map for that cell line | Lower bound |
| Locus-shared mean | Mean observed map of the **other** cell lines at the same locus | Oracle-style reference: how much of a map is cell-type-shared |
| Locus-shared mean + cell offset | Above, plus a learned per-cell-line offset | Oracle-style reference |
| Akita | Reproduced on the official split | Primary comparison |
| Caduceus-Ph/PS | With a contact head, as in the benchmark | Architectural comparison |
| Matched Transformer | Attention over bins; same parameter count, data and schedule | Isolates the state-space encoder |
| Strong CNN | Large-receptive-field convolutional model | Task-specific comparison |
| HyenaDNA; DeepC; ChiMamba | Included subject to pipeline compatibility or full retraining in our format | Optional |

The two locus-shared references consume **measured** maps from the evaluation regions of other cell lines. They are reported separately and are excluded from the comparison set used for the primary test, which contains only sequence-based trained models; otherwise a sequence model would be tested against measurements rather than against a predictor.

Baselines we retrain receive the same tuning budget, EMA and checkpoint-selection procedure as our model, and the budget is reported.

### 4.5 Statistical plan

Pre-specified, one-sided, and fixed before results are seen:

1. Compute SCC per test window for each model and each of 3 seeds.
2. Average over seeds per model and window; then average within non-overlapping 1 Mb genomic blocks. Blocks, not windows, are the unit of resampling, because neighbouring windows overlap and are not independent.
3. Per block b, take the difference against the strongest baseline **in that block**: Δ_b = SCC_ours,b − max_{m ∈ ℬ} SCC_m,b. Taking the per-block maximum prevents post-hoc baseline selection.
4. Paired block bootstrap, 10,000 resamples, over blocks.
5. Test H₀: Δ ≤ 0 against H₁: Δ > 0 at **one-sided α = 0.025**, the level equivalent to a two-sided test at 0.05.
6. Report mean ΔSCC, its 97.5% one-sided confidence bound, the p-value, and per-seed values in the appendix. Where several comparisons form one family, apply Holm correction within that family and state its membership.

A per-window Wilcoxon test is reported only as a secondary, anti-conservative check, never as the primary test.

### 4.6 Hypotheses and success criteria

| ID | Hypothesis | Criterion |
|---|---|---|
| H1 | HiFold-Mamba exceeds the strongest trained baseline in contact-map accuracy | Mean block-wise ΔSCC ≥ +0.01 with a positive one-sided bound under §4.5 |
| H2 | The insulation constraint improves recovery of TADs and loops | ≥ +0.02 AUROC on **independent** boundaries and ≥ +0.02 loop AP, both with bootstrap bounds excluding 0, and no meaningful accuracy loss (lower bound of ΔSCC > −0.005) under equal compute |
| H3 | Pretraining on four cell lines plus few-shot adaptation beats training from scratch on the same limited data | Δ_H3 = SCC_pretrain+few-shot − SCC_scratch > 0 under equal compute and identical fine-tuning data |
| H4 | The model tracks the direction of structural-variant effects on domains and loops | Agreement in sign, and correlation in magnitude, of predicted versus observed changes — **only** where observed before/after maps exist |

**Avoiding circularity in H2.** Training supervises insulation, so boundaries derived from insulation cannot serve as the main evidence; they are reported as internal consistency only. Primary H2 evidence uses boundaries defined independently of both the model and the training supervision — convergent CTCF-bound sites from independent ChIP-seq, or annotations from a different replicate or study — with the source fixed before evaluation. Loop evaluation likewise uses labels from a source independent of the training labels.

**H3 design.** One cell line (for example IMR-90) is withheld from pretraining; the model is pretrained on the remaining four, then fine-tuned on limited data from the withheld line and evaluated on the **official test regions** of that line, which pretraining never saw. Chromosome-held-out evaluation is not used here, because those chromosomes are partly inside the official training split. Limited-data regimes are contiguous subsets of the withheld line's training regions (10%, 25%, 50%) against its full training regions; a read-depth-limited regime requires reprocessing from read pairs and is reported only if that reprocessing reproduces the released targets at full depth. The hold-out is rotated over at least two cell lines, so that no conclusion rests on a single line.

**H4 scope.** H4 is tested only on structural variants with published before/after contact maps. If fewer than ten such cases are assembled, the hypothesis is restricted to an in-silico analysis, labelled as such, whose claim is that the model reproduces structural consequences consistent with patterns learned from Hi-C — not that it predicts a measured biological effect. If no experimental cases exist, variant analysis moves to the Discussion as a qualitative demonstration.

### 4.7 Ablations

Matched Transformer in place of the state-space encoder; RC equivariance removed; structural loss removed under equal compute, and each term removed individually; multi-scale mixer removed; insulation window w ∈ {10, 20, 40}; and optionally the distance bias removed, and a symmetric RC-equivariant 2D refinement added after the pairwise head.

### 4.8 Pre-specified stopping rules

| Condition | Consequence |
|---|---|
| Accuracy below the convolutional and state-space baselines after a bounded development period | Diagnose data, target transform and splits before touching the architecture; if unresolved, re-scope the claim to structure and variant effects |
| Structural loss yields no significant structural gain | Remove every structure-constrained claim, including from the title |
| Model fails to beat the mean-map references | The approach has not demonstrated locus-specific learning and the central claim is withdrawn |
| Fewer than ten experimentally resolved variants | H4 restricted per §4.6 |
| ΔSCC < +0.01 **and** no structural gain after the full ablation | The principal claim is withdrawn; see §8 |

---

## 5. Results

**Not yet run.** Each table is filled only from logged runs, with the config hash, commit id, seeds and date recorded alongside. Cells remain `[TBD]` until then. Negative and null outcomes are reported with the same prominence as positive ones.

### 5.1 Primary comparison (official split)

| Model | SCC (mean ± SD, 3 seeds) | PCC | ΔSCC vs. per-block best | 97.5% one-sided bound | p |
|---|---|---|---|---|---|
| Global cell-line mean | [TBD] | [TBD] | — | — | — |
| Locus-shared mean *(reference, excluded from ℬ)* | [TBD] | [TBD] | — | — | — |
| Locus-shared mean + cell offset *(reference)* | [TBD] | [TBD] | — | — | — |
| Strong CNN | [TBD] | [TBD] | — | — | — |
| Akita (reproduced) | [TBD] | [TBD] | — | — | — |
| Caduceus-Ph/PS | [TBD] | [TBD] | — | — | — |
| Matched Transformer | [TBD] | [TBD] | — | — | — |
| **HiFold-Mamba** | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

Reproduction check: our reproduced Akita and Caduceus scores against their published benchmark values — `[TBD]`.

### 5.2 Distance-stratified accuracy

| Model | 10–100 kb | 100–500 kb | 0.5–≈0.9 Mb |
|---|---|---|---|
| *(all rows)* | [TBD] | [TBD] | [TBD] |

### 5.3 Structural recovery and the insulation constraint (H2)

| Model | Independent boundary AUROC | Boundary F1 | Insulation r | Loop AP | SCC (non-inferiority) |
|---|---|---|---|---|---|
| Base (no structural loss, equal steps) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **+ insulation constraint** | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

Insulation-derived boundaries (internal consistency only): `[TBD]`. Window sensitivity w ∈ {10, 20, 40}: `[TBD]`.

### 5.4 Ablations

| Variant | SCC | Δ vs. full | Independent boundary AUROC | Loop AP |
|---|---|---|---|---|
| Full model | [TBD] | — | [TBD] | [TBD] |
| Transformer encoder (matched) | [TBD] | [TBD] | [TBD] | [TBD] |
| − RC equivariance | [TBD] | [TBD] | [TBD] | [TBD] |
| − structural loss (equal steps) | [TBD] | [TBD] | [TBD] | [TBD] |
| − multi-scale mixer | [TBD] | [TBD] | [TBD] | [TBD] |
| − distance bias | [TBD] | [TBD] | [TBD] | [TBD] |

Equivariance unit test: `[pass/fail, per §3.6]`.

### 5.5 Few-shot cell-type adaptation (H3)

| Held-out line | Regime | Scratch SCC | Pretrain + few-shot SCC | Δ_H3 |
|---|---|---|---|---|
| [line 1] | 10% / 25% / 50% / full | [TBD] | [TBD] | [TBD] |
| [line 2] | 10% / 25% / 50% / full | [TBD] | [TBD] | [TBD] |

### 5.6 Structural variants (H4)

Assembled cases with before/after maps: `[count TBD]`. Scope decision per §4.6: `[TBD]`.

| Case | Variant | ΔSCC | Δinsulation | Δloop score | Sign agreement |
|---|---|---|---|---|---|
| [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

### 5.7 Robustness: chromosome-held-out split

Reported separately; never pooled with §5.1. `[TBD]`

### 5.8 Efficiency

Reported for the long-context and high-resolution phases only. At 512 bins no efficiency advantage is claimed, and none is reported as if it were a finding. `[TBD]`

---

## 6. Discussion

`[TBD — written after §5.]`

Points to address once results exist: whether structural constraint rather than encoder choice accounts for any gain; how far the locus-shared references bound what sequence-only prediction can add; what the distance-stratified profile implies about genuine long-range dependence; and limitations.

---

## 7. Limitations

Stated now, independently of outcome.

1. **Scale.** The window is ≈1 Mb. The work is megabase-scale; "multi-megabase" is not claimed, and Orca already operates at larger scales.
2. **No efficiency claim at this resolution.** With 512 bins, attention over bins is inexpensive; any efficiency argument requires the longer-context or finer-token phase and is deferred to it.
3. **Sequence-only.** Without cell-type-specific inputs the model cannot predict an unseen cell line zero-shot; H3 is explicitly few-shot. Zero-shot generalization needs epigenomic inputs and is future work.
4. **Architectural precedent.** The RC-equivariant bidirectional state-space design follows Caduceus; the novelty claimed here is the structural constraint, multi-scale mixing, adaptation scheme and analysis protocol, not the encoder family.
5. **Structural ground truth.** Boundary and loop annotations derive from Hi-C data related to the training targets; independent sources mitigate but do not eliminate this.
6. **Variant evidence.** Experimentally resolved variants with before/after maps are scarce, so H4 may remain in-silico and qualified.
7. **Five cell lines** bound how strongly cross-cell-type conclusions can be drawn.
8. **Assay heterogeneity.** The five lines span Hi-C and Micro-C protocols; the benchmark's processing is inherited as-is and protocol effects are not separately modelled in this phase.

---

## 8. Conditional title and claim scope

The title is contingent on outcomes, fixed in advance to prevent retrofitting:

- Structural constraint significant → *Insulation-Constrained Multi-Scale State-Space Modeling for Megabase-Scale Prediction of 3D Chromatin Folding*
- Accuracy gain without structural gain → *Multi-Scale State-Space Modeling of Chromatin Contact Maps from DNA Sequence*
- Adaptation result carries the paper → *Few-Shot Cell-Type Adaptation for Sequence-Based Prediction of 3D Chromatin Folding*
- Neither accuracy nor structural criterion met → the central claim is withdrawn; the work is reconsidered rather than reframed as a benchmark or tooling paper.

No variant of the title asserts a physical or energy-based model, since no energy head is used. No variant rests on bidirectionality with RC equivariance alone, since that is Caduceus's contribution.

---

## 9. Statements

**Data availability.** All primary data are public: the DNALongBench contact-map task; Hi-C and Micro-C from 4D Nucleome and related studies; ENCODE ChIP-seq, ATAC-seq and histone-mark data. Accessions, assay, depth, build, licence and access date are recorded in the repository manifest. No individual-level human genotype data are used; the work uses reference sequence and cell-line data only.

**Code availability.** https://github.com/malekpouri/HiFold-Mamba — model, training and evaluation code, configurations, the pre-specified protocol, and the equivariance test.

**Reproducibility.** Every reported number is reproducible from a committed configuration and a recorded commit; three seeds per model; seeds, hardware and software versions logged.

**CRediT.** `[To be agreed with all authors before submission.]` Mohammad Malekpouri: conceptualization, methodology, software, formal analysis, investigation, writing — original draft, visualization. Somayeh Lotfi: supervision, methodology, validation, writing — review & editing.

**Competing interests.** `[TBD]`

**Funding.** `[TBD]`

**Use of AI tools.** AI assistance (Claude, Anthropic) was used for code development, protocol drafting and manuscript preparation. The authors are responsible for all content; every claim, citation and number was verified by the authors. The target journal's disclosure requirements will be followed.

---

## 10. Outstanding items before submission

1. **Target transform** — document the benchmark's exact transform from the released files and code; confirm the transform g used for insulation, applied identically to prediction and target.
2. **ChiMamba** — read the published chapter; tabulate task, inputs, output, resolution and split; decide baseline versus related work.
3. **Verify** every prior-art row, the benchmark's cell lines, split files, sample counts and SCC parameters, and each citation against its own source (one claim, one source — no mismatched links).
4. **Independent boundary and loop sources** — fix before evaluation.
5. **Specify** the convolutional downsampling stack, the boundary-loss form, the RC pairing of pairwise-head and FiLM channels, and all hyperparameters.
6. **Designate** the paper configuration and compute environment.
7. **Confirm** authorship, order and CRediT with all authors.
8. **Target journal** — select after results; format accordingly.
