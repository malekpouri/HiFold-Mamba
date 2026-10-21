# HiFold-Mamba: Insulation-Constrained Multi-Scale State-Space Modeling for Megabase-Scale Prediction of 3D Chromatin Folding

> **Draft status — protocol / registered-report stage.** Methods and the analysis plan are complete and pre-specified. **Results and Discussion are deliberately empty**: no experiment has been run. Every table in §5 is a placeholder with explicit `[TBD]` cells. Nothing in this file may be filled in by estimation, extrapolation or expectation — only from logged experimental runs with a config hash and a commit id. Title is conditional (see §8).

**Authors**

Mohammad Malekpouri^a,\*^ · Somayeh Lotfi^a^

^a^ Department of Computer Engineering, BA.C., Islamic Azad University, Bandar Abbas, Iran

\* Corresponding author. Mohammad Malekpouri — ORCID [0009-0001-6278-6163](https://orcid.org/0009-0001-6278-6163)
Somayeh Lotfi — ORCID [0000-0002-7218-5388](https://orcid.org/0000-0002-7218-5388)

**Code availability:** https://github.com/malekpouri/HiFold-Mamba (protocol only; no code written yet)

---

## Abstract

*(Written last. Results sentences stay blank until §5 is filled.)*

**Background.** Sequence-to-contact-map models such as Akita and DeepC predict Hi-C and Micro-C maps at roughly megabase scale with convolutional architectures. On the DNALongBench contact-map task, general long-range DNA models (HyenaDNA, Caduceus) fall well short of the task-specific Akita model. The open problem is not the choice of sequence backbone but the combination of an efficient long-range encoder, structural inductive bias, and generalization across cell types and sequence variants.

**Methods.** We present HiFold-Mamba, a sequence-only model that predicts 448×448 contact maps from 1,048,576 bp of DNA at 2,048 bp resolution on the DNALongBench contact-map task. It combines (i) a reverse-complement-equivariant bidirectional state-space (Mamba) encoder over 512 bins, (ii) a multi-scale state mixer, (iii) a symmetric biaffine pairwise head with a learned distance bias, (iv) a **differentiable insulation constraint computed from the predicted map itself**, rather than from an auxiliary head, and (v) a FiLM cell-type adapter enabling few-shot adaptation to cell lines never seen in training. The model is trained on five cell lines; the benchmark's four remaining lines are reserved for few-shot evaluation. All hypotheses, metrics, baselines, statistical tests and stopping rules were pre-specified before training.

**Results.** `[TBD — not yet run]`

**Conclusions.** `[TBD]`

**Keywords:** 3D genome; chromatin folding; Hi-C; Micro-C; topologically associating domains; state-space models; Mamba; reverse-complement equivariance; insulation score.

---

## 1. Introduction

The spatial organization of the genome into compartments, topologically associating domains (TADs) and CTCF-anchored loops shapes transcriptional regulation, and its disruption by structural variants causes disease. Hi-C and Micro-C measure this organization genome-wide but are costly per cell type and condition, which motivates models that predict contact maps directly from DNA sequence.

Two lines of work exist. **Task-specific models** map sequence (optionally with epigenomic tracks) to contact maps using convolutional or hybrid convolution–attention encoders: Akita (Fudenberg et al. 2020), DeepC (Schwessinger et al. 2020), Orca (Zhou 2022), C.Origami (Tan et al. 2023) and, most recently, AlphaGenome (Avsec et al. 2026). **General long-range DNA models**, among them HyenaDNA (Nguyen et al. 2023) and Caduceus (Schiff et al. 2024), learn sequence representations over long contexts and are fine-tuned for downstream tasks. On the DNALongBench contact-map task, they score roughly half as well as the task-specific model: an average of 0.115–0.133 for HyenaDNA and the two Caduceus variants against 0.230 for Akita (Cheng et al. 2025, preprint values).

This gap is usually framed as a backbone problem, and the obvious response — replace convolutions or attention with a state-space model (Gu & Dao 2023) — is not by itself a contribution. Caduceus already pairs a bidirectional Mamba with reverse-complement (RC) equivariance for DNA. ChiMamba (Zhang et al. 2024) applies a bidirectional Mamba to chromatin-interaction prediction, using sequence together with chromatin accessibility in six human cell lines, according to its abstract `[full chapter to be read; see §2.1]`. A paper whose claim is "Mamba instead of attention" is therefore already answered.

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
| AlphaGenome (Avsec et al. 2026) | U-Net + Transformer | Sequence | 1 Mb input, many output tracks | Optional strong baseline |
| HyenaDNA (Nguyen et al. 2023); Caduceus-Ph/PS (Schiff et al. 2024) | Hyena; bidirectional RC-equivariant Mamba | Sequence | Long context | Caduceus supplies the RC-equivariant BiMamba design we build on; it is a baseline, not our novelty |
| ChiMamba (Zhang et al. 2024) | Bidirectional Mamba | Sequence + accessibility | `[to extract]` | Closest architectural precedent; resolved in §2.1 before training |

Citations, the Akita and DNALongBench task settings, and AlphaGenome's input length are verified against primary sources. Still to verify: the encoder and window entries for DeepC, Orca and C.Origami, and all ChiMamba details.

---

## 2. Task and data

### 2.1 Prior-art resolution (precondition)

Before any training, the ChiMamba chapter is read in full and its task, inputs, output format, resolution, window size and split are tabulated. If ChiMamba performs sequence-only prediction of contact maps at comparable resolution with a bidirectional Mamba, contribution (1)–(5) above is re-scoped and ChiMamba becomes a required baseline. Otherwise it is positioned as concurrent work on a different task. The outcome is reported either way.

### 2.2 Benchmark

| Item | Specification |
|---|---|
| Benchmark | DNALongBench contact-map task (Cheng et al. 2025), built on the Akita setting (Fudenberg et al. 2020) |
| Genome build | hg38 (GRCh38) |
| Input | 1,048,576 bp of DNA sequence, one-hot encoded; sequence only |
| Binning | 512 bins of 2,048 bp; 32 bins cropped from each edge, which serve as sequence context only |
| Output | 448×448 symmetric contact map; scored on the upper triangle with diagonal offset 2 (99,681 values) |
| Training cell lines | HFF and H1-hESC (Micro-C; Krietenstein et al. 2020), GM12878 and IMR-90 (Hi-C; Rao et al. 2014), HCT116 (Hi-C; Rao et al. 2017) |
| Few-shot held-out lines | HAP1, HeLa, HepG2, K562: the benchmark's remaining four lines, never used in training or pretraining |
| Split | Official DNALongBench train / validation / test files: non-overlapping virtual contigs assigned 8:1:1 |
| Sample counts | 7,008 / 419 / 413 sequences |
| Target transform | The DNALongBench processed target, used unchanged (below) |

Baseline reproductions are compared with published values per cell line for the five training lines, because the benchmark's averages pool all nine lines.

**Target space.** Following Akita, contact matrices are ICE-balanced, adaptively coarse-grained, normalized for distance-dependent decay, natural-log transformed, clipped to (−2, 2), linearly interpolated over missing bins and smoothed with a 2D Gaussian (σ = 1). The result is a log observed/expected map that takes negative values. Before any loss is implemented, the transform, value range, any shipped masks and the diagonal offset are confirmed in the released files. The model is trained and evaluated in the benchmark's own target space so that its scores are commensurable with reproduced baselines. No alternative normalization (for example log1p of balanced counts) is substituted, because that would both break comparability and reintroduce distance-decay dominance.

### 2.3 Robustness split

A chromosome-held-out split (test chr1, chr8, chr12; validation chr3, chr6) is reported as a separate robustness analysis with all baselines retrained on that split. Its numbers are never pooled with the primary benchmark numbers. Window-level random splitting is never used, in any setting, because overlapping windows leak structure between partitions.

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

The full network must satisfy f(RC(x), c) = rot₁₈₀(f(x, c)), where rot₁₈₀(C)_ij = C_{n−1−i, n−1−j}. A unit test asserts this in fp32 (absolute tolerance 10⁻⁴, relative 10⁻⁵) and in bf16 (relative tolerance 10⁻²), on random and real sequences, in evaluation mode and in a deterministic training mode, with and without gradient checkpointing. **Training does not begin until it passes.** The symmetric edge crop preserves the identity.

---

## 4. Training and analysis plan

### 4.1 Objective

$$\mathcal{L} = \mathcal{L}_{\text{contact}} + \lambda_{\text{ins}}\mathcal{L}_{\text{ins}} + \lambda_{\text{bnd}}\mathcal{L}_{\text{bnd}} + \lambda_{\text{loop}}\mathcal{L}_{\text{loop}}$$

ℒ_contact is a masked error on the benchmark target; masked, blacklisted and unbalanced bins and the excluded near-diagonal band never contribute to any loss or metric.

**Differentiable insulation.** A single differentiable transform g maps both the prediction and the target to a contact-like scale, **applied identically to both** so that the two insulation values are commensurable:

$$\tilde{C} = g(\hat{C}), \qquad \tilde{C}^{\text{obs}} = g(C^{\text{target}}), \qquad g(x) = \exp\!\big(2\tanh(x/2)\big)$$

g is a soft-clipped exponential: it is close to exp near zero, maps both maps back towards the observed/expected scale, and is bounded to [e⁻², e²], the range implied by the target's clip.

For each bin i, insulation averages contacts that **cross** i — the upstream flank against the downstream flank, excluding the diagonal and the benchmark's excluded near-diagonal band (d_min = 2 bins):

$$\hat{I}_i = \frac{1}{|\Omega_i|}\sum_{(a,b)\in\Omega_i} \tilde{C}_{ab}, \qquad \Omega_i = \{(a,b) : i-w \le a < i < b \le i+w,\ b-a \ge d_{\min},\ a,b \text{ unmasked}\}$$

with I_i^obs the same operator on C̃^obs, and ℒ_ins the mean of (Î_i − I_i^obs)² over valid bins and windows. Default w = 20 bins (40 kb), with sensitivity reported at w ∈ {10, 20, 40}; insulation is defined for i ∈ [w, 448−w). ℒ_bnd acts on boundary scores derived from Î `[exact form to be specified]`. ℒ_loop uses called loops as positives against distance-matched negatives, with supervision labels drawn only from training regions. The insulation operator follows the crossing-window idea of the insulation score (Crane et al. 2015).

Because g is a smooth function of Ĉ and the operator is linear in C̃, the structural gradient flows into the map. No auxiliary energy head is used: in a log-ratio target space an "energy" is a relabelling of the map and adds no information.

### 4.2 Optimization

AdamW (weight decay 10⁻²), learning rate 3×10⁻⁴ with 1,000-step warmup and cosine decay; bf16 mixed precision, chosen for kernel compatibility and practical stability rather than on any theoretical stability claim; EMA weights; gradient checkpointing. Batch size is set by memory profiling at 1, 2 and 4, with gradient accumulation to an effective 8–16 windows. Checkpoints are selected on mean validation SCC. Training uses the DNALongBench training split only; no validation or test region enters training or pretraining at any stage.

The structural terms are introduced after ℒ_contact has converged. The no-structure control receives **the same total number of optimizer steps**, so that the comparison isolates the constraint rather than additional training.

The paper configuration, sized for a 24–40 GB accelerator, is fixed before training, and all baselines are matched to it. A reduced configuration that fits 8 GB is used for development only and is never reported.

### 4.3 Metrics

The primary metric is **SCC, the stratum-adjusted correlation coefficient** of HiCRep (Yang et al. 2017): a Pearson correlation computed within each genomic-distance stratum and aggregated with stratum weights, which removes the inflation that distance decay produces in a naive whole-map correlation. SCC is computed per window with the same implementation and parameters as the benchmark. PCC is always reported alongside it. Spearman correlation, where reported, is named as such and never abbreviated to SCC.

Secondary: MSE and MAE; SCC within distance strata 10–100 kb, 100–500 kb and 0.5–0.92 Mb (the maximum separation inside a 448-bin window is 447 × 2,048 bp ≈ 915 kb); insulation correlation, boundary F1 and AUROC, and the directionality index; loop average precision, AUROC, and precision/recall within a 1–2 bin radius; and, for the long-context phase only, inference time, peak memory and parameter count.

### 4.4 Baselines

| Baseline | Definition | Role |
|---|---|---|
| Global cell-line mean | Mean training map for that cell line | Lower bound; the minimum bar of the stopping rule (§4.8) |
| Locus-shared mean | Mean observed map of the **other** cell lines at the same locus | Oracle-style reference: how much of a map is cell-type-shared |
| Locus-shared mean + cell offset | Above, plus a learned per-cell-line offset | Oracle-style reference |
| Akita | Reproduced on the official split | Primary comparison |
| Caduceus-Ph/PS | With a contact head, as in the benchmark | Architectural comparison |
| Matched Transformer | Attention over bins; same parameter count, data and schedule | Isolates the state-space encoder |
| Strong CNN | Large-receptive-field convolutional model | Task-specific comparison |
| HyenaDNA; DeepC; ChiMamba | Included subject to pipeline compatibility or full retraining in our format | Optional |

The two locus-shared references consume **measured** maps from the evaluation regions of other cell lines. They are reported separately as context for how cell-specific the maps are. They are excluded from the comparison set ℬ used for the primary test, which contains only sequence-based trained models, and they are never a stopping condition; otherwise a sequence model would be tested against measurements rather than against a predictor. The global cell-line mean is not in ℬ either.

Baselines we retrain receive the same tuning budget, EMA and checkpoint-selection procedure as our model, and the budget is reported.

### 4.5 Statistical plan

Pre-specified, one-sided, and fixed before results are seen:

1. Compute SCC per test window for each model and each of 3 seeds.
2. Average over seeds per model and window; then average within non-overlapping 1 Mb genomic blocks. Blocks, not windows, are the unit of resampling, because neighbouring windows overlap and are not independent.
3. Per block b, take the difference against the strongest baseline **in that block**: Δ_b = SCC_ours,b − max_{m ∈ ℬ} SCC_m,b. Taking the per-block maximum prevents post-hoc baseline selection.
4. Paired block bootstrap, 10,000 resamples, over blocks.
5. Test H₀: Δ ≤ 0 against H₁: Δ > 0 at **one-sided α = 0.025**, the level equivalent to a two-sided test at 0.05.
6. Report mean ΔSCC, the 95% percentile bootstrap interval (its lower limit is the 97.5% one-sided confidence bound), the one-sided p-value, and per-seed values in the appendix.

The same procedure and level apply to every other pre-specified comparison (H2, H3, the multi-scale-mixer ablation and the global-mean check of §4.8). H2's conditions must hold jointly, so they need no multiplicity correction. For H3, the three limited-data regimes within each held-out line are Holm-corrected.

A per-window Wilcoxon test is reported only as a secondary, anti-conservative check, never as the primary test.

### 4.6 Hypotheses and success criteria

| ID | Hypothesis | Criterion |
|---|---|---|
| H1 | HiFold-Mamba exceeds the strongest trained baseline in contact-map accuracy | Mean block-wise ΔSCC ≥ +0.01 with a positive one-sided bound under §4.5 |
| H2 | The insulation constraint improves recovery of TADs and loops | ≥ +0.02 AUROC on **independent** boundaries and ≥ +0.02 loop AP, both with bootstrap bounds excluding 0, and no meaningful accuracy loss (lower bound of ΔSCC > −0.005) under equal compute |
| H3 | Pretraining on the five training lines plus few-shot adaptation beats training from scratch on the same limited data | Δ_H3 = SCC_pretrain+few-shot − SCC_scratch with a positive one-sided bound in at least one limited regime (Holm-corrected over the three), in at least two of the four held-out lines, under equal compute and identical fine-tuning data |
| H4 | The model tracks the direction of structural-variant effects on domains and loops | Agreement in sign, and correlation in magnitude, of predicted versus observed changes — **only** where observed before/after maps exist |

**Avoiding circularity in H2.** Training supervises insulation, so boundaries derived from insulation cannot serve as the main evidence; they are reported as internal consistency only. Primary H2 evidence uses boundaries called, with a caller and parameters fixed in advance, from an independent Hi-C or Micro-C dataset of the same cell line (a different study or biological replicate from the one behind the benchmark target). CTCF ChIP-seq peaks containing a CTCF motif form a secondary, assay-orthogonal set. Loop evaluation likewise uses loops from an independent dataset; loop calls derived from the training data are excluded. All accessions are fixed before evaluation.

**H3 design.** The four benchmark lines never used in training (HAP1, HeLa, HepG2, K562) serve as held-out lines. The main five-line model is fine-tuned on limited data from one held-out line at a time, learning a new FiLM embedding, and is evaluated on that line's **official test regions**, which no training stage saw. The scratch arm trains the same architecture from random initialization on the same data for the same number of steps. Chromosome-held-out evaluation is not used here, because those chromosomes are partly inside the official training split. Limited-data regimes are contiguous-block subsets of the withheld line's training regions (10%, 25%, 50%), with its full training regions reported alongside. A read-depth-limited regime would require reprocessing from read pairs and is deferred to a later phase. All four held-out lines are tested, and success requires at least two, so that no conclusion rests on a single line.

**H4 scope.** H4 is tested only on structural variants with published before/after contact maps. If fewer than ten such cases are assembled, the hypothesis is restricted to an in-silico analysis, labelled as such, whose claim is that the model reproduces structural consequences consistent with patterns learned from Hi-C — not that it predicts a measured biological effect. If no experimental cases exist, variant analysis moves to the Discussion as a qualitative demonstration.

### 4.7 Ablations

Matched Transformer in place of the state-space encoder; RC equivariance removed; structural loss removed under equal compute, and each term removed individually; multi-scale mixer removed; insulation window w ∈ {10, 20, 40}; and optionally the distance bias removed, and a symmetric RC-equivariant 2D refinement added after the pairwise head.

### 4.8 Pre-specified stopping rules

| Condition | Consequence |
|---|---|
| Accuracy below the convolutional and state-space baselines after a bounded development period | Diagnose data, target transform and splits before touching the architecture; if unresolved, re-scope the claim to whichever narrative of §8 can still hold |
| Structural loss yields no significant structural gain (H2 not met) | Remove every structure-constrained claim, including from the title |
| Model fails to beat the global cell-line mean (§4.5 procedure against that baseline alone) | The approach has not demonstrated locus-specific learning and the project fails in its current form. The locus-shared references are context only and are not part of this rule |
| Fewer than ten experimentally resolved variants | H4 restricted per §4.6 |
| None of the three narratives of §8 meets its criterion | The project is stopped and redesigned; see §8 |

---

## 5. Results

**Not yet run.** Each table is filled only from logged runs, with the config hash, commit id, seeds and date recorded alongside. Cells remain `[TBD]` until then. Negative and null outcomes are reported with the same prominence as positive ones.

### 5.1 Primary comparison (official split)

| Model | SCC (mean ± SD, 3 seeds) | PCC | ΔSCC vs. per-block best | 97.5% one-sided bound | p |
|---|---|---|---|---|---|
| Global cell-line mean *(reference, excluded from ℬ)* | [TBD] | [TBD] | — | — | — |
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
7. **Nine cell lines in total** (five for training, four held out) bound how strongly cross-cell-type conclusions can be drawn.
8. **Assay and processing heterogeneity.** The training lines span Hi-C and Micro-C protocols, and the held-out lines were added to the benchmark by its authors, not by Akita's original processing. The benchmark's processing is inherited as-is, and protocol effects are not separately modelled in this phase.

---

## 8. Conditional title and claim scope

The title is contingent on outcomes, fixed in advance to prevent retrofitting. Three narratives each have a pre-specified criterion:

- **N-I, insulation-constrained:** H2 met.
- **N-M, multi-scale:** H1 met **and** removing the multi-scale mixer significantly lowers SCC (§4.5 procedure, full model vs. no-mixer ablation).
- **N-F, few-shot adaptation:** H3 met.

| Narratives met | Title |
|---|---|
| N-I and N-M | *Insulation-Constrained Multi-Scale State-Space Modeling for Megabase-Scale Prediction of 3D Chromatin Folding* (working title) |
| N-I only | *Insulation-Constrained Sequence-to-Contact-Map Prediction at Megabase Scale* |
| N-M only | *Multi-Scale State-Space Modeling of Chromatin Contact Maps from DNA Sequence* |
| N-F only | *Few-Shot Cell-Type Adaptation for Sequence-Based Prediction of 3D Chromatin Folding* |

If N-F holds together with N-I or N-M, the title follows N-I/N-M and the adaptation result is reported as secondary. If none of the three holds, the project is stopped and redesigned rather than reframed as a benchmark, tooling or negative-result paper.

No variant of the title asserts a physical or energy-based model, since no energy head is used. No variant rests on bidirectionality with RC equivariance alone, since that is Caduceus's contribution.

---

## 9. Statements

**Data availability.** All primary data are public: the DNALongBench contact-map task; for structural evaluation, Hi-C and Micro-C from 4D Nucleome and related studies and ENCODE CTCF ChIP-seq and ATAC-seq. Accessions, assay, depth, build, licence and access date will be recorded in the repository data manifest (`docs/data_manifest.md`) as each dataset is obtained. No individual-level human genotype data are used; the work uses reference sequence and cell-line data only.

**Code availability.** https://github.com/malekpouri/HiFold-Mamba — currently holds the pre-specified protocol. Model, training and evaluation code, configurations and the equivariance test will be added there; no code has been written yet.

**Reproducibility.** Every reported number is reproducible from a committed configuration and a recorded commit; three seeds per model; seeds, hardware and software versions logged.

**CRediT.** `[To be agreed with all authors before submission.]` Mohammad Malekpouri: conceptualization, methodology, software, formal analysis, investigation, writing — original draft, visualization. Somayeh Lotfi: supervision, methodology, validation, writing — review & editing.

**Competing interests.** `[TBD]`

**Funding.** `[TBD]`

**Use of AI tools.** AI assistance (Claude, Anthropic) has been used for protocol drafting and manuscript preparation, and will be used for code development. The authors are responsible for all content, and every claim, citation and number will be verified by the authors against its primary source before submission. The target journal's disclosure requirements will be followed. `[Revise to past tense, and describe the actual uses, at submission.]`

---

## 10. Outstanding items before submission

1. **Target transform** — confirm in the released files the transform, range, masks and diagonal offset documented in §2.2. Until then, g in §4.1 is provisional.
2. **ChiMamba** — read the published chapter; tabulate task, inputs, output, resolution and split; confirm related work versus baseline.
3. **Verify** the remaining prior-art details (DeepC, Orca, C.Origami encoder/window entries), the SCC parameters used by the benchmark, DNALongBench's per-cell-line published values, and the final *Nat Commun* version against the preprint values quoted in §1.
4. **Independent boundary and loop sources** — select accessions per §4.6 and record them before evaluation.
5. **Specify** the convolutional downsampling stack, the boundary-loss form, the RC pairing of pairwise-head and FiLM channels, and all hyperparameters.
6. **Compute environment** for the paper configuration.
7. **Confirm** authorship, order and CRediT with all authors.
8. **Target journal** — select after results; format accordingly.

---

## References

Verified against the publisher, PubMed/PMC, arXiv or Crossref on 2026-10-21.

- Avsec Ž, Latysheva N, Cheng J, et al. Advancing regulatory variant effect prediction with AlphaGenome. *Nature* 649, 1206–1218 (2026). doi:10.1038/s41586-025-10014-0
- Cheng W, Song Z, Zhang Y, Wang S, Wang D, Yang M, Li L, Ma J. DNALONGBENCH: a benchmark suite for long-range DNA prediction tasks. *Nat Commun* 16 (2025). doi:10.1038/s41467-025-65077-4. Preprint: bioRxiv doi:10.1101/2025.01.06.631595 `[article number to add]`
- Crane E, Bian Q, McCord RP, et al. Condensin-driven remodelling of X chromosome topology during dosage compensation. *Nature* 523, 240–244 (2015). doi:10.1038/nature14450
- Fudenberg G, Kelley DR, Pollard KS. Predicting 3D genome folding from DNA sequence with Akita. *Nat Methods* 17, 1111–1117 (2020). doi:10.1038/s41592-020-0958-x
- Gu A, Dao T. Mamba: linear-time sequence modeling with selective state spaces. arXiv:2312.00752 (2023).
- Krietenstein N, Abraham S, Venev SV, et al. Ultrastructural details of mammalian chromosome architecture. *Mol Cell* 78, 554–565.e7 (2020). doi:10.1016/j.molcel.2020.03.003
- Nguyen E, Poli M, Faizi M, et al. HyenaDNA: long-range genomic sequence modeling at single nucleotide resolution. *NeurIPS* (2023). arXiv:2306.15794
- Perez E, Strub F, de Vries H, Dumoulin V, Courville A. FiLM: visual reasoning with a general conditioning layer. *AAAI* (2018). arXiv:1709.07871
- Rao SSP, Huntley MH, Durand NC, et al. A 3D map of the human genome at kilobase resolution reveals principles of chromatin looping. *Cell* 159, 1665–1680 (2014). doi:10.1016/j.cell.2014.11.021
- Rao SSP, Huang S-C, Glenn St Hilaire B, et al. Cohesin loss eliminates all loop domains. *Cell* 171, 305–320.e24 (2017). doi:10.1016/j.cell.2017.09.026
- Schiff Y, Kao C-H, Gokaslan A, Dao T, Gu A, Kuleshov V. Caduceus: bi-directional equivariant long-range DNA sequence modeling. *Proc ICML*, PMLR 235, 43632–43648 (2024). PMID 40567809
- Schwessinger R, Gosden M, Downes D, et al. DeepC: predicting 3D genome folding using megabase-scale transfer learning. *Nat Methods* 17, 1118–1124 (2020). doi:10.1038/s41592-020-0960-3
- Tan J, Shenker-Tauris N, Rodriguez-Hernaez J, et al. Cell-type-specific prediction of 3D chromatin organization enables high-throughput in silico genetic screening. *Nat Biotechnol* 41, 1140–1150 (2023). doi:10.1038/s41587-022-01612-8
- Yang T, Zhang F, Yardımcı GG, et al. HiCRep: assessing the reproducibility of Hi-C data using a stratum-adjusted correlation coefficient. *Genome Res* 27, 1939–1949 (2017). doi:10.1101/gr.220640.117
- Zhang J, Song C, Cui T, Li C, Ma J. ChiMamba: predicting chromatin interactions based on Mamba. In: *Advanced Intelligent Computing in Bioinformatics* (ICIC 2024), LNCS 14882, 50–61. Springer (2024). doi:10.1007/978-981-97-5692-6_5
- Zhou J. Sequence-based modeling of three-dimensional genome architecture from kilobase to chromosome scale. *Nat Genet* 54, 725–734 (2022). doi:10.1038/s41588-022-01065-4
