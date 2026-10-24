Dear Managing Board and Recommenders of PCI Registered Reports,

We submit a **Stage 1 Registered Report** for review by PCI RR: "Does a map-level insulation constraint improve sequence-based prediction of 3D genome folding? A Stage 1 Registered Report with a reverse-complement-equivariant state-space model (HiFold-Mamba)".

**Question.** Deep models that predict Hi-C and Micro-C maps from DNA sequence are trained pixel-wise, yet they are judged on map summaries: insulation, domain boundaries and loops. Insulation-based losses exist for Hi-C resolution enhancement (VEHiCLE). We ask whether such a constraint improves *sequence-based* prediction. We judge this against boundary and loop annotations that are independent of the training data, under a design fixed before any confirmatory result is seen.

**Why a Registered Report.** Results on this benchmark are reported as mean scores, and the claims rarely carry a test of their own. Our pilot data show why that matters:

- A non-learned reference built from other cell types' measured maps matches the published expert model on the benchmark metric.
- Two SCC weightings in common use weight distance strata in opposite directions.
- A per-block-best comparison against several baselines, which we had originally planned, has no statistical power at the benchmark's sample size.

We therefore fix the hypotheses, the comparison set, the margins, the statistics and the conditional claims in advance, and we report the outcome whatever its direction.

**What is ready.**

- Model, baselines, training, evaluation and statistics code, with 64 unit tests, including exact reverse-complement equivariance checks.
- Every dataset downloaded with checksums and accessions.
- A positive control showing that the H2 evaluation pipeline recovers the independent annotations from observed maps.
- A simulation-based power analysis.

No trained model has been evaluated on the test set.

**Data status (bias control).** All data are public and already exist. Before this submission we accessed the training and validation data, and the test-set *targets* only for non-learned references and the power analysis. The independent annotations were intersected only with observed validation maps. Section 2.16 lists all data access and the safeguards we have adopted. We declare **bias-control Level 2**: the key variables (trained models' test-set predictions and their overlap with the independent annotations) do not yet exist and will be generated only after in-principle acceptance.

**Practical details.**

- **Timeline.** Stage 2 within six months of in-principle acceptance. The compute (≈ 640 GPU-hours) is on the authors' own workstation, so no further resources or approvals are needed.
- **Ethics.** None required: all data are public and contain no human-subject information.
- **Openness.** Code (MIT licence), configurations and pilot outputs are public; all Stage 2 data and code will be shared.
- **Publication.** After a Stage 2 recommendation we intend to publish in *Peer Community Journal*.

The work has not been submitted elsewhere. Both authors have approved the submission. The study received no external funding, and the authors declare no competing interests.

Sincerely,
Mohammad Malekpouri (corresponding author), on behalf of both authors
Department of Computer Engineering, BA.C., Islamic Azad University, Bandar Abbas, Iran
m.malekpouri@iau.ir · ORCID 0009-0001-6278-6163
