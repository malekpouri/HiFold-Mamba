# Response to internal review round 1 (author role), 2026-10-23

| Item | Action | Where |
|---|---|---|
| M1 H2 not executable | Defined items as genomic bins and pairs, deduplicated across overlapping windows (mean over windows, then seeds), per-line metric averaged over lines, block resampling shared across lines and arms with item weights; exact weighted AUROC and AP (digamma identity); 7 tests | §2.8, §2.9; `src/hifold/eval/structure_eval.py`, `tests/test_structure_eval.py` |
| M2 checkpoint confound | H1/H3/ablations use the Base arm (validation-selected); H2 compares final EMA weights of both arms | §2.5 |
| M3 power ignores max over 3 | Simulated: the per-block-max rule has power 0.00 for Δ ≤ 0.02 at every ρ. **Rule replaced** by an intersection–union test against each baseline; worst-case power ≥ 0.9 at Δ ≈ 0.02 (ρ ≥ 0.95). New Fig. 2c | §2.9, §2.10; `stats.against_each`; `results/pilot/power_h1_{max3,iut3}.tsv` |
| M4 error control | Rationale stated: separate claims each at α; within-hypothesis control by IUT, Holm and fixed sequence; all outcomes reported | §2.9 |
| M5 H3 trivially won | Zero-shot reference added (secondary) | §2.11 |
| M6 published-SCC assumption | Stated | §3.3 |
| m1 overgeneralization | Restricted to verified models | §1.2 |
| m2 Hi-Compass | Corrected | §1.1 |
| m3 block dependence | 3 Mb-block sensitivity added | §2.9 |
| m4 margin scale | Stated and justified | §2.9 |
| m5 Akita criterion | Within 0.02 of the published 0.230; otherwise a reported deviation | §2.4 |
| m6 λ selection data | Acknowledged (validation regions only) | §2.6 |
| m7 environment | Versions and hardware added | Statements |
| m8 duplicate labels | Fixed | figures |
| m9 boundary labels | Strong and Weak both positive | §2.8 |

New deviation for the user: **D6.** H1 changes from a per-block-best rule to an intersection–union test against every baseline (forced by the power analysis).
