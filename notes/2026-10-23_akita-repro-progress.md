# 2026-10-23 — Akita reproduction: progress check at step ~13,100 (of 70,080)

Run: `runs/akita_repro-316312a170-s0` (commit ec3fae9, clean), started 10:45, ≈ 0.32 s/step (≈ 6.2 h total).

| Step | val SCC_benchmark | val SCC_hicrep | val PCC |
|---|---|---|---|
| 3,504 | 0.002 | 0.007 | 0.084 |
| 7,008 | 0.019 | 0.054 | 0.048 |
| 10,512 | 0.030 | 0.076 | 0.026 |

Training loss (mean per 2,000 steps): 0.21, 0.23, 0.19, 0.20, 0.20, 0.21, 0.24.

Trivial references on 200 random training windows:
- Predict 0: MSE 0.256.
- Per-position training mean: MSE 0.226.

**Observation.** After about 3.7 passes over the data, the model is only slightly better than the per-position mean. Validation SCC is rising but still about 10 % of the published Akita test value (0.230, benchmark code).

**Possible causes, not yet tested.** Still early (Akita was trained much longer, with patience-based stopping); learning-rate or momentum translation from Keras SGD; BatchNorm momentum convention; EMA evaluation of BN statistics.

**Decision rule.** If the final validation SCC_benchmark is below 0.15, diagnose before the reproduction is used anywhere. The checks:
1. Train a short run without EMA.
2. Check Keras-to-PyTorch BN momentum (PyTorch momentum = 1 − Keras momentum).
3. Try Adam on a small subset to see whether the architecture can fit at all.

Akita stays in 𝓑 either way (manuscript §2.4); a weak reproduction is reported as a deviation.
