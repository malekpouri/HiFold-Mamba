# 2026-10-23 — G1 closed, GPU profiling, first reference baseline

## Smoke test (open item from 2026-10-21): fixed — it was not a bug

- Both runs were still falling when the 60-step cosine schedule brought the learning rate to zero.
- With identical budgets, the final contact loss was 0.0185 with the shared distance bias vs 0.0103 with `cell_dist_bias`; insulation loss 0.0072 vs 0.0009.
- The test now asserts relative properties: >90 % loss reduction, and the per-cell bias beating the shared one.
- 41/41 tests pass.

## G1 — closed for the 5 training lines

Akita's `bin/akita_data_read.py` (calico/basenji), read 2026-10-21. The pipeline order is:
1. adaptive coarse-graining;
2. local obs/exp;
3. **`np.log` (natural log)**;
4. clip ±clip;
5. `interp_nan`;
6. near-diagonal band set to 0;
7. Gaussian kernel (after the clip).

If a cooler lacks the region, the target is **set to all zeros**; `build_cache.py` now counts such maps.

Akita test split, 413 windows (cache built, hg38-verified, **0 all-zero maps**):
- values lie within [−2, 2]; no NaNs, no masks shipped; 99,681 values per line (offset 2);
- smoother than the four H3 lines, as expected;
- each line has its own residual P(s): the trough runs from −0.17 (HCT116) to −0.30 (HFF, IMR90). See `figures/akita5_test0.png`. This is real-data support for `cell_dist_bias`.

The insulation transform `g = exp(2 tanh(x/2))` stands (G2): its range matches the ±2 clip.

## GPU profiling — full 1,048,576 bp input, bf16, checkpointing, 5 cells per window

| Config | Scan | Peak (GiB) | s/step |
|---|---|---|---|
| fallback (0.57 M) | loop, chunk 16 | 3.43 | 1.47 |
| fallback | **batched**, chunk 16 | 3.79 | **1.10** |
| paper (3.32 M) | loop, chunk 16 | OOM | — |
| paper | loop, chunk 8 | 4.30 | 5.81 |
| paper | **batched, chunk 8** | 4.72 | **3.16** |

- Batch 2 OOMs for both configurations, so the effective batch must come from gradient accumulation.
- The batched scan is exact: it is tested against the sequential reference. Default chunk is now 8.
- **Consequence:** the paper model trains on the local 8 GB GPU at about 6.2 h per pass over the 7,008 training windows. Cloud compute is needed for the full grid (3 seeds × ablations × H3), not for feasibility.

## First reference baseline — locus-shared mean, Akita test split

`scripts/reference_baselines.py`; per-window values in `results/reference/locus_shared_test.tsv`. Oracle-style: it uses the other lines' measured test maps, so it is never in ℬ (G7).

| Line | SCC (benchmark code) | SCC (HiCRep) | PCC (per window) |
|---|---|---|---|
| HFF | 0.248 | 0.572 | 0.623 |
| H1hESC | 0.223 | 0.546 | 0.598 |
| GM12878 | 0.216 | 0.503 | 0.554 |
| IMR90 | 0.240 | 0.540 | 0.591 |
| HCT116 | 0.208 | 0.492 | 0.547 |
| **mean** | **0.227** | **0.531** | **0.582** |

Observations (not claims):
- The two SCC variants rank windows quite differently: Spearman ρ = 0.54 over 2,065 (window, line) pairs. The pending choice of primary SCC is therefore consequential.
- The DNALongBench preprint's Table 3 gives Akita an average contact-map score of 0.230 `[UNVERIFIED: whether that score is this SCC, and whether it pools all 9 lines]`. If it is the benchmark SCC, the measured-map oracle is on par with the published expert model. Check against the published per-line values before any statement in the paper.

## Speed

- SCC is vectorised over diagonals; results are identical to the loop on real data (max diff 0 at 6 decimals).
- Full test-set reference: 10.5 min → 1.5 min.

## Akita port and scan compilation (later the same day)

**Akita in PyTorch** (`src/hifold/model/akita.py`), faithful to `manuscripts/akita/params.json` and basenji `blocks.py`/`layers.py`:
- 746,149 parameters;
- dilation rates [1, 2, 3, 5, 9, 16, 29, 50] (1D trunk) and [1, 2, 3, 5, 9, 16] (2D head);
- symmetric output; cell selection tested.
- Uses RC augmentation (sequence and map together; the `rot180_triu_perm` permutation is tested).
- Profile: batch 2 without checkpointing takes 4.73 GiB and 0.25 s/step. Checkpointing is off, so BatchNorm statistics are not updated twice.

**Where HiFold's time went:** at 3.16 s/step, the tower took 0.37 s and the 32 Mamba direction calls about 2 s.
- Batching the paired paths (BiMamba directions, RCPS x/RC(x)) gave **no** gain: the scan is memory-bandwidth-bound. Reverted.
- `torch.compile` on the scan: 60 → 26 ms per call, peak 1380 → 620 MiB, max deviation 1.1e-5 at scale 28 (fp32 rounding). Used on CUDA only; CPU tests stay eager.

| config, 1 window x 5 lines | s/step | peak | 1 pass over 7,008 windows |
|---|---|---|---|
| HiFold paper, chunk 16 (new default) | 1.34 | 3.84 GiB | 2.6 h |
| HiFold paper, chunk 8 | 1.64 | 2.62 GiB | 3.2 h |
| HiFold fallback | 0.52 | 1.37 GiB | 1.0 h |
| Akita | 0.12 per window | 4.73 GiB at batch 2 | 0.25 h |

HiFold remains about 11x slower per window than Akita. This is a cost, not a result, and it is consistent with the protocol's rule of no efficiency claim at 512 bins.
