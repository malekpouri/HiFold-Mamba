# Status — resume here

Last session: 2026-10-21. Details: `notes/2026-10-21_phase1-2.md`, `notes/2026-10-21_phase3-5.md`.

## Where we are

| Phase | State |
|---|---|
| 1 Environment | Done. uv venv, torch 2.14.1+cu130, RTX 5050 OK (bf16). Fused Mamba kernels not used (pure-PyTorch scan) |
| 2 Data & target | Partly done. Format, alignment to hg38 and the H3 split are verified; H3-line targets inspected. **Download paused at ~1.2 / 26.5 GB** (Akita: 3 / 32 files) |
| 3 Evaluation code | Done and tested (SCC ×2, PCC, block bootstrap, insulation, AUROC/AP) |
| 4 Baselines | Not started (needs data) |
| 5 Model | Done. **Equivariance test passes** |
| 6 Structure losses | Insulation done; boundary/loop losses not yet |
| 7 Dev runs | Not started (needs data + free GPU) |

Tests: 41 pass, **1 fails** (`test_training_loop_runs_and_learns`, see notes).

## Commands to resume

```bash
cd "/mnt/storage/datasets/Human genomes Dataset/new_idea/03-3DChromatin-Folding"

# 1. Resume the download (skips MD5-verified files; ~150 KB/s here, i.e. ~2 days for everything)
nohup python3 -I scripts/download_dnalongbench.py "/mnt/storage/datasets/Human genomes Dataset/new_idea/datasets/dnalongbench" contact_map_prediction/targets > download_targets.log 2>&1 &
#    afterwards: same command with contact_map_prediction/extra_data.complete

# 2. Tests (CPU)
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q tests

# 3. When targets/ is complete: build the cache (verifies records against hg38)
D="/mnt/storage/datasets/Human genomes Dataset/new_idea/datasets"
PYTHONPATH=src .venv/bin/python scripts/build_cache.py "$D/dnalongbench/contact_map_prediction/targets" \
  "$D/dnalongbench/akita_1m_metadata/sequences.bed" HFF,H1hESC,GM12878,IMR90,HCT116 \
  "$D/dnalongbench/cache/akita5" "$D/reference_hg38/hg38.fa"
```

## Decisions waiting for the user

1. **Primary SCC variant.** The benchmark's SCC weights strata ∝ 1/N (the opposite of HiCRep). Options: (a) benchmark SCC primary + HiCRep alongside, or (b) the reverse.
2. **H3 lines are unsmoothed** (`kernel_stddev 0`). Recommended: keep the targets unchanged and report this as a limitation.
3. **`cell_dist_bias`** (per-cell distance bias). Recommended: on.
4. **Slow network** (~150 KB/s): try the Box mirror, or download on a faster connection and copy the files over.

## Next steps (in order)

1. Diagnose the failing smoke test.
2. Write the boundary loss.
3. Write the GPU memory profiler; needs the GPU free of the CRISPR job.
4. Write the baselines: global mean and locus-shared mean.
5. Build the cache once the download completes; inspect the Akita targets (G1 for the five lines).
6. Run the reproduction of Akita.
