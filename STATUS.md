# Status — resume here

Last update: 2026-10-23. Details in `notes/` (latest: `2026-10-23_g1-profiling-reference.md`, `2026-10-23_g8-candidates.md`).

## Where we are

| Phase | State |
|---|---|
| 1 Environment | Done. torch 2.14.1+cu130, compiled pure-PyTorch scan. Paper model fits 8 GB: 1.34 s/step, 3.8 GiB |
| 2 Data & target | **G1 closed** (natural-log O/E, clip ±2, smoothed after the clip). Test cache built and hg38-verified. Download of the 5 training lines running; train/valid cache builds automatically when it finishes. The 4 H3 lines download next |
| 3 Evaluation code | Done. Vectorised SCC (both variants), PCC, block bootstrap, insulation, AUROC/AP, `scripts/evaluate.py` |
| 4 Baselines | Akita ported (746k params). Matched Transformer and matched CNN (parameter-matched, RC-equivariant). Locus-shared reference computed on test; global mean ready |
| 5 Model | Done. Equivariance test passes, including the negative control |
| 6 Structure losses | Insulation loss done. Boundary-loss form still open (architecture §6). Loop loss needs the G8 data |
| 7 Runs | Next: Akita reproduction (`configs/akita_repro.yaml`, ~5 h locally) as soon as the cache is ready |

Tests: **53/53 pass** (`CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q tests`).

## Decisions waiting for the user

1. **Primary SCC variant.** Benchmark code vs HiCRep. On real data they correlate only at ρ = 0.54 across windows. Checkpoint selection already follows CLAUDE.md (HiCRep).
2. **H3 lines unsmoothed.** Recommended: keep the targets unchanged and report this as a limitation.
3. **`cell_dist_bias`.** Recommended: on. Real targets show a line-specific residual P(s) (trough −0.17 to −0.30).
4. **G8 sources.** Approve the boundary/loop table in `notes/2026-10-23_g8-candidates.md`.
5. **Strong CNN definition.** Implemented as HiFold with a dilated-conv mixer (parameter-matched). Confirm, or name a published model instead.
6. **Boundary loss form** (architecture §6, "before coding").

## Commands

```bash
cd "/mnt/storage/datasets/Human genomes Dataset/new_idea/03-3DChromatin-Folding"
# resume the download if interrupted (MD5-verified files are skipped)
R="/mnt/storage/datasets/Human genomes Dataset/new_idea/datasets/dnalongbench"
nohup python3 -I scripts/download_dnalongbench.py "$R" contact_map_prediction/targets > download_targets.log 2>&1 &
# Akita reproduction (after the cache exists)
PYTHONPATH=src nohup .venv/bin/python -m hifold.train configs/akita_repro.yaml > runs_akita.log 2>&1 &
```
