# Status — resume here

Last update: 2026-10-23 (afternoon). Details in `notes/` (latest: `2026-10-23_g1-profiling-reference.md`, `2026-10-23_g8-candidates.md`).

## Where we are

| Phase | State |
|---|---|
| 1 Environment | Done. torch 2.14.1+cu130, compiled pure-PyTorch scan. Paper model fits 8 GB: 1.34 s/step, 3.8 GiB |
| 2 Data & target | **G1 closed** (natural-log O/E, clip ±2, smoothed after the clip). Test cache built and hg38-verified. Download of the 5 training lines running; train/valid cache builds automatically when it finishes. The 4 H3 lines download next |
| 3 Evaluation code | Done. Vectorised SCC (both variants), PCC, block bootstrap, insulation, AUROC/AP, `scripts/evaluate.py` |
| 4 Baselines | Akita ported (746k params). Matched Transformer and matched CNN (parameter-matched, RC-equivariant). Locus-shared reference computed on test; global mean ready |
| 5 Model | Done. Equivariance test passes, including the negative control |
| 6 Structure losses | Insulation and boundary losses done. Loop loss needs the G8 data. Stage-2 weights not yet set |
| 7 Runs | Next: Akita reproduction (`configs/akita_repro.yaml`, ~5 h locally) as soon as the cache is ready |

Tests: **53/53 pass** (`CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q tests`).

## Decisions (approved by the user 2026-10-23, recorded in `Roadmap.md`)

1. Primary SCC = HiCRep weighting; benchmark-code SCC reported alongside.
2. H3 lines: unsmoothed targets used unchanged (limitation).
3. `cell_dist_bias` on (default and all configs).
4. G8 table approved. Accessions are being resolved from 4DN, to be recorded in `docs/data_manifest.md`.
5. Strong CNN = parameter-matched dilated-conv mixer.
6. L_bnd = MSE on boundary-strength profiles (`train.boundary_loss`, config key `struct.bnd`).

## Commands

```bash
cd "/mnt/storage/datasets/Human genomes Dataset/new_idea/03-3DChromatin-Folding"
# resume the download if interrupted (MD5-verified files are skipped)
R="/mnt/storage/datasets/Human genomes Dataset/new_idea/datasets/dnalongbench"
nohup python3 -I scripts/download_dnalongbench.py "$R" contact_map_prediction/targets > download_targets.log 2>&1 &
# Akita reproduction (after the cache exists)
PYTHONPATH=src nohup .venv/bin/python -m hifold.train configs/akita_repro.yaml > runs_akita.log 2>&1 &
```
