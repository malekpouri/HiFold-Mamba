# Status — resume here

Last update: 2026-10-23 (afternoon). Details in `notes/` (latest: `2026-10-23_g1-profiling-reference.md`, `2026-10-23_g8-candidates.md`).

## Where we are

| Phase | State |
|---|---|
| 1 Environment | Done. torch 2.14.1+cu130, compiled pure-PyTorch scan. Paper model fits 8 GB: 1.34 s/step, 3.8 GiB |
| 2 Data & target | **G1 closed** (natural-log O/E, clip ±2, smoothed after the clip). Test cache built and hg38-verified. Download of the 5 training lines running; train/valid cache builds automatically when it finishes. The 4 H3 lines download next |
| 3 Evaluation code | Done. Vectorised SCC (both variants), PCC, block bootstrap, insulation, AUROC/AP, `scripts/evaluate.py` |
| 4 Baselines | Akita ported (746k params). Matched Transformer and matched CNN (parameter-matched, RC-equivariant). Test references: locus-shared SCC_hicrep 0.531; global mean 0.014 (near zero in O/E space, see notes) |
| 5 Model | Done. Equivariance test passes, including the negative control |
| 6 Structure losses | Insulation and boundary losses done. Loop loss needs the G8 data. Stage-2 weights not yet set |
| 7 Runs | Train/valid caches built (7,008 / 419, hg38-verified, no all-zero maps). Akita reproduction queued: starts automatically when the CRISPR job frees the GPU (`runs_akita.log`) |

Tests: **53/53 pass** (`CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q tests`).

## Decisions (approved by the user 2026-10-23, recorded in `Roadmap.md`)

1. Primary SCC = HiCRep weighting; benchmark-code SCC reported alongside.
2. H3 lines: unsmoothed targets used unchanged (limitation).
3. `cell_dist_bias` on (default and all configs).
4. G8 resolved and downloaded: 4DN boundaries for 5 lines, ENCODE CTCF ChIA-PET loops for 4 (none for HFF). Pre-registered: PET ≥ 10, shipped 4DN boundary calls. Open: bin-matching tolerance for the 5/10 kb calls on 2,048 bp bins.
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

## Paper (goal: Q1-ready manuscript), state 2026-10-23 evening
- **Stage 1 Registered Report:** `paper/manuscript.md` → `paper/manuscript.pdf` (21 pages, 3 figures, 23 verified references), plus `paper/cover_letter.md`.
- **Venue:** PCI RR → Peer Community Journal (chosen 2026-10-24). D1–D7 and all statements settled. Remaining: the authors' full read-through, then submission (`paper/pci_rr_submission_notes.md`).
- **Akita seed 0 done:** validation SCC_benchmark 0.196; seeds 1–2 are still needed.
- **Build:** `cd paper && pandoc manuscript.md --citeproc -o manuscript.pdf --pdf-engine=xelatex -V geometry:margin=2.2cm -V fontsize=10pt -V mainfont="DejaVu Serif" -V linestretch=1.15 -V header-includes='\usepackage{lineno}\linenumbers'`
- **Review trail:** five critic/author rounds in `paper/review/`.
- **Analysis code ready:** H1 intersection–union (`stats.against_each`), H2 bootstrap (`eval/structure_eval.py`), H2 positive control (`scripts/pilot_structure.py`), power analysis (`scripts/power_analysis.py`).
- **Do NOT score trained models on the test set before Stage 1 acceptance.**
- **Next compute:** Akita reproduction (queued behind the CRISPR job), then the ρ calibration and the λ pilot (validation only).
