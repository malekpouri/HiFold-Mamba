# Stage 1 submission checklist (state 2026-10-23)

## Ready (done and verified)

- [x] Manuscript: `paper/manuscript.md` → `paper/manuscript.pdf` (pandoc + xelatex; command in `STATUS.md`)
- [x] References: 23 entries, each resolved through Crossref or Europe PMC (`paper/references.bib`)
- [x] Figures 1–3, generated from logged data by `scripts/figure_schematic.py`, `scripts/figures_pilot.py` and `scripts/figure_structure_control.py`
- [x] Study design table (§2.17); power analysis (§2.10, `scripts/power_analysis.py`); positive control (§3.4, `scripts/pilot_structure.py`)
- [x] Analysis code for every confirmatory test, unit-tested: `hifold.eval.stats.against_each` (H1), `hifold.eval.structure_eval` (H2), `hifold.eval.stats.block_bootstrap` / `holm` (H3)
- [x] Data accessions, checksums and access dates (`docs/data_manifest.md`)
- [x] Internal review trail: `paper/review/round1–3*.md`
- [x] Cover letter draft (`paper/cover_letter.md`)

## Settled by the authors (2026-10-24)

- [x] Protocol deviations D1–D7 approved.
- [x] CRediT: Malekpouri (all roles except review-only), Lotfi (Writing – review & editing). Both approve.
- [x] Funding: none. Competing interests: none. Code licence: MIT (copyright Mohammad Malekpouri).
- [x] AI-use statement: code generation and drafting/editing of text (wording in the manuscript).
- [x] No suggested reviewers. Computational resources stated (Statements; §4).
- [x] ChiMamba chapter not accessible; positioned from its abstract, stated in §1.3.

## Still open

- [ ] **Venue:** Q1 and no author fees (see `notes/2026-10-24_venue-options.md`).

## Before Stage 2 (after in-principle acceptance)

- Akita seeds 1–2 (seed 0 done, validation SCC_benchmark 0.196). Then the ρ calibration on validation (§2.10) and the λ pilot (§2.6).
- Unseal the test set only with the frozen evaluation commit.
