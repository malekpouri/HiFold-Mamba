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

## Needs the authors

- [ ] **Venue.** Two Stage 1 routes accept existing data (confirm current policies on the venue pages before submitting):
  - **PCI Registered Reports.** Journal-independent and free. After recommendation, the report can be published in a "PCI RR-friendly" journal. It uses a graded bias-control system for existing data, so declare the level in its checklist, using §2.16.
  - **PNAS Nexus.** Registered Reports, including secondary analyses of existing data with evidence of non-observation. Check the word limits (the abstract is 315 words, longer than its limit).
- [ ] **Approve the protocol deviations D1–D7** (`paper/review/round1.md`, `round1_response.md`, `round3.md`). The manuscript already implements them:
  - **D1:** Caduceus and HyenaDNA appear as published context only.
  - **D2:** no loop loss.
  - **D3:** H2 compares final checkpoints; H1 and H3 use the Base arm.
  - **D4:** no correction across hypotheses (rationale given).
  - **D5:** zero-shot reference added to H3.
  - **D6:** H1 is tested against each baseline (intersection–union), replacing the per-block maximum.
  - **D7:** insulation window w = 40.
- [ ] **Statements:**
  - CRediT roles and confirmation that both authors approve.
  - Funding.
  - Competing interests.
  - LICENSE copyright holder.
  - Final wording of the AI-use statement.
- [ ] **Suggested reviewers** (3–5).
- [ ] **Compute.** ≈ 640 GPU-hours on the local RTX 5050 (§4). Confirm that this is acceptable or arrange cloud/HPC time. This affects the Stage 2 timeline stated in §4.
- [ ] **ChiMamba chapter** (paywalled). Confirm that its task is not sequence-only contact-map prediction at this resolution. The manuscript positions it as related work, based on its abstract.

## Before Stage 2 (after in-principle acceptance)

- Akita reproduction (queued). Then the ρ calibration on validation (§2.10) and the λ pilot (§2.6).
- Unseal the test set only with the frozen evaluation commit.
