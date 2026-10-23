# Internal review, round 3 (critic role) and response, 2026-10-23

**R3-1 (major). No positive control for the H2 evaluation.** A null H2 could not be told apart from a broken pipeline, the margins had no scale, and w = 20 was never checked.

- **Response:** ran the exact Stage 2 pipeline on validation windows with observed maps and the locus-shared reference (`scripts/pilot_structure.py`, `results/pilot/structure_control_valid.tsv`). New §3.4 and Fig. 3.
- **Findings:**
  - Boundary AUROC 0.61–0.91 and loop AUROC 0.86–0.90.
  - w = 40 is better than w = 20 in every line (+0.04–0.10). **Adopted w = 40** (deviation D7), with sensitivity at 20 and 80.
  - The annotations are largely cell-type-invariant (locus-shared ≈ own map). H2 is reinterpreted as recovery of invariant folding features; cell-type claims rest on H3.

**R3-2 (minor). The 10 kb dilution Hi-C boundary sets (GM12878, IMR-90) give low attainable AUROC.** Stated in §3.4. Their per-line results will be shown alongside the line average.

Deviation for the user: **D7**, insulation window 20 → 40 bins (chosen on validation, before any model training).
