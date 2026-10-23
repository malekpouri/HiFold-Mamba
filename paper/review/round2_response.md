# Response to internal review round 2 (author role), 2026-10-23

| Item | Action |
|---|---|
| R2-1 SCC definition | §2.7 corrected (Pearson on raw values, rank-based weights). Weight directions stated (benchmark ∝ 1/N_d, HiCRep ∝ N_d) and linked to ρ = 0.54 in §3.3. Checked against `src/hifold/eval/metrics.py` |
| R2-2 "widely used" | Removed; now "which we had originally planned" |
| R2-3 outcome-neutral checks | Added: pipeline, Akita reproduction and sanity-floor controls; exclusion rules |
| R2-4 H3 subsets | Seeded contig permutation, nested subsets, test unit specified |
| R2-5 overclaim | Restated |
| R2-6 compute | "Equal optimizer steps and data". Overhead described, to be measured, not asserted |
| R2-7 λ pilot | Fork of Base seed 0 at step 52,560 |
| R2-8 "exactly" | "By construction; verified to 10⁻⁴" |
