# Internal review, rounds 4–5 (critic role) and responses, 2026-10-23

## Round 4: consistency and submission readiness

- **Abstract too long** (421 words). Rewritten as a structured abstract of 315 words.
- **Data-access statement out of date.** §2.16 said the annotations had not been intersected with anything. After the positive control this was no longer true. It now states that they were intersected with observed validation maps only.
- **Design table did not mention the H2 positive control or the final-checkpoint rule.** Added.
- **No submission materials.** Added `paper/cover_letter.md` and `paper/SUBMISSION_CHECKLIST.md`.

## Round 5: every pilot number checked against its result file

- **§2.10 max-bias range.** It was stated as 0.02–0.04. The actual range is 0.846·σ·(1 − ρ)^½ = 0.023–0.073 over ρ ∈ [0.8, 0.98]. Corrected to 0.02–0.07.
- **Abstract power claim** was missing its condition. Added "between-model correlation ≥ 0.95".
- **Checked and correct:**
  - §3.4: boundary AUROC at w = 40, the gain from w = 20 to 40 (+0.04–0.10), base rates (3.5–5.6 %), loop AUROC (0.86–0.90) and AP (0.71–0.79), against `results/pilot/structure_control_valid.tsv`.
  - §3.3 against `results/reference/*.tsv`.
  - The power tables against `results/pilot/power_h1*.tsv`.
  - Parameter counts against the model code.
