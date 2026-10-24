# PCI RR Stage 1: submission notes and draft survey answers

**[UNVERIFIED]** The PCI RR website blocks automated access, so its form fields could not be read directly. Check each answer against the live form at <https://rr.peercommunityin.org> when submitting.

## What to upload / enter

| Item | Source |
|---|---|
| Title, abstract, keywords | `manuscript.md` (front matter, Abstract). Keywords: 3D genome; Hi-C; Micro-C; contact-map prediction; deep learning; state-space models; insulation; registered report |
| Stage 1 manuscript (PDF, line-numbered) | `manuscript.pdf` |
| Cover letter | `cover_letter.md` |
| Code / data link | https://github.com/malekpouri/HiFold-Mamba (record the commit hash at submission) |
| Study design table | Manuscript §2.17 |
| Bias-control level | **Level 2** (manuscript §2.16) |

## Draft answers to the Stage 1 report survey

- **Type of submission:** regular Stage 1 (not programmatic, not scheduled review).
- **Bias-control level:** Level 2. Some data, including the test-set targets, were accessed and partly observed. The key variables (trained-model test predictions) do not exist yet.
- **Data and code availability:** all public; DOIs and accessions in §2.1, §2.8 and `docs/data_manifest.md`.
- **Ethics approval:** not required (public, non-identifiable genomic data).
- **Required resources in place:** yes. A local workstation is sufficient, and there is no external funding dependency.
- **Stage 2 deadline:** six months after in-principle acceptance.
- **Target journal after recommendation:** Peer Community Journal.
- **Suggested reviewers:** none. **Opposed reviewers:** none.
- **Prior preregistration:** none.
- **Conflicts of interest:** none.

## After in-principle acceptance

- The Stage 1 manuscript is registered on OSF (PCI RR guides this step). Freeze and record the evaluation commit.
- Unseal the test set only after the registration.
