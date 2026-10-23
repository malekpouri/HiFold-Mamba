# 2026-10-23 — G8: candidate independent boundary / loop datasets (proposal, not yet downloaded)

Rule (protocol §8 G8, resolved):
- **Boundaries:** from an *independent* Hi-C/Micro-C dataset of the same line, i.e. a different study or replicate from the training target. Secondary set: CTCF ChIP-seq peaks with a CTCF motif.
- **Loops:** from an independent dataset. Loop calls derived from the training data are excluded.

Survey: 4DN Data Portal, released ExperimentSetReplicates per cell line, queried 2026-10-23 (`scratchpad/survey4dn.py`). The training targets come from `akita_1m_metadata/targets.txt`.

| Line | Training target (source) | Boundary candidate (independence) | Loop candidate |
|---|---|---|---|
| HFF | Micro-C, Dekker lab (Krietenstein 2020), 4DNFIP5EUOFX | in situ Hi-C HFFc6, Dekker lab (4 sets): **same lab, different assay and library** | in situ ChIA-PET, Ruan lab (2 sets): **different lab** |
| H1-hESC | Micro-C, Dekker lab, 4DNFI1O6IL1Q | in situ Hi-C, **Huangfu lab, MSKCC** (3) or **Zhong lab, UCSD** (4): **different lab** | in situ ChIA-PET, Ruan lab (2) |
| GM12878 | in situ Hi-C MboI combined, Aiden lab (Rao 2014) | Dilution Hi-C, **Ren lab, UCSD** (1): **different lab**; fallback: Aiden-lab DpnII/HindIII/NcoI protocol variants (different library, same lab) | in situ ChIA-PET (7) / ChIA-PET (2), **Ruan lab** |
| IMR-90 | in situ Hi-C, Aiden lab (Rao 2014) | Dilution Hi-C, **Ren lab, UCSD** (1): **different lab** (resolved: Jin et al. 2013, GSE43070) | none on 4DN: CTCF-motif-pair loops from ENCODE CTCF ChIP-seq only (secondary) |
| HCT116 | in situ Hi-C, Aiden lab (Rao 2017, unsynchronized) | in situ Hi-C, **Casellas lab, NIH** (43) or **Cremins lab, UPENN** (2): **different lab** | in situ ChIA-PET, Ruan lab (7) |

Caveats to check before choosing specific accessions:
- **Factor:** which ChIA-PET sets target CTCF and which target RNAPII. Use CTCF for loop AP.
- **Genotype and condition:** e.g. HCT116 sets that are auxin-degron or treated lines. Use untreated wild-type conditions only.
- **Processed files:** is an hg38 `.mcool` available (boundary calling), and does 4DN ship loop calls or only pairs? This changes the download size; at current bandwidth, multi-GB `.mcool` files take hours each.
- **Depth:** dilution Hi-C (Ren) is shallow. Boundaries at 2 kb need enough depth; a coarser boundary resolution may be required.
- **Boundary caller:** use a different caller from our insulation operator for the independent boundaries, to reduce shared methodological bias. Fix it before evaluation.

**Approved by the user 2026-10-23; resolved in `2026-10-23_g8-resolved.md`.** Original request: approve this table (one boundary and one loop source per line), or adjust it. Accessions will then be resolved and recorded in `docs/data_manifest.md` before any evaluation.
