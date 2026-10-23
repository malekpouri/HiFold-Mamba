# Data manifest

Every dataset used by this project: source, accession, licence, access date, local path, checksums.
Data files themselves are never committed (`.gitignore`).

Data root: `/mnt/storage/datasets/Human genomes Dataset/new_idea/datasets/dnalongbench/`
(also reachable via the `~/Downloads/...` symlink). Also listed in that folder's shared `MANIFEST.md`, Part C.

## 1. DNALongBench, contact-map task

| Field | Value |
|---|---|
| Source | Harvard Dataverse, doi:10.7910/DVN/AZM25S, version 1.0 (linked from github.com/ma-compbio/DNALONGBENCH) |
| Licence | CC0 1.0 |
| Accessed | 2026-10-21 |
| Download | `scripts/download_dnalongbench.py` (Dataverse API; MD5 of every file checked against the Dataverse record) |
| Genome build | hg38 |

Parts downloaded:

| Dataverse folder | Content | Files | Size | Used for |
|---|---|---|---|---|
| `contact_map_prediction/targets/` | Akita's five lines (HFF, H1hESC, GM12878, IMR90, HCT116), 7,008 / 419 / 413 windows in 28 / 2 / 2 TFRecords | 32 | ≈ 11.1 GB | Training, H1, H2 |
| `contact_map_prediction/extra_data.complete/` | DNALongBench-added lines (HAP1, HeLa, HepG2, K562), 7,009 / 6,342 / 6,291 windows, plus metadata | 160 | ≈ 15.4 GB | H3 held-out lines only |
| `contact_map_prediction/extra_data.1m/` | 5 % subsample of the previous row | — | — | **Not downloaded** |

Record format (from the DNALongBench loader and Akita): ZLIB-compressed TFRecord; features `sequence` (raw uint8 one-hot, 1,048,576 × 4) and `target` (raw float16, 99,681 × n_lines; upper triangle, diagonal offset 2, row-major). Read by `src/hifold/data/tfrecord.py` (no TensorFlow).

### Processing metadata (verbatim facts from the shipped files)

- **Akita five** (`targets.txt` from Akita's own bucket, below): coolers HFF `4DNFIP5EUOFX` and H1hESC `4DNFI1O6IL1Q` (Krietenstein 2019/2020 Micro-C, MAPQ 30), GM12878 and IMR90 in-situ MboI (Rao 2014), HCT116 unsynchronized (Rao 2017); 2,048 bp; `clip 384`, `sum_stat sum`. Smoothing per the Akita paper: 2D Gaussian σ = 1.
- **DNALongBench four** (`extra_data.complete/targets.txt`, `options.json`): coolers HAP1 `4DNFIWGGYEW2`, HeLa `4DNFI65WJKMT`, HepG2 `4DNFIQ4G74OW`, K562 `4DNFI2R1W3YW` (4DN); `as_obsexp: true`, `no_log: false`, `clip 2`, **`kernel_stddev: 0` (no Gaussian smoothing)**, `diagonal_offset: 2`, `crop_bp: 65536`, `stride_train: 262144`, `stride_test: 32768`, split `.1/.1`, seed 44, gaps file `hg38_gaps_binsize2048_numconseq10.bed`.

### Split check (`scripts/check_h3_overlap.py`, 2026-10-21)

Both parts use the same 354 contigs (`contigs.bed`). Results:
- The DNALongBench four's valid/test windows overlap **no** Akita training window (0 / 6,342 and 0 / 6,291).
- No Akita valid/test window overlaps a training window of the four.
- The four's valid/test windows overlap each other heavily (32 kb stride), so evaluation must aggregate into 1 Mb blocks (protocol §3).

## 2. Akita window coordinates and target list

| Field | Value |
|---|---|
| Source | `https://storage.googleapis.com/basenji_hic/1m/data/{sequences.bed,targets.txt}`, the bucket used by Akita's official `calico/basenji/manuscripts/akita/get_data.sh` |
| Local | `dnalongbench/akita_1m_metadata/` (with a copy of `get_data.sh` as `get_data.sh.txt`) |
| Accessed | 2026-10-21 |
| Content | `sequences.bed`: 7,840 windows (7,008 train / 419 valid / 413 test), hg38 |

Why: the DNALongBench `targets/` folder ships TFRecords without coordinates. The 28/2/2 TFRecord counts match Akita's `get_data.sh`.

## 3. Reference genome

Not needed for v1: sequences are embedded in the TFRecords. A UCSC hg38 FASTA exists at `datasets/reference_hg38/` (shared manifest, Part B) for later variant (H4) work.

## 4. G8 independent loop and boundary sets

Approved 2026-10-23 (`notes/2026-10-23_g8-resolved.md`). Downloaded with `scripts/download_g8.py OUT_DIR`, MD5-checked against portal metadata. Local: `datasets/g8/{loops,boundaries}/`, index in `datasets/g8/files.tsv`.

**Loops: ENCODE CTCF ChIA-PET, GRCh38 `loops` (preferred default), downloaded 2026-10-23.** Format: bedpe, 7 columns (anchor 1, anchor 2, PET count). These are ChIA-PIPE PET clusters with ≥ 3 PETs, all intra-chromosomal, **not** significance-filtered loop calls. A positive set needs a PET threshold, fixed before evaluation (open).

| Line | Experiment | File | Lab | Rows | Bytes | MD5 |
|---|---|---|---|---|---|---|
| H1hESC | ENCSR095SXC | ENCFF401IWZ | Charles Lee, JAX | 2,111,961 | 33,072,294 | 98354c097ec1bf32fac206656c620205 |
| GM12878 | ENCSR184YZV | ENCFF780PGS | Yijun Ruan, JAX | 1,181,427 | 18,513,400 | ad835b1606f745eb5ee3bfd3f2327bcf |
| IMR90 | ENCSR076TTY | ENCFF682YFU | Yijun Ruan, JAX | 471,557 | 7,780,283 | be6ce1cd0808fd811bada14306eb4cf8 |
| HCT116 | ENCSR278IZK | ENCFF431BUN | Charles Lee, JAX | 276,384 | 4,171,667 | 89e034b1f5ca2e20fb7f1a19b783bc5f |
| HFF | — | — | — | no CTCF ChIA-PET with processed loops on ENCODE or 4DN | | |

Licence: ENCODE data, no restrictions on use (ENCODE data-use policy).

**Boundaries: 4DN processed `boundaries` files (GRCh38), downloaded 2026-10-23 with a 4DN access key.** 4DN refuses anonymous downloads (HTTP 403). The key file is local only and git-ignored. Format: bed, 5 columns (interval, Strong/Weak, boundary strength), from the 4DN insulation pipeline.

| Line | Set | File | Resolution | Rows | Bytes | MD5 |
|---|---|---|---|---|---|---|
| HFF | 4DNESNMAAN97 | 4DNFI3CGVRKU | 5 kb | 10,366 | 171,021 | f84bb7e06f594ab6864f0d506f76bf45 |
| H1hESC | 4DNESFSCP5L8 | 4DNFICEHILOK | 5 kb | 10,577 | 174,155 | dd33c22f03b976caf538128a28c0701b |
| GM12878 | 4DNESLLTENG9 | 4DNFIQBWXNL1 | 10 kb | 9,635 | 156,234 | 614216d99ef05ac72539145e765ab2f2 |
| IMR90 | 4DNES4296C5Q | 4DNFIUI2RM2Q | 10 kb | 9,043 | 147,126 | f3f4d4720bfe17f10a14b5b5d36ed42d |
| HCT116 | 4DNESNSTBMBY | 4DNFIG49IPR6 | 5 kb | 8,246 | 136,353 | d947aa77b6ace12a8ee615548ad0e831 |

Licence: 4DN data, publicly released (4DN data-use guidelines: cite the producing lab and publication).
