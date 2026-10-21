"""Convert DNALongBench contact-map TFRecords into memory-mapped arrays for training.

Per split it writes, into OUT_DIR/<split>/:
  seq.u8      (N, 1_048_576) uint8 base codes A=0 C=1 G=2 T=3 (4 = no base)
  target.f16  (N, 99_681, n_lines) float16, exactly as shipped
  windows.tsv  chrom, start, end per record (from sequences.bed, in record order)
  meta.json    shapes, line names, source files

CRC checks are skipped (file MD5s were verified at download). Instead, record i is checked
against window i of sequences.bed by comparing the embedded sequence with hg38 at a sample
of records; any mismatch aborts.

Usage: python scripts/build_cache.py TFR_DIR SEQUENCES_BED LINE_NAMES OUT_DIR HG38_FA [split ...]
"""

import json
import sys
from pathlib import Path

import numpy as np

from hifold.data.fasta import Fasta
from hifold.data.tfrecord import iter_examples, iter_records, parse_example

SEQ_LEN = 1_048_576
N_TRIU = 99_681
CODE = np.frombuffer(b"ACGT", np.uint8)


def onehot_to_codes(seq_u8: np.ndarray) -> np.ndarray:
    oh = seq_u8.reshape(SEQ_LEN, 4)
    codes = oh.argmax(1).astype(np.uint8)
    codes[oh.sum(1) == 0] = 4
    return codes


def codes_to_str(codes: np.ndarray) -> str:
    lut = np.frombuffer(b"ACGTN", np.uint8)
    return lut[codes].tobytes().decode()


def tfr_files(tfr_dir: Path, split: str):
    files = sorted(tfr_dir.glob(f"{split}-*.tfr"), key=lambda p: int(p.stem.split("-")[1]))
    if not files:
        raise FileNotFoundError(f"no {split}-*.tfr in {tfr_dir}")
    return files


def build_split(tfr_dir, windows, names, out_dir, fasta, split, check_every=97):
    files = tfr_files(tfr_dir, split)
    wins = [w for w in windows if w[3] == split]
    n, L = len(wins), len(names)
    out = out_dir / split
    out.mkdir(parents=True, exist_ok=True)
    seq_mm = np.lib.format.open_memmap(out / "seq.u8.npy", mode="w+", dtype=np.uint8, shape=(n, SEQ_LEN))
    tgt_mm = np.lib.format.open_memmap(out / "target.f16.npy", mode="w+", dtype=np.float16, shape=(n, N_TRIU, L))
    i = 0
    n_checked = mismatches = 0
    for f in files:
        for rec in iter_records(f, check_crc=False):
            ex = parse_example(rec)
            if i >= n:
                raise RuntimeError(f"{split}: more records than windows in sequences.bed ({n})")
            codes = onehot_to_codes(np.frombuffer(ex["sequence"][0], np.uint8))
            seq_mm[i] = codes
            tgt_mm[i] = np.frombuffer(ex["target"][0], np.float16).reshape(N_TRIU, L)
            if fasta is not None and (i % check_every == 0 or i == n - 1):
                chrom, start, end, _ = wins[i]
                ref = np.frombuffer(fasta.fetch(chrom, start, end).encode(), np.uint8)
                got = np.frombuffer(codes_to_str(codes).encode(), np.uint8)
                both = (ref != ord("N")) & (got != ord("N"))
                agree = np.mean(ref[both] == got[both]) if both.any() else np.nan
                n_checked += 1
                if not agree > 0.99:
                    mismatches += 1
                    print(f"  MISMATCH record {i} {chrom}:{start}-{end} agreement {agree:.4f}", flush=True)
            i += 1
        print(f"  {split}: {f.name} done, {i}/{n} records", flush=True)
    if i != n:
        raise RuntimeError(f"{split}: {i} records but {n} windows")
    if mismatches:
        raise RuntimeError(f"{split}: {mismatches}/{n_checked} sampled records do not match hg38")
    seq_mm.flush()
    tgt_mm.flush()
    with open(out / "windows.tsv", "w") as fh:
        for chrom, start, end, _ in wins:
            fh.write(f"{chrom}\t{start}\t{end}\n")
    meta = {"split": split, "n": n, "lines": names, "seq_len": SEQ_LEN, "n_triu": N_TRIU,
            "files": [p.name for p in files], "hg38_checked_records": n_checked}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{split}: {n} records cached; {n_checked} sampled records match hg38", flush=True)


def main(tfr_dir, bed, names, out_dir, hg38, *splits):
    windows = []
    with open(bed) as fh:
        for line in fh:
            c, s, e, sp = line.split()[:4]
            windows.append((c, int(s), int(e), sp))
    fasta = Fasta(hg38, keep={w[0] for w in windows}) if hg38 != "-" else None
    for split in splits or ("train", "valid", "test"):
        build_split(Path(tfr_dir), windows, names.split(","), Path(out_dir), fasta, split)


if __name__ == "__main__":
    main(*sys.argv[1:])
