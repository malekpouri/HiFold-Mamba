"""Sanity checks on DNALongBench contact-map TFRecords (protocol G1).

Reports per cell line: value range, clip saturation, NaN count, mean/SD, the one-hot integrity
of the sequence, and P(s) (mean target per diagonal). Writes a heatmap + P(s) figure.

Usage: python scripts/inspect_targets.py TFR_FILE LINE_NAMES(comma-separated) OUT_PNG [N_RECORDS]
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hifold.data.tfrecord import iter_examples
from hifold.eval.metrics import N_BINS, triu_to_matrix

SEQ_LEN = 1_048_576
N_TRIU = 99_681


def main(tfr, names, out_png, n_records=16):
    names = names.split(",")
    targets, seq_ok, n_seen = [], [], 0
    for ex in iter_examples(tfr):
        seq = np.frombuffer(ex["sequence"][0], np.uint8)
        tgt = np.frombuffer(ex["target"][0], np.float16).astype(np.float32)
        assert seq.size == SEQ_LEN * 4, f"sequence bytes {seq.size}"
        assert tgt.size == N_TRIU * len(names), f"target values {tgt.size} vs {N_TRIU}x{len(names)}"
        seq = seq.reshape(SEQ_LEN, 4)
        row_sum = seq.sum(1)
        seq_ok.append((np.isin(seq, (0, 1)).all(), (row_sum == 1).mean(), (row_sum == 0).mean()))
        targets.append(tgt.reshape(N_TRIU, len(names)))
        n_seen += 1
        if n_seen >= n_records:
            break
    t = np.stack(targets)  # (records, 99681, lines)
    print(f"{tfr}\nrecords inspected: {n_seen}; target array {t.shape}")
    print(f"sequence: binary {all(s[0] for s in seq_ok)}, "
          f"one-hot rows {np.mean([s[1] for s in seq_ok]):.4f}, all-zero rows (N) {np.mean([s[2] for s in seq_ok]):.4f}")
    for k, name in enumerate(names):
        v = t[..., k]
        fin = v[np.isfinite(v)]
        print(f"  {name:8s} nan {np.isnan(v).sum():6d}  min {fin.min():+.3f}  max {fin.max():+.3f}  "
              f"mean {fin.mean():+.4f}  sd {fin.std():.3f}  |v|>=1.99 {np.mean(np.abs(fin) >= 1.99):.4f}  "
              f"exact0 {np.mean(fin == 0):.4f}")

    fig, axes = plt.subplots(2, len(names), figsize=(3.2 * len(names), 6.4), squeeze=False)
    for k, name in enumerate(names):
        mat = triu_to_matrix(t[0, :, k])
        axes[0, k].imshow(mat, cmap="RdBu_r", vmin=-2, vmax=2)
        axes[0, k].set_title(f"{name}, record 0")
        axes[0, k].axis("off")
        ps = [np.nanmean([np.nanmean(np.diagonal(triu_to_matrix(t[r, :, k]), d)) for r in range(n_seen)])
              for d in range(2, N_BINS)]
        axes[1, k].plot(np.arange(2, N_BINS) * 2.048, ps)
        axes[1, k].axhline(0, color="grey", lw=0.5)
        axes[1, k].set_xscale("log")
        axes[1, k].set_xlabel("separation (kb)")
        axes[1, k].set_ylabel("mean target")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    print(f"figure: {out_png}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], *(int(a) for a in sys.argv[4:5]))
