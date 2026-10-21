"""Minimal random-access FASTA reader for an uncompressed, fixed-line-width FASTA (no .fai needed).

Builds an index by one sequential scan (cached next to nothing: kept in memory) and returns
0-based, half-open slices.
"""

from pathlib import Path


class Fasta:
    def __init__(self, path, keep=None):
        self.path = Path(path)
        self.index = {}  # name -> (length, byte offset of first base, bases per line, bytes per line)
        with open(self.path, "rb") as fh:
            name = None
            offset = 0
            for line in fh:
                if line.startswith(b">"):
                    name = line[1:].split()[0].decode()
                    seq_start, length, line_bases, line_bytes = offset + len(line), 0, None, None
                    if keep is None or name in keep:
                        self.index[name] = [0, seq_start, None, None]
                elif name in self.index:
                    entry = self.index[name]
                    if entry[2] is None:
                        entry[2], entry[3] = len(line.rstrip(b"\r\n")), len(line)
                    entry[0] += len(line.rstrip(b"\r\n"))
                offset += len(line)
        self._fh = open(self.path, "rb")

    def fetch(self, chrom: str, start: int, end: int) -> str:
        length, base0, lb, lbytes = self.index[chrom]
        if not (0 <= start <= end <= length):
            raise ValueError(f"{chrom}:{start}-{end} outside 0..{length}")
        first = base0 + (start // lb) * lbytes + start % lb
        last = base0 + (end // lb) * lbytes + end % lb
        self._fh.seek(first)
        raw = self._fh.read(last - first)
        return raw.replace(b"\n", b"").replace(b"\r", b"").decode().upper()
