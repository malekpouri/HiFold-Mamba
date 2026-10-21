"""Download the DNALongBench contact-map files from Harvard Dataverse, with MD5 checks.

Dataset: doi:10.7910/DVN/AZM25S (CC0). Files already present with the right MD5 are
skipped, so the script can be re-run to resume.

Usage: python -I scripts/download_dnalongbench.py OUT_DIR [PREFIX ...]
  PREFIX filters on the Dataverse directory label, e.g. contact_map_prediction/targets
"""

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

API = "https://dataverse.harvard.edu/api"
DOI = "doi:10.7910/DVN/AZM25S"
# Dataverse returns 403 for urllib's default User-Agent.
HEADERS = {"User-Agent": "hifold-mamba-downloader/0.1 (+https://github.com/malekpouri/HiFold-Mamba)"}


def _get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120)


def md5sum(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def list_files():
    with _get(f"{API}/datasets/:persistentId/?persistentId={DOI}") as r:
        version = json.load(r)["data"]["latestVersion"]
    return version["files"]


def fetch(file_id, dest, tries=5):
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, tries + 1):
        try:
            with _get(f"{API}/access/datafile/{file_id}") as r, open(tmp, "wb") as fh:
                while block := r.read(1 << 22):
                    fh.write(block)
            tmp.rename(dest)
            return
        except OSError as err:
            print(f"  attempt {attempt} failed: {err}", flush=True)
            time.sleep(10 * attempt)
    raise RuntimeError(f"giving up on {dest}")


def main(out_dir, *prefixes):
    out = Path(out_dir)
    files = [f for f in list_files()
             if not prefixes or any(f.get("directoryLabel", "").startswith(p) for p in prefixes)]
    total = sum(f["dataFile"]["filesize"] for f in files)
    print(f"{len(files)} files, {total / 1e9:.2f} GB", flush=True)
    for i, f in enumerate(sorted(files, key=lambda f: (f.get("directoryLabel", ""), f["dataFile"]["filename"])), 1):
        df = f["dataFile"]
        dest = out / f.get("directoryLabel", "") / df["filename"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and md5sum(dest) == df["md5"]:
            continue
        print(f"[{i}/{len(files)}] {dest.relative_to(out)} ({df['filesize'] / 1e6:.0f} MB)", flush=True)
        fetch(df["id"], dest)
        if md5sum(dest) != df["md5"]:
            dest.unlink()
            raise RuntimeError(f"MD5 mismatch: {dest}")
    print("done; all MD5 verified", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
