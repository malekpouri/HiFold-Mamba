"""Download the approved G8 independent boundary and loop sets (notes/2026-10-23_g8-resolved.md).

  python3 -I scripts/download_g8.py OUT_DIR

Boundaries: 4DN processed `boundaries` files (GRCh38) of an independent Hi-C set per line.
Loops: ENCODE CTCF ChIA-PET `loops` (GRCh38, preferred default) per line; none exists for HFF.
MD5 is taken from the portal metadata and checked; verified files are skipped on rerun.
4DN serves file downloads only with an access key (free account, "Access Keys" in the profile):
set FOURDN_KEY and FOURDN_SECRET. Without them the 4DN files are skipped with a message.
"""

import base64
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

H = {"User-Agent": "hifold-g8-download", "Accept": "application/json"}

# (line, kind, portal, file accession, experiment set / experiment)
FILES = [
    ("HFF", "boundaries", "4dn", "4DNFI3CGVRKU", "4DNESNMAAN97"),
    ("H1hESC", "boundaries", "4dn", "4DNFICEHILOK", "4DNESFSCP5L8"),
    ("GM12878", "boundaries", "4dn", "4DNFIQBWXNL1", "4DNESLLTENG9"),
    ("IMR90", "boundaries", "4dn", "4DNFIUI2RM2Q", "4DNES4296C5Q"),
    ("HCT116", "boundaries", "4dn", "4DNFIG49IPR6", "4DNESNSTBMBY"),
    ("H1hESC", "loops", "encode", "ENCFF401IWZ", "ENCSR095SXC"),
    ("GM12878", "loops", "encode", "ENCFF780PGS", "ENCSR184YZV"),
    ("IMR90", "loops", "encode", "ENCFF682YFU", "ENCSR076TTY"),
    ("HCT116", "loops", "encode", "ENCFF431BUN", "ENCSR278IZK"),
]
BASE = {"4dn": "https://data.4dnucleome.org", "encode": "https://www.encodeproject.org"}


def meta(portal, acc):
    path = f"/{acc}/" if portal == "4dn" else f"/files/{acc}/"
    with urllib.request.urlopen(urllib.request.Request(BASE[portal] + path + "?format=json", headers=H), timeout=120) as r:
        return json.load(r)


def auth(portal):
    if portal == "4dn" and os.environ.get("FOURDN_KEY"):
        tok = base64.b64encode(f"{os.environ['FOURDN_KEY']}:{os.environ['FOURDN_SECRET']}".encode()).decode()
        return {"Authorization": "Basic " + tok}
    return {}


class _NoAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """4DN redirects to a pre-signed S3 URL, which rejects a second (Authorization) credential."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            new.remove_header("Authorization")
        return new


OPENER = urllib.request.build_opener(_NoAuthOnRedirect)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(out):
    out = Path(out)
    rows = []
    for line, kind, portal, acc, exp in FILES:
        m = meta(portal, acc)
        url = BASE[portal] + m["href"]
        dest = out / kind / f"{line}_{acc}{''.join(Path(m['href']).suffixes)}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not (dest.exists() and md5(dest) == m["md5sum"]):
            if portal == "4dn" and not auth(portal):
                print(f"skip {acc}: 4DN download needs FOURDN_KEY / FOURDN_SECRET", flush=True)
                continue
            tmp = dest.with_suffix(dest.suffix + ".part")
            with OPENER.open(urllib.request.Request(url, headers={"User-Agent": H["User-Agent"], **auth(portal)}), timeout=600) as r, \
                    open(tmp, "wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            if md5(tmp) != m["md5sum"]:
                sys.exit(f"MD5 mismatch: {acc}")
            tmp.rename(dest)
        rows.append((line, kind, portal, exp, acc, m.get("assembly") or m.get("genome_assembly"), m["md5sum"], dest.stat().st_size, url))
        print(*rows[-1], sep="\t", flush=True)
    with open(out / "files.tsv", "w") as fh:
        fh.write("line\tkind\tportal\texperiment\tfile\tassembly\tmd5\tbytes\turl\n")
        for r in rows:
            fh.write("\t".join(map(str, r)) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
