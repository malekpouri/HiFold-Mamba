#!/usr/bin/env bash
# Build the HiFold-Mamba image from the current commit and save it as a transferable file.
#   docker/build.sh            -> image hifold-mamba:<short sha> + dist/hifold-mamba_<short sha>.tar.gz
#                                 + dist/run-queue.ps1 (replaces any earlier files in dist/)
#   docker/build.sh --no-save  -> image only
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo"

if [ -n "$(git status --porcelain -- src scripts configs tests docker pyproject.toml requirements.lock.txt)" ]; then
  echo "Uncommitted changes in packaged files: commit first so the image matches a commit." >&2
  exit 1
fi
sha="$(git rev-parse HEAD)"; short="${sha:0:7}"

# the environment copied into the image must be exactly the lock file
if ! diff <(uv pip freeze --python .venv/bin/python | sort) \
          <(grep -v -e '^#' -e '^$' requirements.lock.txt | sort) >/dev/null; then
  echo ".venv does not match requirements.lock.txt" >&2
  exit 1
fi

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
git archive HEAD | tar -x -C "$tmp"

docker build -f docker/Dockerfile \
  --build-context sitepkgs=.venv/lib/python3.12/site-packages \
  --build-context code="$tmp" \
  --build-arg HIFOLD_COMMIT="$sha" \
  -t "hifold-mamba:$short" -t hifold-mamba:latest \
  docker

if [ "${1:-}" != "--no-save" ]; then
  mkdir -p dist
  rm -f dist/hifold-mamba_*.tar.gz dist/hifold-mamba_*.tar.gz.sha256 dist/run-queue.ps1   # one transferable set only
  out="dist/hifold-mamba_${short}.tar.gz"
  docker save "hifold-mamba:$short" | gzip -1 > "$out"
  (cd dist && sha256sum "${out#dist/}" > "${out#dist/}.sha256")
  sed "s/IMAGE_TAG/$short/" docker/run-queue.ps1 > dist/run-queue.ps1     # Windows queue for this image
  echo "saved $out ($(du -h "$out" | cut -f1)) and dist/run-queue.ps1"
fi
