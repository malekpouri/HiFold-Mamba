# HiFold-Mamba runtime image (GPU). Build with docker/build.sh, not directly.
#
# Named build contexts (supplied by build.sh):
#   sitepkgs : site-packages of the project .venv, which matches requirements.lock.txt exactly
#              (torch 2.14.1+cu130 with its bundled CUDA 13 libraries). This avoids re-downloading ~5 GB.
#   code     : `git archive` of the commit being packaged (src, scripts, configs, tests, ...)
#
# The host needs only an NVIDIA driver for CUDA 13 (>= 580) and the NVIDIA Container Toolkit.
FROM python:3.12-slim-bookworm

# Triton (used by torch.compile for the selective scan) compiles a small C launcher at run time,
# so a C compiler is required; git lets train.py record the commit if a checkout is mounted.
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libc6-dev git \
 && rm -rf /var/lib/apt/lists/*

COPY --from=sitepkgs . /usr/local/lib/python3.12/site-packages/
RUN rm -f /usr/local/lib/python3.12/site-packages/_virtualenv.pth \
          /usr/local/lib/python3.12/site-packages/_virtualenv.py

COPY --from=code . /app
COPY entrypoint.sh /usr/local/bin/hifold
RUN chmod 755 /usr/local/bin/hifold

# The configs point at the authors' data path. Mapping that path to /data/cache keeps every
# config, and therefore every config hash and run-directory name, identical to the host runs.
RUN mkdir -p "/mnt/storage/datasets/Human genomes Dataset/new_idea/datasets/dnalongbench" \
 && ln -s /data/cache "/mnt/storage/datasets/Human genomes Dataset/new_idea/datasets/dnalongbench/cache" \
 && mkdir -p /data/cache /work && chmod 1777 /work

ARG HIFOLD_COMMIT=unknown
ENV HIFOLD_COMMIT=${HIFOLD_COMMIT} \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    HOME=/work \
    TRITON_CACHE_DIR=/work/.cache/triton \
    TORCHINDUCTOR_CACHE_DIR=/work/.cache/inductor

LABEL org.opencontainers.image.title="hifold-mamba" \
      org.opencontainers.image.source="https://github.com/malekpouri/HiFold-Mamba" \
      org.opencontainers.image.revision="${HIFOLD_COMMIT}" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app
ENTRYPOINT ["hifold"]
CMD ["help"]
