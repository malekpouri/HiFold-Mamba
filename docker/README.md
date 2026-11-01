# Running HiFold-Mamba in Docker

The image contains the code of one git commit and the locked environment from `requirements.lock.txt`: Python 3.12, torch 2.14.1 with CUDA 13 bundled, and gcc for Triton. Data is not inside the image. It is mounted at run time.

**Host requirements:**
- an NVIDIA driver for CUDA 13 (≥ 580; check with `nvidia-smi`);
- Docker with the NVIDIA Container Toolkit.

The wheels support sm_75 to sm_120, i.e. T4 up to RTX 50xx.

## Build and transfer (authors' workstation)

```bash
docker/build.sh            # needs a clean commit; writes dist/hifold-mamba_<sha>.tar.gz (+ .sha256)
```

Copy two things to the target machine:
- the `.tar.gz` and `.sha256` files;
- the DNALongBench cache (`cache/akita5`, 15 GB; `cache/h3`, 34 GB, needed only for H3).

## On the target machine

```bash
sha256sum -c hifold-mamba_<sha>.tar.gz.sha256
docker load -i hifold-mamba_<sha>.tar.gz

CACHE=/path/to/dnalongbench/cache      # contains akita5/ and h3/
WORK=/path/to/hifold-work              # outputs; must be writable
mkdir -p "$WORK"
alias hifold='docker run --rm --gpus all --shm-size=16g --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$CACHE",dst=/data/cache,readonly -v "$WORK":/work hifold-mamba:<sha>'

hifold check                 # GPU, CUDA, bf16, data mount
hifold test -m "not slow"    # unit tests (one test skips: it needs the hg38 FASTA)
hifold profile               # speed test: step time and peak memory per batch size
hifold train hifold_paper seed=0        # writes $WORK/runs/<name>-<hash>-s0/
```

**Long runs:**
- Use `docker run -d --name <run>` instead of `--rm`, so a run survives logout.
- Follow a run with `docker logs -f <run>`.

**Paths:** the configs refer to the authors' data path. Inside the image that path points to `/data/cache`, so configs and config hashes are the same as in the runs on the authors' workstation.

**Commit:** the commit the image was built from is recorded in each run's metadata.
