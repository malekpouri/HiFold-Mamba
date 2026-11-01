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

## Web dashboard (progress and reports)

The dashboard runs as a second container next to the training runs. It only **reads** `$WORK/runs`: it never starts or stops a run, and never reads test-set results. Generated reports are written to `$WORK/reports`.

```bash
docker run -d --name hifold-dash --restart unless-stopped --gpus all \
  -p 127.0.0.1:8080:8080 --user "$(id -u):$(id -g)" -v "$WORK":/work hifold-mamba:<sha> dashboard
```

`--gpus all` only lets the dashboard show GPU load. It uses no GPU memory.

**Opening it from your own computer.** The port is published on 127.0.0.1 only, so it is not open to the university network. Use an SSH tunnel and browse to http://localhost:8080:

```bash
ssh -L 8080:localhost:8080 you@university-machine
```

To publish it on the network instead, use `-p 8080:8080` **and** a token: add `-e HIFOLD_DASH_TOKEN=<long random string>`, then open `http://host:8080/?token=<string>` once. A cookie keeps you signed in.

**What it shows:**
- every run: state, progress, speed, time left, best validation SCC, curves, config and errors;
- the planned Stage 2 programme from `configs/programme.yaml`;
- GPU load and free disk space.

The page refreshes every 15 s. A link to one run is `/#run=<run name>`.

**Run states:**

| State | Meaning |
|---|---|
| running | log lines are arriving |
| finished | the run reached its last step |
| failed | the run stopped with an error, or was stopped. The error and traceback are shown. |
| stalled | the status file says running, but no log line for 60 min (`HIFOLD_STALE_MIN`). The process was probably killed, ran out of memory or lost power. Check `docker ps -a` and `docker logs`. |

**Reports:** "Generate report now" writes `report-<date>.html` (open it and print to PDF), `.md` and `.csv` into `$WORK/reports`.

**Naming runs for the programme:**
- λ pilot runs: `name=lambda_pilot_<k>`;
- H3 runs: `name=h3_<...>`;
- robustness runs: `name=robust_<...>`.

`name` does not enter the config hash, so renaming a run does not change what is trained.
