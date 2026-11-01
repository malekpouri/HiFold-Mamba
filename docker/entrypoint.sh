#!/bin/sh
# HiFold-Mamba container entry point.
#   /data/cache : the DNALongBench cache (contains akita5/ and h3/), mounted read-only
#   /work       : writable output directory (runs/, logs, compiler caches)
set -e
cd /app
cmd="${1:-help}"
[ "$#" -gt 0 ] && shift

case "$cmd" in
  help)
    cat <<'EOF'
HiFold-Mamba container. Commands:
  check                 GPU, CUDA, bf16 and data-mount check (run this first)
  profile [paper|fallback]
                        step time and peak memory per batch size (PROFILE_DTYPE=bf16|fp16|fp32)
  test [pytest args]    unit tests (add -m "not slow" for the quick subset)
  train CONFIG [key=value ...]
                        e.g. train hifold_paper seed=1 ; runs are written to /work/runs
  evaluate ARGS         scripts/evaluate.py, e.g. evaluate checkpoint /work/runs/<run>/best.pt valid /work/out.tsv
  dashboard [--port N]  read-only web monitor of /work/runs, reports in /work/reports
                        (publish with -p 127.0.0.1:8080:8080; HIFOLD_DASH_TOKEN=... to require a token)
  python ARGS | bash    anything else
EOF
    ;;
  check)
    python - <<'EOF'
import os, torch
print(f"commit {os.environ.get('HIFOLD_COMMIT')}  torch {torch.__version__}  CUDA {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("NO GPU visible: run with --gpus all (NVIDIA Container Toolkit) and a driver >= 580")
p = torch.cuda.get_device_properties(0)
print(f"GPU {p.name}  capability sm_{p.major}{p.minor}  memory {p.total_memory / 2**30:.1f} GiB")
print(f"wheel archs {torch.cuda.get_arch_list()}")
print(f"bf16 native {torch.cuda.is_bf16_supported(including_emulation=False)}")
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print(f"bf16 matmul ok, |x@x| = {(x @ x).float().norm():.1f}")
from hifold.model.ssm import selective_scan  # noqa: F401
for sub in ("akita5", "h3"):
    d = f"/data/cache/{sub}"
    print(f"data {d}: {'present, ' + str(len(os.listdir(d))) + ' entries' if os.path.isdir(d) else 'MISSING'}")
print("work dir writable:", os.access("/work", os.W_OK))
EOF
    ;;
  profile) exec python scripts/profile_memory.py "$@" ;;
  test)    exec python -m pytest -q "$@" ;;
  train)
    [ -n "$1" ] || { echo "usage: train CONFIG [key=value ...]"; exit 2; }
    cfg="$1"; shift
    case "$cfg" in */*|*.yaml) ;; *) cfg="configs/$cfg.yaml" ;; esac
    mkdir -p /work/runs
    exec python -m hifold.train "$cfg" run_root=/work/runs "$@"
    ;;
  evaluate) exec python scripts/evaluate.py "$@" ;;
  dashboard)
    mkdir -p /work/runs /work/reports
    exec python -m hifold.monitor --runs /work/runs --reports /work/reports \
         --programme configs/programme.yaml --host 0.0.0.0 "$@"
    ;;
  *) exec "$cmd" "$@" ;;
esac
