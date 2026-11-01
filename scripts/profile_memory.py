"""Peak GPU memory and step time for one training step at full input length.

Usage: python scripts/profile_memory.py [config_name ...]   (names below; default: all)
Precision: PROFILE_DTYPE=bf16 (default) | fp16 | fp32. fp16 is for GPUs without bf16 tensor cores
(e.g. Tesla T4); this only measures speed/memory, it does not change the pre-registered training.
"""

import os

import sys
import time

import torch

from hifold.model.hifold import HiFold, HiFoldConfig
from hifold.train import contact_loss, to_triu, triu_index

DTYPE = os.environ.get("PROFILE_DTYPE", "bf16")
AMP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[DTYPE]

CONFIGS = {
    "paper": HiFoldConfig(),
    "fallback": HiFoldConfig(n_blocks=4, d_model=128, tower_channels=[32, 32, 48, 48, 64, 64, 80, 80, 96, 96, 128]),
}


def profile(name, cfg, batch, n_cells=5, steps=3):
    dev = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = HiFold(cfg).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    iu = triu_index(cfg.out_bins, device=dev)
    x = torch.nn.functional.one_hot(torch.randint(0, 4, (batch, cfg.seq_len), device=dev), 4).float()
    cells = torch.arange(n_cells, device=dev).repeat(batch, 1)
    y = torch.randn(batch, n_cells, len(iu[0]), device=dev)
    times = []
    for s in range(steps):
        torch.cuda.synchronize()
        t = time.time()
        with torch.autocast("cuda", dtype=AMP or torch.float32, enabled=AMP is not None):
            pred = model(x, cells)
        loss = contact_loss(to_triu(pred.float(), iu), y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.time() - t)
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"{name:9s} batch {batch} x {n_cells} cells: peak {peak:5.2f} GiB, "
          f"[{DTYPE}] step {min(times[1:]):.2f} s (first {times[0]:.2f} s)", flush=True)
    del model, opt
    return peak


if __name__ == "__main__":
    free, total = torch.cuda.mem_get_info()
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, arch list {torch.cuda.get_arch_list()}, "
          f"capability {torch.cuda.get_device_capability(0)}, bf16 native {torch.cuda.is_bf16_supported(including_emulation=False)}")
    print(f"GPU {torch.cuda.get_device_name(0)}: free {free / 2**30:.2f} / {total / 2**30:.2f} GiB")
    for name in sys.argv[1:] or CONFIGS:
        for batch in (1, 2, 4):
            try:
                profile(name, CONFIGS[name], batch)
            except torch.OutOfMemoryError:
                print(f"{name:9s} batch {batch}: OOM", flush=True)
                torch.cuda.empty_cache()
                break
