# Architecture and Training — HiFold-Mamba (v1)

Status: **locked.** G1/G2 are provisional until the DNALongBench files confirm the target (see `evaluation_protocol.md` §8). G3, G6 and G10 were resolved on 2026-10-21. Check that the name "HiFold-Mamba" doesn't clash with existing tools/papers `[UNVERIFIED]`.

## 1. Architecture

| # | Module | Design | RC-equivariance requirement |
|---|---|---|---|
| 1 | Local sequence encoder | Conv1D (kernel 5–9), downsampling 1,048,576 bp → 512 bins of 2 kb (stack still to be specified) | RC-paired kernels, or both strands with shared weights |
| 2 | Bi-Mamba encoder | 8–12 blocks over 512 bins, Caduceus-PS / MambaDNA style; norm + FFN | Forward and RC paths share weights |
| 3 | Multi-scale state mixer | Shallow/intermediate/deep outputs, pyramidal pooling, back to 512 bins | Pooling and layer combination symmetric under RC |
| 4 | Pairwise head | $\hat{C} = \mathrm{symmetrize}\big(UV^\top/\sqrt{d} + b_{\lvert i-j\rvert}\big)$, cropped to 448×448 | $U, V$ transformed so that RC of the input gives a 180° rotation of $\hat{C}$ |
| 5 | Structure terms | No separate heads. Insulation, boundaries and loops computed **from $\hat{C}$**. **No energy head** (removed: in log/O/E space it is just the map with its sign flipped) | Operators applied to the map; RC-safe if the map is |
| 6 | Cell-type adapter (mandatory) | Embedding $e_c$; FiLM in every Bi-Mamba block: $\gamma_c, \beta_c = \mathrm{MLP}(e_c)$, $Z \leftarrow \gamma_c \odot Z + \beta_c$ | $\gamma_c, \beta_c$ depend only on the cell type, not on strand. They must also respect the channel pairing of the RC-equivariant representation |
| — | Distance bias $b_{c,\lvert i-j\rvert}$ | Learned, **per cell line** (decided 2026-10-23) | Depends only on $\lvert i-j\rvert$ and the cell line, so it's RC-safe. The shared $b_{\lvert i-j\rvert}$ can't express the line-specific residual P(s) seen in the targets (trough −0.17 to −0.30). A new H3 line gets its own bias, learned in fine-tuning |

Global property: $f(\mathrm{RC}(x), c) = \mathrm{rot}_{180}(f(x, c))$ with $\mathrm{rot}_{180}(C)_{ij} = C_{n-1-i,\,n-1-j}$. The RC-equivariance claim is about the **whole network**, not just the Mamba block.

### Mandatory equivariance unit test (before training)

Specification (written by the user; the implementation follows when code is allowed):

```python
y = model(x, cell_type)
y_rc = model(reverse_complement(x), cell_type)
assert allclose(y, rot180(y_rc), atol=1e-4)
```

Must pass for:
- a random sequence
- a real chromosome sequence
- input with a cell-type embedding
- train and eval mode
- with and without gradient checkpointing

Conditions (G6, resolved): exact test in fp32 (atol 1e-4, rtol 1e-5), plus a separate bf16 check (rtol 1e-2). In train mode, set dropout to 0 or use identical RNG masks.

## 2. Structure loss

$$\mathcal{L} = \mathcal{L}_{\text{contact}} + \mathcal{L}_{\text{struct}}, \qquad \mathcal{L}_{\text{struct}} = \lambda_{\text{ins}}\mathcal{L}_{\text{ins}} + \lambda_{\text{bnd}}\mathcal{L}_{\text{bnd}} + \lambda_{\text{loop}}\mathcal{L}_{\text{loop}}$$

The user's version also contained $\lambda_{\text{sym}}\lVert\hat{C}-\hat{C}^\top\rVert_1$. It is identically 0 because $\hat{C}$ is symmetrized, so it is dropped (G3, resolved). Symmetry stays as an architectural prior.

### Differentiable insulation (F1, G2 provisional)

Map to contact scale with a differentiable transform $g$, applied **identically** to the prediction and the target (G2):

$$\tilde{C} = g(\hat{C}), \qquad \tilde{C}^{\text{true}} = g(C^{\text{target}}), \qquad g(x) = \exp\!\big(2\tanh(x/2)\big)$$

$g$ is a soft-clipped exp: smooth, bounded to $[e^{-2}, e^{2}]$ to match the target's (−2, 2) natural-log clip, and ≈ exp near 0. It replaces the original "softplus on the prediction, raw contacts on the truth", which was inconsistent (G2). Final once G1 confirms the log base and range in the files.

$$\hat{I}_i = \frac{1}{\lvert\Omega_i\rvert}\sum_{(a,b)\in\Omega_i}\tilde{C}_{ab}, \qquad \Omega_i = \{(a,b) : i-w \le a < i < b \le i+w,\ b-a \ge d_{\min},\ a,b \text{ unmasked}\}$$

$I^{\text{true}}_i$ is the same operator on $\tilde{C}^{\text{true}}$, and $\mathcal{L}_{\text{ins}}$ is the mean of $(\hat{I}_i - I^{\text{true}}_i)^2$ over all valid $i$ (non-empty $\Omega_i$, $i \in [w, 448-w)$) and all windows in the batch.

- The diagonal is excluded ($a < i < b$), and so is the benchmark's near-diagonal band ($b - a < d_{\min}$; DNALongBench uses diagonal offset 2, so $d_{\min} = 2$, which $a < i < b$ already enforces; confirm in the files under G1). The sum covers contacts **crossing** $i$, not row $i$.
- The gradient reaches $\hat{C}$ directly.
- $w$ = 20 bins (40 kb) by default; sensitivity reported at $w = 10, 20, 40$. Insulation is defined for $i \in [w, 448-w)$.
- $\mathcal{L}_{\text{bnd}}$ (specified 2026-10-23): masked MSE between boundary-strength profiles, $\mathcal{L}_{\text{bnd}} = \operatorname{mean}_i \big(s_i(\hat{I}) - s_i(I^{\text{true}})\big)^2$ with $s_i(I) = \overline{\log_2 I}_{[i-5,\,i+5]} - \log_2 I_i$ (depth of the insulation dip relative to its ±5-bin neighbourhood). Same operator on both maps and the same $s$ as the evaluation's boundary score. NaN positions are masked. It is a regression on dip depth, so it needs no boundary calls from the training data.
- $\mathcal{L}_{\text{loop}}$: on the predicted map. HiCCUPS / CTCF–CTCF positives + distance-matched negatives, training regions only.
- $\mathcal{L}_{\text{contact}}$: masked MSE in the target space (the target space itself depends on G1).

### Structural content of v1
Differentiable insulation from $\hat{C}$, the boundary loss derived from insulation, the loop loss on the map, the learnable distance bias, and architectural symmetry.

## 3. Training

| Item | Final |
|---|---|
| Optimizer | AdamW, weight decay 1e-2 |
| Learning rate | 3e-4, 1,000-step warmup, cosine decay |
| Precision | bf16 mixed precision (a practical choice for stability and compatibility with Mamba kernels; no stronger claim) |
| Compute | Assume a 24–40 GB GPU, batch 1–4 + gradient accumulation, effective batch 8–16. **8 GB fallback:** 4 Bi-Mamba layers, hidden 128, for development only. The paper configuration is the 24–40 GB model (G10, resolved) |
| Weights | EMA |
| Checkpoint selection | Mean validation HiCRep SCC (stratum-adjusted) |
| Data | Official DNALongBench training split, 5 training cell lines (the 4 H3 held-out lines are never used here) |
| Schedule | $\mathcal{L}_{\text{contact}}$ until convergence, then $\mathcal{L}_{\text{struct}}$ with small weights. The Base model gets the same total steps |

## 4. Efficiency claims

Only for 4–8 Mb inputs (2,048–4,096 bins) or 128 bp tokens (~8,192 per 1 Mb). Never at 512 bins.

## 5. Required ablations

- Bi-Mamba → matched Transformer (same parameters, data and schedule).
- Without RC equivariance.
- Without $\mathcal{L}_{\text{struct}}$ (same compute), and each term on its own (ins / bnd / loop).
- Without the multi-scale mixer.
- Insulation window $w$ = 10 / 20 / 40.
- Optional: without the distance bias; biaffine only vs. + a small symmetric, RC-equivariant 2D refinement.

## 6. Still to specify (before coding)

- Conv downsampling stack (channels, pooling/stride per layer).
- Exact RC pairing of $U$/$V$ and FiLM channels.
- Form of $\mathcal{L}_{\text{bnd}}$.
- Hyperparameters per run in `configs/`: d_model, d_state, expand factor, FFN size, pyramid levels, λ values and their schedule, EMA decay, early-stopping patience.
