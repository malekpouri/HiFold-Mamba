# Architecture and Training — HiFold-Mamba (v1)

Status: **locked by the user, except open items G1–G3, G6 and G10** (see `evaluation_protocol.md` §8). Check that the name "HiFold-Mamba" doesn't clash with existing tools/papers `[UNVERIFIED]`.

## 1. Architecture

| # | Module | Design | RC-equivariance requirement |
|---|---|---|---|
| 1 | Local sequence encoder | Conv1D (kernel 5–9), downsampling 1,048,576 bp → 512 bins of 2 kb (stack still to be specified) | RC-paired kernels, or both strands with shared weights |
| 2 | Bi-Mamba encoder | 8–12 blocks over 512 bins, Caduceus-PS / MambaDNA style; norm + FFN | Forward and RC paths share weights |
| 3 | Multi-scale state mixer | Shallow/intermediate/deep outputs, pyramidal pooling, back to 512 bins | Pooling and layer combination symmetric under RC |
| 4 | Pairwise head | $\hat{C} = \mathrm{symmetrize}\big(UV^\top/\sqrt{d} + b_{\lvert i-j\rvert}\big)$, cropped to 448×448 | $U, V$ transformed so that RC of the input gives a 180° rotation of $\hat{C}$ |
| 5 | Structure terms | No separate heads. Insulation, boundaries and loops computed **from $\hat{C}$**. **No energy head** (removed: in log/O/E space it is just the map with its sign flipped) | Operators applied to the map; RC-safe if the map is |
| 6 | Cell-type adapter (mandatory) | Embedding $e_c$; FiLM in every Bi-Mamba block: $\gamma_c, \beta_c = \mathrm{MLP}(e_c)$, $Z \leftarrow \gamma_c \odot Z + \beta_c$ | $\gamma_c, \beta_c$ depend only on the cell type, not on strand. They must also respect the channel pairing of the RC-equivariant representation |
| — | Distance bias $b_{\lvert i-j\rvert}$ | Learned | Depends only on $\lvert i-j\rvert$, so it's RC-safe |

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

Conditions (G6): exact test in fp32, plus a separate bf16 check with a looser tolerance. In train mode, disable dropout or use identical RNG masks.

## 2. Structure loss

$$\mathcal{L} = \mathcal{L}_{\text{contact}} + \mathcal{L}_{\text{struct}}, \qquad \mathcal{L}_{\text{struct}} = \lambda_{\text{ins}}\mathcal{L}_{\text{ins}} + \lambda_{\text{bnd}}\mathcal{L}_{\text{bnd}} + \lambda_{\text{loop}}\mathcal{L}_{\text{loop}}$$

The user's version also contained $\lambda_{\text{sym}}\lVert\hat{C}-\hat{C}^\top\rVert_1$. It is identically 0 because $\hat{C}$ is symmetrized (G3). Symmetry stays as an architectural prior.

### Differentiable insulation (F1, with the G2 correction pending)

Map to contact scale with a differentiable transform $g$, applied **identically** to the prediction and the target (G2):

$$\tilde{C} = g(\hat{C}), \qquad \tilde{C}^{\text{true}} = g(C^{\text{target}})$$

User's proposal: $g = \operatorname{softplus}$ for the prediction, raw contacts for the truth. That isn't consistent (G2). Recommended: $g = \exp$ on both sides if the target is log O/E.

$$\hat{I}_i = \frac{1}{\lvert\Omega_i\rvert}\sum_{(a,b)\in\Omega_i}\tilde{C}_{ab}, \qquad \Omega_i = \{(a,b) : i-w \le a < i < b \le i+w,\ a,b \text{ unmasked}\}$$

$I^{\text{true}}_i$ is the same operator on $\tilde{C}^{\text{true}}$, and $\mathcal{L}_{\text{ins}} = \frac{1}{B}\sum_i(\hat{I}_i - I^{\text{true}}_i)^2$.

- The diagonal is excluded ($a < i < b$). The sum covers contacts **crossing** $i$, not row $i$.
- The gradient reaches $\hat{C}$ directly.
- $w$ = 20 bins (40 kb) by default; sensitivity reported at $w = 10, 20, 40$. Insulation is defined for $i \in [w, 448-w)$.
- $\mathcal{L}_{\text{bnd}}$: a boundary loss derived from $\hat{I}$ (e.g. on the insulation drop/minima). The exact form is still to be specified.
- $\mathcal{L}_{\text{loop}}$: on the predicted map. HiCCUPS / CTCF–CTCF positives + distance-matched negatives, training regions only.
- $\mathcal{L}_{\text{contact}}$: masked MSE in the target space (the target space itself depends on G1).

### Physics/structure content of v1
Differentiable insulation from $\hat{C}$, the boundary loss derived from insulation, the loop loss on the map, the learnable distance bias, and architectural symmetry.

## 3. Training

| Item | Final |
|---|---|
| Optimizer | AdamW, weight decay 1e-2 |
| Learning rate | 3e-4, 1,000-step warmup, cosine decay |
| Precision | bf16 mixed precision (a practical choice for stability and compatibility with Mamba kernels; no stronger claim) |
| Compute | Assume a 24–40 GB GPU, batch 1–4 + gradient accumulation, effective batch 8–16. **8 GB fallback:** 4 Bi-Mamba layers, hidden 128. Choose the paper configuration before training (G10) |
| Weights | EMA |
| Checkpoint selection | Mean validation HiCRep SCC (stratum-adjusted) |
| Data | Official DNALongBench training split, 5 cell lines |
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
