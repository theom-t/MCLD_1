---
name: mcld-ml-development
description: >-
  Use this skill when building, training, debugging, or modifying the ML models
  in MCLD-1 — including the Stage 2 Temporal-JEPA encoder and the Stage 3 SVGP
  latent flow field.
---

# MCLD-1 ML Development Skill (Stages 2 & 3)

## Scope

Covers the two core ML components:
- **Stage 2:** Temporal-JEPA self-supervised representation learning
- **Stage 3:** Sparse Variational GP (SVGP) latent dynamics and uncertainty engine

---

## Stage 2: Temporal-JEPA

### Architecture Specification

| Component | Spec |
|-----------|------|
| Context Encoder (E_θ) | 3-layer causal TCN, dilation base 2, kernel 3, hidden 64, LayerNorm, GELU |
| Target Encoder (E_ϕ) | Clone of E_θ; EMA update: `ϕ = 0.996·ϕ + 0.004·θ` |
| Predictor (P_ψ) | 2-layer residual MLP: `R^6 → R^6` |
| Context Window | 36 months |
| Target Horizon | 6 months |
| Latent Dimension | 6 |
| Parameter Budget | 10⁴–10⁵ |

### VICReg Loss

```python
# Hyperparameters — DO NOT change without plan amendment
LAMBDA_INV = 25.0
LAMBDA_VAR = 25.0
LAMBDA_COV = 1.0

# L_inv: invariance (smooth L1 between predicted and target embeddings)
# L_var: variance hinge (prevent dimension collapse)
# L_cov: covariance decorrelation (prevent redundant dimensions)
```

### Training Rules

1. **Shape annotations.** Every tensor operation must have a shape comment:
   ```python
   x_context = panel[:, t-35:t+1, :]  # (B, 36, D)
   s_x = context_encoder(x_context)     # (B, 6)
   ```
2. **EMA update** must happen AFTER the optimiser step, not before.
3. **No gradient through target encoder.** Always `torch.no_grad()` for E_ϕ.
4. **Batch size.** Use 32–64. VICReg does NOT require large batches (unlike contrastive methods).
5. **Data augmentation.** Use random temporal patch masking and feature sub-sampling.

### Validation Gate (Must Pass Before Stage 3)

| Metric | Target | How to Compute |
|--------|--------|----------------|
| Stable Rank | ≥ 4.20 | `‖Z‖²_F / ‖Z‖²_2` via SVD on test embeddings |
| Covariance Penalty | < 1e-3 | Sum of squared off-diagonal covariance entries |
| Temporal Roughness | ≤ 0.20 | `E[‖z_{t+1} − z_t‖²] / Var(z_t)` |
| Linear Probe R² | ≥ 0.75 | Frozen-backbone ridge regression → Debt/GDP, CPI |

### Stage 2 Fallbacks

**Dimensional collapse (srank < 4.20):**
- L1: TF-JEPA + DINO centering (dual time/spectral view)
- L2: Causal PatchMAE (30% masked reconstruction)

**High roughness / low R² (R_z > 0.20 or R² < 0.75):**
- L1: Temporal Neighbourhood Contrastive (TNC) with InfoNCE
- L2: Variational DFM (deep state-space VAE with linear-Gaussian prior)

---

## Stage 3: SVGP Latent Flow Field

### Specification

| Parameter | Value |
|-----------|-------|
| Input | `z_t ∈ R^6`, `Δz_t`, `z̄_t` (global centroid) |
| Target | `z_{t+6} − z_t ∈ R^6` (forward latent shift) |
| Model | Sparse Variational GP, M=500 inducing points |
| Kernel | `k_Matérn5/2 · k_Linear + k_SpectralMixture` |
| Output | `μ(z*) ∈ R^6` (drift) + `σ²(z*) ∈ R^6` (uncertainty) |

### Implementation Rules

1. **Use GPyTorch or GPflow** — do NOT write custom GP inference from scratch.
2. **Inducing point initialisation:** Use k-means++ on the training latent vectors.
3. **Multi-output GP:** Use an Independent Multi-Output GP or Linear Model of Coregionalisation (LMC) for the 6D output.
4. **Kernel composition rationale:**
   - Matérn-5/2: non-infinitely-differentiable trajectories (realistic macro shocks)
   - Linear: secular trend capture
   - Spectral Mixture: multi-decade periodic components
5. **Never predict raw market prices.** Stage 3 operates entirely in latent space (Formulation A).

### Validation Gate (Must Pass Before Stage 4)

| Metric | Target |
|--------|--------|
| NLPD | ≤ −1.85 nats/sample |
| PICP₉₅ | 93.0% – 97.0% |
| OOD Variance Ratio | ≥ 3.50 |
| 12-Month Rollout DTW | < 0.15 |

### Stage 3 Fallbacks

**ELBO / coverage miss:**
- L1: Hilbert-Space GP (HSGP) — Laplacian eigenfunction expansion
- L2: Latent Neural SDE with Deep Ensembles (E=10)

**Rollout divergence:**
- L1: Two-Layer Deep GP
- L2: Kim Markov-Switching SSM (K=4 regimes)

---

## Temporal Split (Enforced Globally)

| Partition | Period | Usage |
|-----------|--------|-------|
| Train | 1980–2005 | Model fitting |
| Validation | 2005–2015 | Hyperparameter selection, gate checks |
| Test | 2015–2025 | Final acceptance (NEVER touch during development) |

**The test set (2015–2025) must NEVER be used for model selection, hyperparameter tuning, or architecture decisions.**

## Code Location

- Stage 2: `src/models/jepa/`
- Stage 3: `src/models/gp/`
- Shared utilities: `src/utils/`
- Configs: `configs/jepa.yaml`, `configs/gp.yaml`
