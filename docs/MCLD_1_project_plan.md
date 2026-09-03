# MCLD-1: Macro-Cycle Latent Dynamics Model — Project Plan

> **Version:** 1.0  
> **Date:** 2026-09-02  
> **Status:** Pre-Development  

---

## 1. Executive Summary

MCLD-1 is a self-supervised quantitative framework that maps monthly macroeconomic indicators across ~50 sovereign nations into a continuous low-dimensional latent manifold, models the autonomous dynamics of empire-scale economic cycles, and quantifies structural regime uncertainty for risk-budgeted global macro execution.

The system bridges Ray Dalio's centuries-long "Big Cycle" theory to actionable monthly signals by treating macro-regime evolution as a continuous-time dynamical system — learned without ground-truth labels, forecast with Bayesian uncertainty, and monetised through cross-sectional sovereign asset allocation.

### Core Thesis

High-frequency monthly micro-metrics (debt dynamics, institutional friction, geopolitical flows) compound into the slow-moving secular cycles Dalio describes. A Temporal-JEPA learns these latent dynamics self-supervised; a Sparse Variational GP models their forward trajectory and flags structural uncertainty; a risk-budgeting layer translates this into dollar-neutral, friction-aware sovereign portfolios.

### Research Questions

| ID | Research Question |
|----|-------------------|
| **RQ1** | Can a Temporal-JEPA learn continuous, economically decodable representations of secular macro cycles from noisy, asynchronous monthly panels — without ground-truth regime labels? |
| **RQ2** | Can a continuous-time SVGP vector field on those representations accurately predict multi-step macro trajectory drift and yield calibrated epistemic uncertainty that serves as an early-warning signal for sovereign regime bifurcations? |
| **RQ3** | Does conditioning cross-sectional sovereign allocation on latent cycle velocity, penalised by GPR epistemic uncertainty, generate persistent risk-adjusted excess returns after non-linear market impact and carry drag? |

### Key Insights from Prior Art

1. **Predict latent vectors, not raw data.** Generative reconstruction wastes capacity on high-frequency noise. JEPA's latent-space prediction bypasses this.
2. **Avoid isotropic priors.** Standard JEPA regularisation forces flat Euclidean geometry; macro cycles are non-linear dynamical systems. Non-Euclidean / Hamiltonian constraints on the predictor are preferred.
3. **Additive GP kernels scale.** Multiplicative kernels blow up at O(M^D); additive kernels linearise to O(D × M̃), enabling 50-country panels.
---

## 2. End-to-End Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               END-TO-END PIPELINE TOPOLOGY                              │
│                                                                                         │
│ [Stage 1: Bitemporal Data Ingestion] ──► [Stage 2: Self-Supervised Temporal-JEPA]       │
│                                                          │                              │
│                                                          ▼                              │
│ [Stage 5: Friction-Aware Backtest]  ◄── [Stage 4: Risk Budget] ◄── [Stage 3: SVGP]     │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Stage Definitions

---

### Stage 1 — Bitemporal Data Ingestion, Stationarity & Panel Synchronisation

#### 1.1 Objective

Construct a causally correct, stationary, gap-filled multi-country monthly panel tensor ready for self-supervised representation learning.

#### 1.2 Specification

| Parameter | Value |
|-----------|-------|
| Panel Tensor | `X_raw ∈ R^{T × N × D}` |
| Time (T) | 540 months (1980–2025) |
| Countries (N) | 50 sovereign economies |
| Features (D) | 48 (16 per domain) |

**Feature Domains (16 features each):**

- **Monetary & Debt:** Central bank balance sheet Δ, sovereign debt/GDP, yield curve slope (10Y–2Y), ACM term premia, FX reserve velocity (COFER), M2 expansion rate, net treasury issuance absorption, repo rate differentials, real yield levels, credit-to-GDP gap, public debt service ratio, primary surplus/deficit, interbank spread, foreign-held debt share, currency swap line utilisation, monetary base growth rate.
- **Internal Order & Institutional Friction:** Real wage vs. asset price divergence, Economic Policy Uncertainty (EPU), Geopolitical Risk (GPR) index, strike/labour unrest frequency, fiscal deficit acceleration, legislative NLP polarisation vectors, Gini coefficient proxy shifts, municipal fiscal stress spreads, top-decile income share growth, consumer confidence divergence, judicial independence scores, political turnover rate, media polarisation indices, social unrest event frequency, urban migration velocity, housing affordability ratio.
- **External Order & Geopolitics:** Non-dollar trade settlement ratio (SWIFT vs. CIPS), bilateral trade concentration (HHI), net FDI flow velocity, defence budget/GDP delta, critical supply chain concentration, tariff rate changes, rare-earth/semiconductor export licensing frequency, capital flight velocity, cross-border banking flow (BIS), bilateral treaty/alliance count delta, sanctions exposure index, technology patent balance, immigration policy tightness, military spending acceleration, bilateral FX agreement count, foreign aid dependency ratio.

**Data Sources:**

| Category | Sources | Depth |
|----------|---------|-------|
| Long-Horizon Macrohistory | Jordà-Schularick-Taylor (JST), Maddison Project | 1870–present |
| Sovereign Debt & FX Reserves | IMF IFS, World Bank WDI, BIS Debt Securities | 1960–present |
| Central Bank & Money Supply | FRED, BIS Central Bank Balance Sheets | 1970–present |
| Trade & Supply Chain | UN Comtrade, IMF DOTS | 1962–present |
| Financial Conditions | Refinitiv / Bloomberg / FRED / Global Financial Data | 1980–present |

**Point-in-Time Alignment:**
- Storage in **ArcticDB** with versioned, immutable bitemporal snapshots.
- Ingestion via **Polars** `join_asof(strategy="backward")` — strictly matching publication release timestamps `t_pub ≤ t_trade`.

**Stationarity Transformation — ~~Fractional Differentiation~~ Reversible Instance Normalization (RevIN):**

~~Apply component-wise: `(1 - B)^d x_t`, choosing optimal `d_i*` as the minimum `d ∈ [0.2, 0.7]` satisfying ADF test at `p < 0.01`. This preserves multi-decade memory while achieving stationarity.~~
*Pivot 2026-09-03:* Fractional differencing destroyed structural memory. We will pass raw PCHIP-interpolated data directly into Stage 2, and use **RevIN** natively inside the JAX/Flax Temporal-JEPA to dynamically normalize the 36-month windows on the fly.

**Ragged-Edge Handling:**
- Quarterly/annual series mapped to monthly via PCHIP (Piecewise Cubic Hermite Interpolating Polynomials).
- Accompanied by a binary mask tensor `M ∈ {0,1}^{T × N × D}` indicating observed vs. interpolated values.

#### 1.3 Validation Gate [COMPLETED: 2026-09-03]

| Metric | Target | Method |
|--------|--------|--------|
| ~~ADF Stationarity~~ / Internal Stationarity | Mean = 0.0, Var = 1.0 | JAX/Flax RevIN rolling window normalization |
| ~~Memory Retention~~ / Signal Preservation | `corr(raw, norm) = 1.0` | Pearson cross-correlation of RevIN windows |
| Imputation Fidelity | `NRMSE ≤ 0.08` on synthetic 20% MCAR drop | Normalised Root Mean Square Error |
| Causal Integrity | 0.0% future leakage | Automated timestamp delta assertion audit |

#### 1.4 Contingency Protocol

```
                       [Stage 1 Validation Gate]
                                   │
                ┌─────────────────┴──────────────────┐
                ▼ Pass                                ▼ Fail
         [Proceed to Stage 2]          ┌──────────────┴──────────────┐
                                       ▼                             ▼
                              [Stationarity Fails]          [Imputation Fails]
                                       │                             │
                                       ▼                             ▼
                              L1: Wavelet MODWT             L1: GP State-Space
                              (J=6 scales, discard A6,      (Matérn-3/2 kernel,
                               retain D1…D5)                 per feature per country)
                                       │                             │
                                       ▼ still failing               ▼ still failing
                              L2: Christiano-Fitzgerald     L2: EM Dynamic Factor
                              Bandpass (18–96 month band)    (Kalman-smoothed DFM)
```

**Detail:**
- **Stationarity L1 — Wavelet MODWT:** Decompose into J=6 multiresolution levels; retain stationary detail bands D1–D5, difference smooth approximation A6. Re-run ADF.
- **Stationarity L2 — Christiano-Fitzgerald Bandpass:** Asymmetric real-time bandpass isolating 18–96 month cycle frequencies, eliminating unit-root components without memory loss.
- **Imputation L1 — GP State-Space Interpolator:** Fit independent 1D continuous-time GPs with Matérn-3/2 kernels per feature to interpolate asynchronous reporting dates.
- **Imputation L2 — EM Dynamic Factor Imputer:** Cross-sectional DFM via EM to infer missing values through Kalman smoothing across all 50 countries simultaneously.

---

### Stage 2 — Self-Supervised Latent Representation Engine (Temporal-JEPA)

#### 2.1 Objective

Learn a continuous, low-dimensional, temporally smooth representation of each country's macroeconomic state — entirely self-supervised, with zero ground-truth regime labels.

#### 2.2 Specification

| Parameter | Value |
|-----------|-------|
| Context Window (L_x) | 36 months |
| Target Horizon (L_y) | 6 months (t+1 to t+6) |
| Latent Dimension (d) | 6 |
| Training Sequences | ~25,000 country-months (50 × 45yr × 12mo, rolling stride-1) |
| Parameter Budget | 10⁴–10⁵ parameters (compact inductive bias) |

**Architecture:**

| Component | Specification |
|-----------|---------------|
| **Context Encoder (E_θ)** | 3-layer causal TCN, dilation base 2, kernel size 3, hidden dim 64, LayerNorm, GELU |
| **Target Encoder (E_ϕ)** | Structural clone of E_θ; weights via EMA: `ϕ_t = 0.996·ϕ_{t-1} + 0.004·θ_t` |
| **Predictor (P_ψ)** | 2-layer residual MLP: `s_x ∈ R^6 → ŝ_y ∈ R^6` |

```
  Context x_{t-35:t} ──► [Context Encoder E_θ] ──► s_x ──► [Predictor P_ψ] ──► ŝ_y
                                                                                  │
                                                                        (Smooth L1 Loss)
                                                                                  │
  Target  x_{t+1:t+6} ──► [Target Encoder E_ϕ (EMA)]  ──────────► s_y ──────────┘
```

**VICReg Objective:**

```
L_JEPA = λ_inv · L_inv(s_y, ŝ_y) + λ_var · [L_var(s_x) + L_var(s_y)] + λ_cov · [L_cov(s_x) + L_cov(s_y)]
```

- **Invariance:** `L_inv = (1/B) Σ ‖ŝ_{y,i} − s_{y,i}‖²`
- **Variance (hinge):** `L_var(Z) = (1/d) Σ max(0, 1 − √(Var(z_j) + ε))`
- **Covariance (decorrelation):** `L_cov(Z) = (1/d) Σ_{j≠k} C(Z)²_{j,k}`
- **Hyperparameters:** `λ_inv = 25.0, λ_var = 25.0, λ_cov = 1.0`

**Why JEPA works on macro data:**
- Cross-sectional parameter sharing across 50 countries provides spatial augmentation.
- Compact backbone (2–3 layer TCN, d_model ∈ [32,64]) avoids overfitting.
- VICReg / non-contrastive regularisation avoids massive batch-size requirements.

#### 2.3 Validation Gate

| Metric | Target | Method |
|--------|--------|--------|
| Stable Rank `srank(Z)` | ≥ 4.20 (70% of d=6) | SVD of embedding batch: `‖Z‖²_F / ‖Z‖²_2` |
| Covariance Penalty `C(Z)` | < 1.0 × 10⁻³ | Sum of squared off-diagonal entries in sample covariance |
| Temporal Roughness `R_z` | `E[‖z_{t+1} − z_t‖²] / Var(z_t) ≤ 0.20` | Normalised mean squared first-difference |
| Linear Probe R² | ≥ 0.75 on held-out test | Frozen-backbone ridge regression → Debt/GDP, YoY CPI |

#### 2.4 Contingency Protocol

```
                       [Stage 2 Validation Gate]
                                   │
                ┌─────────────────┴──────────────────┐
                ▼ Pass                                ▼ Fail
         [Proceed to Stage 3]          ┌──────────────┴──────────────┐
                                       ▼                             ▼
                              [Dimensional Collapse]      [High Roughness / Low R²]
                              (srank < 4.20)               (R_z > 0.20 or R² < 0.75)
                                       │                             │
                                       ▼                             ▼
                              L1: TF-JEPA + DINO           L1: Temporal Neighbourhood
                              (Dual time/spectral view,     Contrastive (TNC)
                               Sinkhorn-Knopp centering)    (InfoNCE on stationary
                                                             ε-windows)
                                       │                             │
                                       ▼ still failing               ▼ still failing
                              L2: Causal PatchMAE           L2: Variational DFM (V-DFM)
                              (30% masked temporal patch     (Deep state-space VAE,
                               reconstruction — collapse     explicit linear-Gaussian
                               impossible by construction)   transition prior)
```

**Detail:**
- **Collapse L1 — TF-JEPA + DINO:** Branch encoder into parallel time-domain and Fourier spectral pipelines; enforce cross-domain invariance using Sinkhorn-Knopp centering and sharpening instead of VICReg.
- **Collapse L2 — Causal PatchMAE:** Replace joint-embedding with causal Masked Autoencoder reconstructing 30% randomly masked temporal patches; reconstruction loss mathematically prohibits dimensional collapse.
- **Roughness L1 — Temporal Neighbourhood Contrastive (TNC):** InfoNCE contrastive loss treating temporally adjacent stationary ε-neighbourhoods as positive pairs while contrasting non-stationary segments.
- **Roughness L2 — Variational DFM (V-DFM):** Deep state-space VAE with explicit linear-Gaussian transition prior `p(z_t | z_{t-1}) = N(Az_{t-1}, Q)` enforcing smooth dynamics analytically.

---

### Stage 3 — SVGP Autonomous Macro Flow Field & Uncertainty Engine

#### 3.1 Objective

Model the continuous-time transition dynamics of the latent macro-cycle as an autonomous vector field, producing both forward trajectory predictions and calibrated epistemic uncertainty that detects structural novelty.

#### 3.2 Specification

**Formulation A (Self-Supervised Latent Vector Field):**

The GPR learns the motion law: `Δz_{i,t→t+h} = z_{i,t+h} − z_{i,t} = f(z_{i,t}) + ε`

| Parameter | Value |
|-----------|-------|
| Input | `z_{i,t} ∈ R^6` (JEPA latent), `Δz_{i,t}` (current velocity), `z̄_t` (global centroid) |
| Target | Forward latent shift: `z_{i,t+6} − z_{i,t} ∈ R^6` |
| Method | Sparse Variational GP (SVGP), M=500 inducing points |
| Kernel | `k_Matérn5/2 · k_Linear + k_SpectralMixture` |

```
[JEPA State z_t ("You are here")]
             │
             ▼
    [GPR Motion Model]
             │
     ┌───────┴────────────────────────┐
     ▼                                ▼
[Direction & Speed]           [Confidence Score]
 μ(z*): Expected Drift         σ²(z*): Epistemic
 "Heading toward Stage 5"      Uncertainty
```

**Why Formulation A over Formulation B (direct market prediction):**
- Stays entirely within the noise-free latent manifold — never sees noisy market prices.
- Correctly tracks structural deterioration even when markets are decoupled (e.g., QE-inflated equities despite fiscal decay).
- Far more academically novel: treats empire cycles as an autonomous continuous-time dynamical system.
- Formulation B is layered downstream as a lightweight execution policy (Stage 4).

**SVGP Details:**
- **ELBO:** `ELBO = Σ E_q(f_i)[log p(y_i|f_i)] − KL(q(u) ‖ p(u))`
- **Kernel rationale:** Matérn-5/2 allows non-infinitely-differentiable trajectories; Linear captures secular trends; Spectral Mixture models multi-decade periodicities.
- **Outputs:** Expected drift vector `μ(z*) ∈ R^6` + structural uncertainty `σ²(z*) ∈ R^6`.

#### 3.3 Validation Gate

| Metric | Target | Method |
|--------|--------|--------|
| Negative Log Predictive Density | `NLPD ≤ −1.85` nats/sample | Out-of-sample test partition (2015–2025) |
| Coverage Calibration (PICP₉₅) | 93.0% ≤ PICP₉₅ ≤ 97.0% | Empirical coverage within `[μ ± 1.96σ]` |
| OOD Variance Ratio | `E_OOD[σ²] / E_train[σ²] ≥ 3.50` | Epistemic variance on extreme synthetic shocks |
| 12-Month Rollout Divergence | `DTW < 0.15` | Autonomous ODE integration `ẑ_{t+1} = ẑ_t + μ(ẑ_t)` vs. actuals |

#### 3.4 Contingency Protocol

```
                       [Stage 3 Validation Gate]
                                   │
                ┌─────────────────┴──────────────────┐
                ▼ Pass                                ▼ Fail
         [Proceed to Stage 4]          ┌──────────────┴──────────────┐
                                       ▼                             ▼
                              [ELBO / Coverage Miss]        [Rollout Divergence]
                                       │                             │
                                       ▼                             ▼
                              L1: Hilbert-Space GP (HSGP)   L1: Two-Layer Deep GP
                              (Laplacian eigenfunction       (Hierarchical GP:
                               expansion → exact BLR)         f2(f1(z_t)))
                                       │                             │
                                       ▼ still failing               ▼ still failing
                              L2: Latent Neural SDE          L2: Kim Markov-Switching
                              (dz = f_θ(z)dt + g_ϕ(z)dW,    (K=4 regime matrices,
                               Deep Ensembles E=10)           Hamilton transitions)
```

**Detail:**
- **ELBO L1 — Hilbert-Space GP (HSGP):** Approximate GP using low-rank Laplacian eigenfunction expansions on bounded domain Ω ⊂ R^6, converting non-parametric inference into stable Bayesian linear regression.
- **ELBO L2 — Latent Neural SDE:** Model dynamics as continuous SDEs `dz = f_θ(z)dt + g_ϕ(z)dW`; extract epistemic uncertainty via Deep Ensembles (E=10) with randomised priors.
- **Rollout L1 — Two-Layer Deep GP:** Hierarchical mapping `z_{t+h} = f2(f1(z_t))` where layer 1 warps non-linear state space, layer 2 models smooth velocity vectors.
- **Rollout L2 — Kim Markov-Switching SSM:** Discretise latent trajectory into K=4 regime matrices `(A_k, Q_k)` with Hamilton transition probabilities, collapsing mixture states via Kim's algorithm.

---

### Stage 4 — Uncertainty-Damped Risk Budgeting & Dynamic Signal Construction

#### 4.1 Objective

Translate latent cycle velocity and uncertainty into tradable cross-sectional sovereign signals with automatic capital dampening in uncharted macro territory.

#### 4.2 Specification

**Ascension Projection:**
```
α_i(t) = ⟨μ(z_{i,t}), v_ascension⟩
```
Where `v_ascension` is the principal eigenvector corresponding to positive structural growth (Stage 1 → Stage 3).

**Uncertainty-Damped Dynamic Sizing:**
```
w_i(t) = clip( [α_i(t) − ᾱ(t)] / σ_i^asset(t), −w_max, w_max )
         × exp( −γ · ‖σ²(z_{i,t})‖₂ / σ̄_global(t) )
```
Where `γ = 2.0` penalises allocation into unmapped, high-uncertainty latent zones.

```
Latent Velocity μ(z_i) ──► Macro Ascension Score α_i
                                    │
                                    ▼
Raw Asset Volatility   ──► Unscaled Weight w_i ──► [Uncertainty Damper: exp(−γ·σ²)] ──► w_final
```

**Interpretability Layer:**
- TreeSHAP over k-means clusters in z-space → marginal Shapley contribution of each raw micro-metric to the state transition.

**Monetisation Channels:**

| Country Stage | Strategy | Instruments |
|---------------|----------|-------------|
| Early Ascending (1–2) | Long domestic growth beta | Country ETFs, Index Futures |
| Peak / Overextended (3–4) | Tilt large-cap multinationals | Broad benchmark futures |
| Late Decline (5–6) | Short broad index; long exporters, hard assets | Inverse ETFs, CDS proxies |

Plus cross-cutting: FX relative value (long ascending / short declining currencies), sovereign rates spreads, and real-asset / debasement hedges.

#### 4.3 Validation Gate

| Metric | Target | Method |
|--------|--------|--------|
| Crisis Lead-Time | `σ²(z_{i,t}) ≥ 2σ_baseline` at least **90 days** prior | Historical audit: 2015 Eurozone, 2020 COVID, 2022 UK Gilt |
| Signal Monotonicity | Rank IC ≥ 0.05 (t-stat > 3.0) | Spearman corr(α_i(t), forward 3M relative return) |
| Uncertainty Drawdown Mitigation | MDD reduced ≥ 30% vs. un-damped baseline | Ablation: γ=0 vs. γ=2.0 |

#### 4.4 Contingency Protocol

```
                       [Stage 4 Validation Gate]
                                   │
                ┌─────────────────┴──────────────────┐
                ▼ Pass                                ▼ Fail
         [Proceed to Stage 5]          ┌──────────────┴──────────────┐
                                       ▼                             ▼
                              [Signal Drag / Low IC]        [Crisis Lead-Time Miss]
                                       │                             │
                                       ▼                             ▼
                              L1: HRP Quantile Gating       L1: Latent Acceleration
                              (Trade Q1 vs Q5 only;          Penalty
                               HRP weights on latent         (exp(−γ₁‖σ²‖ − γ₂‖d²z/dt²‖))
                               distance matrix)
                                       │                             │
                                       ▼ still failing               ▼ still failing
                              L2: Contextual Thompson       L2: Asymmetric CDS/Skew
                              Sampling Bandit                Overlay (1.5% annual risk
                              (MAB with latent-conditional   budget → 25Δ puts on
                               reward posteriors)             late-cycle currencies)
```

**Detail:**
- **Signal L1 — HRP Quantile Gating:** Restrict execution to top-decile (Q1 Long) vs. bottom-decile (Q5 Short) sovereign baskets; allocate internal weights via HRP clustering on latent distance matrix.
- **Signal L2 — Contextual Thompson Sampling Bandit:** Bayesian Thompson Sampling across asset buckets, updating reward posteriors conditioned on latent coordinate and uncertainty.
- **Crisis L1 — Latent Acceleration Penalty:** Augment uncertainty damper with second-derivative term: `Penalty = exp(−γ₁‖σ²‖ − γ₂‖d²z/dt²‖)`.
- **Crisis L2 — Asymmetric Tail Overlay:** Systematically route 1.5% annual risk budget into OTM 25Δ puts on late-cycle sovereign currencies when latent velocity points toward Stage 5/6.

---

### Stage 5 — Friction-Aware Backtest, Execution Simulation & Final Acceptance

#### 5.1 Objective

Validate the complete pipeline under realistic market friction, proving the strategy survives transaction costs, market impact, and derivative carry drag at institutional scale.

#### 5.2 Specification

**Cross-Validation:** Combinatorial Purged CV (CPCV) via Riskfolio-Lib with 6-month purge window and 3-month embargo.

**Non-Linear Market Impact (Almgren-Chriss Square-Root):**
```
Cost_impact(Q_i) = S_i · Q_i/2                              (half-spread)
                 + γ · σ_{i,t} · (Q_i/V_{i,t})^0.5 · Q_i   (temporary impact)
                 + β · σ_{i,t} · √(Q_i/V_{i,t})             (permanent impact)
```
Where `Q_i = |Δw_i| · AUM`, `V_{i,t} = ADV_20d`, `γ = 0.40` (DMs) / `0.90` (EMs).

| Parameter | Developed Markets | Emerging Markets |
|-----------|-------------------|------------------|
| Max Order Capacity | ≤ 1.0% ADV/day | ≤ 0.2% ADV/day |
| Base Bid-Ask Spread | 0.2–0.8 bps | 3.0–15.0 bps |
| Volatility Multiplier | Baseline | Spikes 3× in crises |

**Derivatives Carry & Roll Drag:**
```
Cost_FX_Roll = (r_counter − r_base + Basis_ccy) · Δt/360 + Spread_fwd
Cost_Futures_Roll = (P_back − P_front) + Spread_calendar
```

**Gradual Execution — Partial Adjustment:**
```
w_t = w_{t-1} + κ · (w*_t − w_{t-1}),  κ = 0.33 (3-day TWAP)
```

**Integrated Objective with Impact Penalty:**
```
max_w [ w'μ − (λ/2)·w'Σw − Σ c_i|w_i − w_{i,prev}| − Σ γ·σ_i/(V_i^0.5) · AUM^1.5 · |Δw_i|^1.5 ]
```

**Backtesting Engines:**
- **VectorBT:** Matrix-level parameter sweeps with dynamic slippage matrices (`vbt.Portfolio.from_orders(..., slippage=dynamic_slippage_matrix)`).
- **NautilusTrader:** Event-driven execution simulation against historical L1/L2 book data for tick-level fill-rate validation.
- **Qlib (Microsoft Research):** IC/Rank IC evaluation and benchmark comparison against standard quant baselines.

#### 5.3 Validation Gate (Final System Acceptance)

| Metric | Target | Method |
|--------|--------|--------|
| Out-of-Sample Sharpe | `Sharpe_net ≥ 1.35` (2015–2025) | VectorBT net of commissions, impact, carry |
| Maximum Drawdown | `MDD ≤ 12.0%` under 10% vol target | Peak-to-trough net equity curve |
| Tail Risk Ratio | `CVaR₉₅ < 1.8 × VaR₉₅` | Conditional VaR under historical simulation |
| Friction Capacity Retention | `Sharpe_net($100M) / Sharpe_gross ≥ 0.75` | Net Sharpe under full Almgren-Chriss scaling |

#### 5.4 Contingency Protocol

```
                       [Stage 5 Validation Gate]
                                   │
                ┌─────────────────┴──────────────────┐
                ▼ Pass                                ▼ Fail
         [System Acceptance]          ┌──────────────┴──────────────┐
                                       ▼                             ▼
                              [Capacity / Friction Drag]    [Sharpe / Tail Risk Miss]
                                       │                             │
                                       ▼                             ▼
                              L1: Turnover Damping          L1: Regime Leverage Scaling
                              (κ → 0.15 = 5-day VWAP;       (Gross leverage ∝
                               + L1 turnover penalty         1/σ̄_global(t))
                               in optimiser)
                                       │                             │
                                       ▼ still failing               ▼ still failing
                              L2: Liquid Core Universe      L2: Static Beta-Neutralisation
                              (Prune to Top-30 most          (QP-enforced dollar-neutral
                               liquid sovereign pairs)        + regional zero-beta)
```

**Detail:**
- **Capacity L1 — Turnover Damping:** Reduce rebalancing speed `κ → 0.15` (5-day VWAP window); add L1 turnover penalty to portfolio optimiser.
- **Capacity L2 — Liquid Core Universe:** Prune to Top-30 highest-liquidity sovereign currencies and bond futures, eliminating wide-spread EM pairs.
- **Sharpe L1 — Regime Leverage Scaling:** Scale gross leverage inversely to global structural uncertainty: `Leverage(t) = Lev_target · min(1.0, σ_threshold / σ̄(t))`.
- **Sharpe L2 — Static Beta-Neutralisation:** Impose strict dollar-neutral and regional zero-beta constraints via quadratic programming.

---

## 4. Global Fallback Summary Matrix

```
┌──────────────────────────┬───────────────────────────────┬───────────────────────────────┐
│ System Layer             │ Primary Engine                │ Fully Validated Fallback      │
├──────────────────────────┼───────────────────────────────┼───────────────────────────────┤
│ 1. Data Ingestion        │ Fractional Diff (ADF p<0.01)  │ Wavelet MODWT + Bandpass      │
│ 2. Representation        │ Temporal-JEPA (VICReg)        │ Causal PatchMAE / V-DFM       │
│ 3. Latent Dynamics       │ SVGP (Matérn + Spectral)      │ Two-Layer DGP / Neural SDE    │
│ 4. Portfolio Sizing      │ Alpha Projection + Exp Damper │ Contextual Thompson MAB + HRP │
│ 5. Execution Validation  │ VectorBT + Almgren-Chriss     │ Liquid Core + Beta Neutral    │
└──────────────────────────┴───────────────────────────────┴───────────────────────────────┘
```

**Escalation Rule:** If a validation gate is breached, execution halts. The Level-1 (intra-paradigm) pivot activates and the validation suite re-runs. If Level-1 fails after Optuna hyperparameter optimisation (TPE budget of 200 trials), the system pivots permanently to the Level-2 (structural model) fallback before advancing downstream.

---

## 5. Technical Stack

```
┌─────────────────────────────────────────────────────────────┐
│ Core Engine:   JAX / Flax (Autograd, vmap, & XLA Compilation)       │
│ GP Modelling:  GPJax (GPU-Accelerated SVGP)                 │
│ Data Pipe:     Polars + Apache Arrow (Zero-Copy Processing) │
│ Storage:       ArcticDB (Bitemporal Point-in-Time)          │
│ Optimisation:  Optuna (Multi-Objective TPE)                 │
│ Attribution:   SHAP / Captum (Latent Gradient Attribution)  │
│ Backtesting:   VectorBT (Vectorised) + NautilusTrader       │
│ Validation:    Riskfolio-Lib (Purged CV) + Qlib (IC/RankIC) │
│ Visualisation: Plotly / Matplotlib / TDA (Persistent Homology)│
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Temporal Splits (Preventing Lookahead Bias)

| Partition | Period | Purpose |
|-----------|--------|---------|
| Train | 1980–2005 (all 50 countries) | Model fitting |
| Validation | 2005–2015 | Hyperparameter selection, gate checks |
| Out-of-Sample Test | 2015–2025 | Final acceptance metrics |

---

## 7. Production Execution Schedule

| Month | Deliverable |
|-------|-------------|
| Month 1 | Stage 1: ArcticDB + Polars pipeline, fractional differencing, panel QA |
| Month 2 | Stage 2: Temporal-JEPA training, VICReg optimisation, representation health checks |
| Month 3 | Stage 3: SVGP inducing point selection, kernel optimisation, flow field training |
| Month 4 | Stage 4: Uncertainty budgeting integration, SHAP attribution, signal construction |
| Month 5 | Stage 5: VectorBT / NautilusTrader backtest, CPCV engine, friction modelling |
| Month 6 | Final paper deliverable, production readiness review |

---

## 8. Document References

- [Initial Brainstorm Transcript](inital_brainstorm_source.md) — Full conversation record capturing ideation, architecture decisions, and prior art analysis.
