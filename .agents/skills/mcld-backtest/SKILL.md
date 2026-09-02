---
name: mcld-backtest
description: >-
  Use this skill when building, debugging, or running the Stage 4 risk budgeting
  and Stage 5 friction-aware backtest — including portfolio construction, market
  impact modelling, execution simulation, and final system acceptance validation.
---

# MCLD-1 Backtest & Portfolio Skill (Stages 4 & 5)

## Scope

Covers:
- **Stage 4:** Signal construction, uncertainty-damped sizing, SHAP attribution
- **Stage 5:** Friction-aware backtesting, market impact, execution simulation, final acceptance

---

## Stage 4: Risk Budgeting & Signal Construction

### Signal Formula

```
α_i(t) = ⟨μ(z_{i,t}), v_ascension⟩
```

Where `v_ascension` is the principal eigenvector of positive structural growth from the latent space.

### Portfolio Weight Formula

```
w_i(t) = clip( [α_i(t) − ᾱ(t)] / σ_i^asset(t), −w_max, w_max )
         × exp( −γ · ‖σ²(z_{i,t})‖₂ / σ̄_global(t) )
```

- `γ = 2.0` (uncertainty dampening factor)
- The exponential damper automatically scales down allocation when the GPR flags uncharted macro territory.

### Rules

1. **Portfolio must be dollar-neutral.** `Σ w_i = 0` at every rebalance.
2. **Volatility targeting.** Scale each leg by inverse rolling asset volatility.
3. **Interpretability is mandatory.** Every signal must have TreeSHAP attribution mapping back to raw micro-metrics.

### Validation Gate

| Metric | Target |
|--------|--------|
| Crisis Lead-Time | σ² ≥ 2σ_baseline at least 90 days prior to known events |
| Rank IC | ≥ 0.05 (t-stat > 3.0) |
| MDD Reduction (damped vs. un-damped) | ≥ 30% |

---

## Stage 5: Friction-Aware Backtest

### Market Impact Model (Almgren-Chriss)

```
Cost(Q_i) = S_i·Q_i/2 + γ·σ_i·(Q_i/V_i)^0.5·Q_i + β·σ_i·√(Q_i/V_i)
```

| Parameter | DM Value | EM Value |
|-----------|----------|----------|
| γ_impact | 0.40 | 0.90 |
| Max % ADV/day | 1.0% | 0.2% |
| Spread | 0.2–0.8 bps | 3–15 bps |

### Execution Rules

1. **No instantaneous rebalancing.** Use partial adjustment: `w_t = w_{t-1} + 0.33·(w*_t − w_{t-1})` (3-day TWAP).
2. **Dynamic slippage matrix.** Pass time-varying `Cost_{N×T}` into VectorBT, calculated from rolling ADV and volatility.
3. **Include carry costs.** FX roll (interest rate differential + cross-currency basis) and futures calendar roll slippage must be deducted.

### Cross-Validation

Use **Combinatorial Purged CV (CPCV)** via Riskfolio-Lib:
- 6-month purge window
- 3-month embargo window
- This prevents auto-regressive label leakage that standard K-fold introduces.

### Backtesting Engines

| Engine | Role |
|--------|------|
| **VectorBT** | Primary: vectorised matrix-level backtesting with dynamic slippage |
| **NautilusTrader** | Secondary: event-driven L1/L2 book simulation for fill-rate validation |
| **Qlib** | Benchmark: IC/RankIC evaluation against quant baselines |

### Validation Gate (Final System Acceptance)

| Metric | Target |
|--------|--------|
| Out-of-Sample Sharpe | ≥ 1.35 (2015–2025, net of all costs) |
| Maximum Drawdown | ≤ 12.0% under 10% vol target |
| CVaR₉₅ | < 1.8 × VaR₉₅ |
| Friction Capacity Retention | Sharpe_net($100M) / Sharpe_gross ≥ 0.75 |

### Stage 5 Fallbacks

**Capacity / friction drag:**
- L1: Reduce κ → 0.15 (5-day VWAP) + L1 turnover penalty
- L2: Prune to Top-30 liquid sovereign pairs

**Sharpe / tail risk miss:**
- L1: Regime leverage scaling (leverage ∝ 1/σ̄_global)
- L2: QP-enforced dollar-neutral + regional zero-beta

## Code Location

- Stage 4: `src/portfolio/`
- Stage 5: `src/backtest/`
- Configs: `configs/portfolio.yaml`, `configs/backtest.yaml`
