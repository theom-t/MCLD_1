---
name: mcld-stage-gate
description: >-
  Use this skill when evaluating whether a pipeline stage has passed its
  validation gate, when a validation threshold has been breached and a fallback
  must be activated, or when deciding whether to escalate from Level-1 to
  Level-2 contingency for any MCLD-1 stage.
---

# MCLD-1 Stage Gate Evaluation Skill

## Purpose

This skill defines the decision process for evaluating validation gates and activating contingency fallbacks. Every stage in the MCLD-1 pipeline has a strict pass/fail gate. **No stage may proceed to the next until ALL gate metrics are met.**

## Gate Evaluation Process

```
1. Run all validation metrics for the current stage
2. IF all metrics pass → proceed to next stage
3. IF any metric fails:
   a. Diagnose which failure category the breach falls into
   b. Activate the Level-1 (intra-paradigm) pivot for that failure category
   c. Re-run Optuna hyperparameter search (TPE, budget = 200 trials)
   d. Re-evaluate all validation metrics
   e. IF Level-1 passes → proceed to next stage
   f. IF Level-1 fails → permanently pivot to Level-2 (structural fallback)
   g. Re-evaluate all validation metrics with Level-2
   h. IF Level-2 passes → proceed to next stage
   i. IF Level-2 fails → HALT and escalate to human review
```

## Complete Gate Reference

### Stage 1: Data Ingestion

| Metric | Target | Failure Category |
|--------|--------|------------------|
| ADF Stationarity | p < 0.01 on ≥ 98% | Stationarity |
| Memory Retention | corr ≥ 0.70 | Stationarity |
| Imputation NRMSE | ≤ 0.08 | Imputation |
| Causal Integrity | 0.0% leakage | **HARD ABORT** (no fallback — rebuild pipeline) |

**Fallbacks:**
- Stationarity: L1 → Wavelet MODWT | L2 → C-F Bandpass
- Imputation: L1 → GP State-Space | L2 → EM Dynamic Factor

### Stage 2: Temporal-JEPA

| Metric | Target | Failure Category |
|--------|--------|------------------|
| Stable Rank | ≥ 4.20 | Dimensional Collapse |
| Covariance Penalty | < 1e-3 | Dimensional Collapse |
| Temporal Roughness | ≤ 0.20 | Roughness / Decodability |
| Linear Probe R² | ≥ 0.75 | Roughness / Decodability |

**Fallbacks:**
- Collapse: L1 → TF-JEPA + DINO | L2 → Causal PatchMAE
- Roughness: L1 → TNC (InfoNCE) | L2 → Variational DFM

### Stage 3: SVGP Flow Field

| Metric | Target | Failure Category |
|--------|--------|------------------|
| NLPD | ≤ −1.85 nats/sample | ELBO / Coverage |
| PICP₉₅ | 93.0% – 97.0% | ELBO / Coverage |
| OOD Variance Ratio | ≥ 3.50 | ELBO / Coverage |
| 12-Month DTW | < 0.15 | Rollout Divergence |

**Fallbacks:**
- ELBO: L1 → HSGP | L2 → Latent Neural SDE
- Rollout: L1 → Two-Layer Deep GP | L2 → Kim Markov-Switching

### Stage 4: Risk Budgeting

| Metric | Target | Failure Category |
|--------|--------|------------------|
| Crisis Lead-Time | ≥ 90 days | Crisis Detection |
| Rank IC | ≥ 0.05 (t > 3.0) | Signal Quality |
| MDD Reduction | ≥ 30% | Signal Quality |

**Fallbacks:**
- Signal: L1 → HRP Quantile Gating | L2 → Thompson Bandit
- Crisis: L1 → Latent Acceleration Penalty | L2 → CDS/Skew Overlay

### Stage 5: Friction-Aware Backtest

| Metric | Target | Failure Category |
|--------|--------|------------------|
| OOS Sharpe | ≥ 1.35 | Sharpe / Tail Risk |
| MDD | ≤ 12.0% | Sharpe / Tail Risk |
| CVaR₉₅ | < 1.8 × VaR₉₅ | Sharpe / Tail Risk |
| Capacity Retention | ≥ 0.75 | Capacity / Friction |

**Fallbacks:**
- Sharpe: L1 → Regime Leverage Scaling | L2 → Static Beta-Neutral
- Capacity: L1 → Turnover Damping | L2 → Liquid Core Universe (Top-30)

## Reporting

When evaluating a gate, produce a structured report:

```markdown
## Stage [N] Gate Evaluation — [Date]

### Results
| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| ...    | ...    | ...    | ✅/❌  |

### Decision
- [ ] ALL PASS → Proceed to Stage [N+1]
- [ ] FAIL → Activate L1 fallback for [failure category]
- [ ] L1 FAIL → Escalate to L2 fallback for [failure category]
- [ ] L2 FAIL → HALT — escalate to human review

### Notes
[Any observations, diagnostic details, or recommendations]
```

Save gate evaluation reports to `docs/gate_reports/stage_N_eval_YYYYMMDD.md`.
