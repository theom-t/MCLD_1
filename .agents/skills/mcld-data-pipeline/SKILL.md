---
name: mcld-data-pipeline
description: >-
  Use this skill when building, debugging, or modifying the Stage 1 data
  ingestion pipeline — including ArcticDB storage, Polars asof joins, fractional
  differencing, PCHIP interpolation, and panel synchronisation for MCLD-1.
---

# MCLD-1 Data Pipeline Skill (Stage 1)

## Scope

Stage 1 constructs the causally correct, stationary, gap-filled multi-country monthly panel tensor `X ∈ R^{T × N × D}` where T=540 months, N=50 countries, D=48 features.

## Critical Constraints

### Zero Future Leakage

This is the single most important rule in the entire pipeline. Every data join MUST be backward-looking:

```python
# CORRECT: backward asof join on release timestamps
aligned = market_df.join_asof(
    macro_df,
    on="release_date",      # NOT observation_date
    by="country_iso",
    strategy="backward"     # MANDATORY
)

# WRONG: joining on observation date introduces lookahead
aligned = market_df.join(macro_df, on="observation_date")  # NEVER DO THIS
```

Every data pipeline PR must include an automated causal integrity assertion:
```python
assert (aligned["release_date"] <= aligned["decision_date"]).all(), "FUTURE LEAKAGE DETECTED"
```

### Fractional Differencing Protocol

1. For each series `i`, find the minimum `d_i* ∈ [0.2, 0.7]` that passes ADF at `p < 0.01`.
2. Never use integer differencing (`d=1`) — it destroys multi-decade memory.
3. Preserve the memory retention check: `corr(x_t, (1-B)^d x_t) ≥ 0.70`.
4. If ADF fails at `d=0.7`, escalate to Level-1 fallback (Wavelet MODWT).

### Ragged-Edge Interpolation

- Use PCHIP (Piecewise Cubic Hermite) for quarterly/annual → monthly mapping.
- Always maintain the binary mask tensor `M ∈ {0,1}^{T×N×D}` alongside the data tensor.
- Never forward-fill without setting the corresponding mask to 0.

## Data Sources & Feature Domains

Three domains, 16 features each:

| Domain | Key Sources |
|--------|-------------|
| Monetary & Debt | FRED, IMF IFS, BIS, World Bank WDI |
| Internal Order | EPU index, BLS, congressional records |
| External Order | UN Comtrade, IMF DOTS, BIS cross-border flows |

## Storage

- **ArcticDB** for bitemporal snapshots (versioned, immutable).
- **Polars + Apache Arrow** for zero-copy in-memory processing.

## Validation Gate (Must Pass Before Stage 2)

| Metric | Target |
|--------|--------|
| ADF Stationarity | `p < 0.01` on ≥ 98% of series |
| Memory Retention | `corr ≥ 0.70` |
| Imputation Fidelity | `NRMSE ≤ 0.08` on synthetic 20% MCAR drop |
| Causal Integrity | 0.0% future leakage |

## Contingency Fallbacks

If the primary approach fails validation:

**Stationarity failures:**
- L1: Wavelet MODWT (J=6 scales, discard A6, retain D1–D5)
- L2: Christiano-Fitzgerald Bandpass (18–96 month band)

**Imputation failures:**
- L1: GP State-Space Interpolator (Matérn-3/2 kernel per feature)
- L2: EM Dynamic Factor Imputer (Kalman-smoothed DFM)

## Code Location

All Stage 1 code lives in `src/data/`. Key modules:
- `src/data/ingestion.py` — Raw data fetching and ArcticDB storage
- `src/data/fractional_diff.py` — Fractional differencing implementation
- `src/data/panel_sync.py` — Polars asof joins and panel alignment
- `src/data/interpolation.py` — PCHIP and mask tensor generation
- `src/data/validation.py` — ADF tests, memory checks, leakage audits
