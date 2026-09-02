# MCLD-1 — Agent Rules

## Project Identity

This repository is **MCLD-1 (Macro-Cycle Latent Dynamics Model)** — a self-supervised quantitative framework that:
1. Ingests bitemporal macroeconomic panel data across ~50 sovereign nations.
2. Learns latent regime representations via a Temporal-JEPA (VICReg).
3. Models autonomous macro-cycle dynamics via a Sparse Variational GP (SVGP).
4. Constructs uncertainty-damped, friction-aware global macro portfolios.

The canonical project plan is at `docs/MCLD_1_project_plan.md`. Always consult it before proposing architectural decisions. The original brainstorm transcript is at `docs/inital_brainstorm_source.md`.

## Pipeline Stages

The system is built in 5 sequential stages with strict validation gates:

| Stage | Name | Key Tech |
|-------|------|----------|
| 1 | Bitemporal Data Ingestion | ArcticDB, Polars, Fractional Differencing |
| 2 | Temporal-JEPA Representation | PyTorch, VICReg, TCN Encoder |
| 3 | SVGP Latent Flow Field | GPyTorch / GPflow, Matérn + Spectral kernel |
| 4 | Uncertainty-Damped Risk Budgeting | TreeSHAP, Ascension Projection |
| 5 | Friction-Aware Backtest | VectorBT, NautilusTrader, Almgren-Chriss |

**A stage may NOT proceed to the next until its validation gate thresholds are met.** If thresholds are breached, follow the two-tier contingency protocol (Level-1 intra-paradigm pivot → Level-2 structural fallback) defined in the project plan.

## Technical Stack & Conventions

- **Language:** Python 3.11+
- **ML Framework:** PyTorch 2.x (primary), JAX (optional alternative)
- **GP Modelling:** GPyTorch or GPflow (GPU-accelerated SVGP)
- **Data Pipeline:** Polars + Apache Arrow (zero-copy); ArcticDB for bitemporal storage
- **Backtesting:** VectorBT (vectorised), NautilusTrader (event-driven)
- **Validation:** Riskfolio-Lib (Purged CV), Qlib (IC/RankIC)
- **Hyperparameter Search:** Optuna (multi-objective TPE)
- **Attribution:** SHAP / Captum

## Code Quality Rules

1. **Type hints everywhere.** All function signatures must include type annotations.
2. **Docstrings.** Every public function, class, and module must have a Google-style docstring.
3. **No future data leakage.** Any code that joins, merges, or aligns time-series data MUST use backward-looking joins (`strategy="backward"`) or explicitly assert `t_data ≤ t_decision`. Flag any code that could introduce lookahead bias.
4. **Reproducibility.** All random seeds must be configurable. Training scripts must log seeds, hyperparameters, and git commit hashes.
5. **Tensor dimension comments.** When writing PyTorch/JAX code, annotate tensor shapes in comments (e.g., `# (B, T, D)`).
6. **Config-driven.** Hyperparameters, file paths, and feature lists must live in config files (YAML/TOML), not hard-coded in source.

## Data Hygiene Rules

- **Bitemporal discipline.** Always distinguish between observation date (when data describes) and release date (when data was published). Use release dates for alignment.
- **Fractional differencing.** When making time-series stationary, use fractional differencing with `d ∈ [0.2, 0.7]` to preserve long-term memory. Never default to `d=1` integer differencing.
- **Missing data.** Use PCHIP interpolation with explicit binary masks. Never forward-fill without flagging.

## Research & Writing Rules

- When writing research content, always cite specific metrics and thresholds from the project plan.
- Mathematical notation must use LaTeX-compatible syntax.
- Claims about model performance must reference specific validation metrics (Sharpe, NLPD, PICP, DTW, etc.) and their target thresholds.

## Directory Structure

```
MCLD_1/
├── .agents/                 # Agentic guidelines (this directory)
├── docs/                    # Project plan, brainstorm, research notes
├── src/
│   ├── data/                # Stage 1: ingestion, fractional diff, panel sync
│   ├── models/
│   │   ├── jepa/            # Stage 2: Temporal-JEPA encoder
│   │   └── gp/              # Stage 3: SVGP flow field
│   ├── portfolio/           # Stage 4: risk budgeting, signal construction
│   ├── backtest/            # Stage 5: friction-aware backtest engine
│   └── utils/               # Shared utilities
├── configs/                 # YAML/TOML configuration files
├── tests/                   # Unit and integration tests
├── notebooks/               # Exploration and visualisation
└── scripts/                 # CLI entry points and runners
```
