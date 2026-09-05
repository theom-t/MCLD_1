# MCLD-1: Macro-Cycle Latent Dynamics Model

MCLD-1 is a self-supervised quantitative framework designed to:
1. Ingest bitemporal macroeconomic panel data across ~50 sovereign nations.
2. Learn latent regime representations via a Temporal-JEPA (VICReg).
3. Model autonomous macro-cycle dynamics via a Sparse Variational GP (SVGP).
4. Construct uncertainty-damped, friction-aware global macro portfolios.

## Project Status

- **Stage 1 (Bitemporal Data Ingestion & Stationarity):** ✅ **[COMPLETED]** 
  - *Curated 34 Sovereign Nations across 23 core Dalio-cycle macroeconomic features (1950-2025).*
  - *Successfully built strict backward-looking bitemporal ingestion pipeline.*
  - *Pivoted stationarity engine to Reversible Instance Normalization (RevIN) to preserve secular memory.*
- **Stage 2 (Temporal-JEPA):** ✅ **[COMPLETED]**
  - *Engineered pure Flax functional architecture (RevIN + Causal TCN).*
  - *Ported advanced PID-Variance and Macro-Topological Metric Learning from V-JEPA.*
  - *End-to-end XLA step execution: ~1.0 millisecond.*
- **Stage 3 (SVGP Latent Dynamics):** ⏳ Pending
- **Stage 4 (Risk Budgeting):** ⏳ Pending
- **Stage 5 (Friction-Aware Backtest):** ⏳ Pending

## Project Structure

This repository is modularly structured to support both research and deployment pipelines:

```text
MCLD_1/
├── .agents/                 # Agentic rules and skills
├── configs/                 # YAML configuration files for pipeline stages
├── docs/                    # Architecture plans and research notes
├── notebooks/               # EDA, visualisations, and research scratchpads
├── scripts/                 # CLI entry points and orchestration runners
├── src/                     # Core library code
│   ├── data/                # Stage 1: Ingestion, fractional diff, bitemporal alignment
│   ├── models/
│   │   ├── gp/              # Stage 3: Sparse Variational GP (SVGP)
│   │   └── jepa/            # Stage 2: Temporal-JEPA representation learning
│   ├── portfolio/           # Stage 4: Signal generation and risk budgeting
│   ├── backtest/            # Stage 5: Market impact and execution simulation
│   └── utils/               # Shared utilities
└── tests/                   # Pytest suite
```

## Environment Setup (`uv`)

This project uses `uv` for lightning-fast dependency management and resolution.

> **Hardware Note (RTX 5090 / Blackwell):**  
> As per global configuration, this project requires `jax[cuda12]` for the RTX 5090. Do NOT install conda CUDA packages as they will conflict with the custom `dollar_alpha` library path setup.

### 1. Create and Sync the Environment

Run the following to create the virtual environment (`.venv`) and sync standard dependencies (Polars, ArcticDB, VectorBT, etc.):

```bash
uv sync --extra dev
```

### 2. Install JAX Nightly (Blackwell Support)

Activate the environment and install JAX natively using the CUDA 12.4+ index:

```bash
source .venv/bin/activate
uv pip install -U "jax[cuda12]"
```

## Running the Pipeline

Pipeline execution is divided into the 5 discrete stages outlined in `docs/MCLD_1_project_plan.md`. 
Each stage enforces strict validation gates before allowing progression to the next phase.
