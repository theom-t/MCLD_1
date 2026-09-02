# MCLD-1: Architectural & Research Decision Log

This document serves as the immutable ledger of methodological, mathematical, and architectural decisions made throughout the MCLD-1 project.

**Format for Entries:**
- **Date:** YYYY-MM-DD
- **Context:** What problem or design choice were we facing?
- **Decision:** What did we decide to do?
- **Alternatives Considered:** What else did we look at (and what did the literature say)?
- **Rationale ("The Why"):** Why is this the best choice for the integrity of the whitepaper and the model?

---

### [2026-09-02] Project Foundation & Agentic Workflow
- **Context:** Establishing the repository structure, dependency management, and agent workflow for the MCLD-1 research project.
- **Decision:** Use `uv` for lightning-fast dependency management, enforce strict `.gitignore` rules preventing data commits, and establish a mandatory "Discuss → Research → Log → Code" protocol in the `AGENTS.md` guidelines.
- **Alternatives Considered:** Standard `pip`/`conda` environments; standard agile coding practices without explicit decision logging.
- **Rationale:** Because this culminates in an academic/quant whitepaper, methodological transparency is paramount. The "Discuss → Research → Log" loop ensures every mathematical and architectural choice has theoretical backing and is perfectly documented for the final paper write-up. `uv` was selected over conda to avoid conflicts with the host's existing `dollar_alpha` RTX 5090 Blackwell PyTorch setup.

---

### [2026-09-02] Stage 1: Bitemporal Data & Zero-Leakage Enforcement
- **Context:** Free macro data APIs (DBnomics, World Bank, BIS) provide observation dates but rarely provide the exact historical publication timestamps necessary to prevent lookahead bias.
- **Decision:** We will use a hybrid ingestion architecture. (1) Where available (via FRED/ALFRED), we will extract true `realtime_start` publication vintages. (2) For all other free sources (DBnomics), we will construct a conservative `release_date` by adding IMF SDDS prescribed lag offsets to the observation date (e.g., +30 days for CPI, +1 day for Policy Rates). 
- **Alternatives Considered:** Relying entirely on expensive commercial Point-in-Time datasets (e.g., Macrobond), or using purely observation dates (which introduces severe lookahead bias, ruining the integrity of the backtest).
- **Rationale:** This hybrid approach maintains 100% causal integrity (zero future leakage). By using IMF-mandated upper-bound reporting lags for DBnomics data, we are actually penalizing our model slightly (simulating a worst-case data delay), which makes the resulting whitepaper metrics highly conservative and robust to academic scrutiny.
