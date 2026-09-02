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
