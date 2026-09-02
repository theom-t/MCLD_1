---
name: mcld-research
description: >-
  Use this skill when conducting literature reviews, writing research sections,
  evaluating prior art, formulating research questions, or drafting whitepaper
  content for the MCLD-1 project.
---

# MCLD-1 Research Skill

## Scope

This skill covers all research activities for the MCLD-1 project including:
- Literature surveys and prior art analysis
- Research question refinement (RQ1, RQ2, RQ3)
- Whitepaper section drafting
- Mathematical formulation documentation
- Novelty analysis against existing work

## Research Questions

Always frame research work around the three core RQs:

1. **RQ1 (Representation Learning):** Can a Temporal-JEPA learn continuous, economically decodable representations of secular macro cycles from noisy, asynchronous monthly panels — without ground-truth regime labels?
2. **RQ2 (Bayesian Dynamics):** Can a continuous-time SVGP vector field accurately predict multi-step macro trajectory drift and yield calibrated epistemic uncertainty as an early-warning signal for sovereign regime bifurcations?
3. **RQ3 (Empirical Asset Pricing):** Does conditioning cross-sectional sovereign allocation on latent cycle velocity, penalised by GPR epistemic uncertainty, generate persistent risk-adjusted excess returns after non-linear market impact and carry drag?

## Literature Search Domains

When searching for prior art, cover these domains:

| Domain | Key Search Terms |
|--------|-----------------|
| Self-supervised time-series | JEPA, VICReg, BYOL, time-series SSL, masked autoencoder, PatchTST |
| Macro regime modelling | Hidden Markov Model macro, Dynamic Factor Model, Markov-switching, regime detection |
| Gaussian Processes for macro | GP-DFM, sparse variational GP, spectral mixture kernel, non-stationary GP |
| Global macro quant | cross-sectional momentum, sovereign carry, currency value, macro factor investing |
| Dalio's Big Cycle | long-term debt cycle, reserve currency dynamics, empire lifecycle |

## Key Prior Art Insights to Preserve

These findings from the initial research must be reflected in all research output:

1. **Predict latent vectors, not raw data.** Generative reconstruction wastes capacity on high-frequency noise.
2. **Avoid isotropic priors.** Standard JEPA regularisation forces flat Euclidean geometry onto non-linear dynamical systems. Consider Hamiltonian / symplectic constraints.
3. **Additive GP kernels solve scaling.** Multiplicative kernels are O(M^D); additive kernels linearise to O(D × M̃).

## Writing Standards

1. **Mathematical notation:** Use LaTeX syntax. Define all symbols on first use.
2. **Citations:** Use author-year format (e.g., Dalio, 2021; LeCun, 2022).
3. **Claims require evidence:** Every performance claim must reference a specific validation metric and its threshold from the project plan.
4. **Comparative framing:** Always position MCLD-1 against at least two baselines:
   - Classical Dynamic Factor Model (DFM)
   - Hamilton Markov-Switching model
5. **Structure:** Follow the standard academic structure — Abstract, Introduction, Related Work, Method, Experiments, Results, Discussion, Conclusion.

## Research Output Locations

- Literature notes → `docs/research/`
- Mathematical derivations → `docs/math/`
- Whitepaper drafts → `docs/paper/`
