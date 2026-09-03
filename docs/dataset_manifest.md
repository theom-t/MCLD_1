# MCLD-1 Dataset Manifest & Curation Plan

## 1. Features Extracted (23 Core Indicators)
Our master bitemporal tensor tracks 23 distinct macroeconomic indicators mapped directly to Ray Dalio's Big Cycle framework.

### Monetary & Debt Cycle
1. **`policy_rate` (Monthly):** Central bank target interest rate.
2. **`10y_yield` (Monthly):** Sovereign borrowing costs.
3. **`m2_money` (Weekly):** Broad liquidity and money printing.
4. **`credit_to_gdp_gap` (Quarterly):** BIS leading indicator for banking crises.
5. **`gov_debt_gdp` (Annual):** Central government leverage (World Bank).
6. **`stock_market_index` (Monthly):** Broad equity market pricing (OECD).

### Internal Order & Demographics
7. **`cpi_yoy` (Monthly):** Consumer inflation.
8. **`unemployment` (Monthly):** Labor market slack.
9. **`gini_disp` (Annual):** Post-tax inequality / wealth gap (SWIID).
10. **`working_age_pop` (Annual):** Demographic dividend (World Bank).
11. **`housing_price_index` (Quarterly):** Real estate bubble proxy (BIS).
12. **`consumer_confidence` (Monthly):** Internal domestic sentiment (OECD).

### External Order & Power
13. **`exports` (Monthly):** Global demand for domestic output.
14. **`imports` (Monthly):** Domestic demand for foreign output.
15. **`current_account_gdp` (Annual):** Capital flow proxy.
16. **`fx_reserves` (Monthly):** Central bank war-chest (IMF).
17. **`reer` (Monthly):** Real Effective Exchange Rate / Global competitiveness (BIS).

### Real Economy & Geopolitics
18. **`industrial_production` (Monthly):** Real economic output (GDP proxy).
19. **`market_cap_gdp` (Annual):** Financialization of the real economy.
20. **`rnd_gdp` (Annual):** Innovation capacity.
21. **`epu_index` (Monthly):** Economic Policy Uncertainty.
22. **`gpr_index` (Monthly):** Geopolitical Risk.
23. **`milex_gdp` (Annual):** Military Expenditure / Hard Power (SIPRI).

---

## 2. Country Selection Logic
We initially ingested 48 nations. However, many frontier markets (e.g., Vietnam, Nigeria, Egypt) completely lack historical records for core fundamentals like inflation (`cpi_yoy`), output (`industrial_production`), and liquidity (`m2_money`) in the free databases (FRED/IMF IFS). 

**The Curation Rule:** Any country with **10 or fewer valid features** (where a feature has >50 historical observations) is dropped. This prevents the Temporal-JEPA from training on empty noise matrices.

*   **Dropped (14):** SG, PH, PE, SA, AR, TH, MY, CO, RO, PK, EG, AE, NG, VN.
*   **Retained (34):** Listed below.

---

## 3. The 34 Retained Nations: Data Ranges & Archetypes
Below is the explicit mapping of our curated dataset. Notice how it perfectly captures the structural mechanics of the Dalio cycle across 4 distinct archetypes. 

*(Note: Data for developed markets generally begins between 1946–1960, while emerging markets generally begin between 1979–1985).*

### Archetype 1: The Rising Challengers (Mid/Late Expansion)
These represent the massive, rapidly financializing geopolitical rivals to the existing hegemony.
*   **Brazil (BR)** (Starts 1950) | *Missing:* 10y_yield, gov_debt, market_cap, rnd, unemployment
*   **India (IN)** (Starts 1946) | *Missing:* 10y_yield, consumer_conf, gov_debt, m2, market_cap, rnd, unemployment
*   **Russia (RU)** (Starts 1985) | *Missing:* consumer_conf, current_acct, gini, gov_debt, market_cap, milex, rnd, working_age_pop
*   **China (CN)** (Starts 1960) | *Missing:* 10y_yield, current_acct, gini, gov_debt, industrial_production, market_cap, milex, rnd, unemployment

### Archetype 2: The Resource & Boom/Bust Edge (Volatile)
These nations track the volatile edges of the cycle, acting as the commodity engine for the global machine.
*   **Chile (CL)** (Starts 1950) | *Missing:* consumer_conf, gov_debt, m2, market_cap, rnd, unemployment
*   **Mexico (MX)** (Starts 1949) | *Missing:* current_acct, gov_debt, industrial_production, market_cap, rnd, unemployment, working_age_pop
*   **South Africa (ZA)** (Starts 1950) | *Missing:* current_acct, epu, gini, gov_debt, industrial_production, market_cap, rnd, unemployment, working_age_pop

### Archetype 3: The Emerging Engines (Early Expansion)
These nations are currently entering their demographic and manufacturing expansion phases.
*   **Indonesia (ID)** (Starts 1950) | *Missing:* 10y_yield, current_acct, epu, gov_debt, industrial_production, market_cap, rnd, unemployment
*   **Turkey (TR)** (Starts 1950) | *Missing:* 10y_yield, current_acct, epu, gini, gov_debt, market_cap, rnd, unemployment, working_age_pop

### Archetype 4: Core Developed / Mature Cycle (Late/Declining)
The incumbent powers experiencing peak debt, declining demographics, and internal polarization.
*   **Super-Dense (18-19 Features)**
    *   **United States (US) (1919)** | *Missing:* current_acct, gov_debt, market_cap, rnd, working_age_pop
    *   **Japan (JP) (1946)** | *Missing:* current_acct, gov_debt, rnd, unemployment
    *   **Germany (DE) (1950)** | *Missing:* current_acct, gov_debt, m2, rnd
    *   **France (FR) (1950)** | *Missing:* gov_debt, market_cap, rnd, unemployment
    *   **Italy (IT) (1947)** | *Missing:* gov_debt, market_cap, rnd, unemployment
    *   **Canada (CA) (1914)** | *Missing:* gov_debt, m2, market_cap, rnd, unemployment
*   **Dense (14-17 Features)**
    *   **Finland (FI)** (1950), **United Kingdom (GB)** (1946), **Belgium (BE)** (1950), **Ireland (IE)** (1950), **Denmark (DK)** (1946), **Spain (ES)** (1950), **South Korea (KR)** (1950), **Hungary (HU)** (1960), **Sweden (SE)** (1946), **Netherlands (NL)** (1950), **Australia (AU)** (1950), **Portugal (PT)** (1950), **Norway (NO)** (1949), **Greece (GR)** (1949), **Israel (IL)** (1950), **Poland (PL)** (1979), **Switzerland (CH)** (1946), **Czechia (CZ)** (1960).
    *   *(Missing features for this group generally consist of niche World Bank metrics like R&D to GDP, Market Cap to GDP, or Annual Government Debt).*

---

## Conclusion & Next Step
As clearly shown above, dropping the bottom 14 countries fundamentally protects the integrity of our model, while preserving 34 highly descriptive nations that perfectly encapsulate every phase of the macro-cycle over a 75-year time horizon.

**Next Step:** I will log this exact selection criteria into `docs/decision_log.md` and build `src/data/filter_panel.py` to physically drop the 14 excluded nations from the parquet file before we begin mathematical interpolation.
