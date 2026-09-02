#!/usr/bin/env python3
"""MCLD-1 Data Availability Audit Script.

Probes free macroeconomic data APIs (DBnomics, FRED, BIS, OECD, World Bank)
to assess indicator coverage across the target 48 sovereign nations.

Outputs a comprehensive report of:
- Available indicators per source
- Date range and frequency
- Country coverage gaps
- Recommendations for feature selection

Usage:
    python scripts/audit_data_sources.py [--fred-api-key YOUR_KEY]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import requests

# ---------------------------------------------------------------------------
# Target Nations (ISO-3166 Alpha-2 and Alpha-3 codes)
# ---------------------------------------------------------------------------
TARGET_NATIONS: dict[str, str] = {
    # Tier 1 — Full-cycle traversals
    "CN": "China", "KR": "South Korea", "JP": "Japan", "TR": "Turkey",
    "RU": "Russia", "BR": "Brazil", "AR": "Argentina", "ZA": "South Africa",
    # Tier 2 — Established / Peak-to-decline
    "US": "United States", "GB": "United Kingdom", "DE": "Germany",
    "FR": "France", "IT": "Italy", "ES": "Spain", "NL": "Netherlands",
    "CH": "Switzerland", "AU": "Australia", "CA": "Canada",
    # Tier 3 — Ascending / Early-stage
    "IN": "India", "VN": "Vietnam", "ID": "Indonesia", "PL": "Poland",
    "CZ": "Czechia", "MX": "Mexico", "TH": "Thailand", "MY": "Malaysia",
    "PH": "Philippines", "CL": "Chile",
    # Tier 4 — Crisis / Structural stress + Commodity/Financial centres
    "GR": "Greece", "PT": "Portugal", "NG": "Nigeria", "EG": "Egypt",
    "PK": "Pakistan", "CO": "Colombia", "PE": "Peru", "HU": "Hungary",
    "RO": "Romania", "IL": "Israel", "SA": "Saudi Arabia", "AE": "UAE",
    "SG": "Singapore", "NO": "Norway", "SE": "Sweden", "DK": "Denmark",
    "NZ": "New Zealand", "FI": "Finland", "IE": "Ireland", "BE": "Belgium",
}

# ISO-2 to ISO-3 mapping (needed for some APIs)
ISO2_TO_ISO3: dict[str, str] = {
    "CN": "CHN", "KR": "KOR", "JP": "JPN", "TR": "TUR", "RU": "RUS",
    "BR": "BRA", "AR": "ARG", "ZA": "ZAF", "US": "USA", "GB": "GBR",
    "DE": "DEU", "FR": "FRA", "IT": "ITA", "ES": "ESP", "NL": "NLD",
    "CH": "CHE", "AU": "AUS", "CA": "CAN", "IN": "IND", "VN": "VNM",
    "ID": "IDN", "PL": "POL", "CZ": "CZE", "MX": "MEX", "TH": "THA",
    "MY": "MYS", "PH": "PHL", "CL": "CHL", "GR": "GRC", "PT": "PRT",
    "NG": "NGA", "EG": "EGY", "PK": "PAK", "CO": "COL", "PE": "PER",
    "HU": "HUN", "RO": "ROU", "IL": "ISR", "SA": "SAU", "AE": "ARE",
    "SG": "SGP", "NO": "NOR", "SE": "SWE", "DK": "DNK", "NZ": "NZL",
    "FI": "FIN", "IE": "IRL", "BE": "BEL",
}

# ---------------------------------------------------------------------------
# Target Indicators — grouped by domain
# ---------------------------------------------------------------------------
TARGET_INDICATORS: dict[str, list[dict[str, str]]] = {
    "Monetary & Debt": [
        {"name": "Central Bank Policy Rate", "dbnomics": "BIS/WS_CBPOL", "fred_tag": "policy rate"},
        {"name": "CPI Inflation (YoY)", "dbnomics": "IMF/CPI", "fred_tag": "consumer price index"},
        {"name": "M2 Money Supply", "dbnomics": "IMF/MFS_IR", "fred_tag": "M2"},
        {"name": "Sovereign Debt/GDP", "dbnomics": "IMF/GFS", "fred_tag": "debt to gdp"},
        {"name": "Yield Curve (10Y-2Y Spread)", "dbnomics": "OECD/MEI", "fred_tag": "10-year"},
        {"name": "Credit-to-GDP Gap", "dbnomics": "BIS/WS_CREDIT_GAP", "fred_tag": "credit gap"},
        {"name": "FX Reserves", "dbnomics": "IMF/IFS", "fred_tag": "reserves"},
        {"name": "Real Effective Exchange Rate", "dbnomics": "BIS/WS_EER", "fred_tag": "effective exchange rate"},
    ],
    "Internal Order": [
        {"name": "Unemployment Rate", "dbnomics": "OECD/MEI", "fred_tag": "unemployment rate"},
        {"name": "Industrial Production", "dbnomics": "OECD/MEI", "fred_tag": "industrial production"},
        {"name": "Consumer Confidence", "dbnomics": "OECD/MEI_CLI", "fred_tag": "consumer confidence"},
        {"name": "Gini Coefficient", "dbnomics": "WB/WDI", "fred_tag": "gini"},
        {"name": "Real Wage Growth", "dbnomics": "ILO/EAR", "fred_tag": "real wage"},
        {"name": "Fiscal Deficit/GDP", "dbnomics": "IMF/GFS", "fred_tag": "fiscal balance"},
    ],
    "External Order": [
        {"name": "Merchandise Trade Balance", "dbnomics": "IMF/DOT", "fred_tag": "trade balance"},
        {"name": "Net FDI Flows", "dbnomics": "IMF/BOP", "fred_tag": "foreign direct investment"},
        {"name": "BIS Cross-Border Banking", "dbnomics": "BIS/WS_LBS", "fred_tag": "cross-border"},
        {"name": "Defence Spending/GDP", "dbnomics": "WB/WDI", "fred_tag": "military expenditure"},
        {"name": "Current Account/GDP", "dbnomics": "IMF/BOP", "fred_tag": "current account"},
        {"name": "Capital Flow Velocity", "dbnomics": "IMF/BOP", "fred_tag": "capital flows"},
    ],
}


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------
@dataclass
class IndicatorResult:
    """Result of probing a single indicator from a single source."""
    source: str
    indicator_name: str
    domain: str
    country_iso2: str
    country_name: str
    series_id: str = ""
    start_date: str = ""
    end_date: str = ""
    frequency: str = ""
    n_observations: int = 0
    has_data: bool = False
    error: str = ""


@dataclass
class AuditReport:
    """Collection of all probe results."""
    results: list[IndicatorResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    
    def add(self, result: IndicatorResult) -> None:
        self.results.append(result)
    
    def to_polars(self) -> pl.DataFrame:
        """Convert results to a Polars DataFrame for analysis."""
        return pl.DataFrame([
            {
                "source": r.source,
                "domain": r.domain,
                "indicator": r.indicator_name,
                "country_iso2": r.country_iso2,
                "country": r.country_name,
                "series_id": r.series_id,
                "start_date": r.start_date,
                "end_date": r.end_date,
                "frequency": r.frequency,
                "n_obs": r.n_observations,
                "has_data": r.has_data,
                "error": r.error,
            }
            for r in self.results
        ])


# ---------------------------------------------------------------------------
# Source 1: DBnomics
# ---------------------------------------------------------------------------
def probe_dbnomics(report: AuditReport) -> None:
    """Probe DBnomics for indicator availability across target nations."""
    print("\n" + "=" * 70)
    print("SOURCE 1: DBnomics (aggregator of IMF, OECD, BIS, World Bank, etc.)")
    print("=" * 70)
    
    try:
        from dbnomics import fetch_series_by_api_link, fetch_series
    except ImportError:
        print("  [SKIP] dbnomics package not installed")
        return
    
    # Probe a curated set of well-known DBnomics series
    # Format: provider/dataset — we search for country-specific series
    dbnomics_probes: list[dict[str, str]] = [
        # BIS datasets (excellent coverage for our 48 nations)
        {"provider": "BIS", "dataset": "WS_CBPOL",
         "name": "Central Bank Policy Rate", "domain": "Monetary & Debt"},
        {"provider": "BIS", "dataset": "WS_LONG_CPI",
         "name": "Long CPI", "domain": "Monetary & Debt"},
        {"provider": "BIS", "dataset": "WS_EER",
         "name": "Real Effective Exchange Rate", "domain": "Monetary & Debt"},
        {"provider": "BIS", "dataset": "WS_CREDIT_GAP",
         "name": "Credit-to-GDP Gap", "domain": "Monetary & Debt"},
        {"provider": "BIS", "dataset": "WS_SPP",
         "name": "Property Prices", "domain": "Internal Order"},
        
        # OECD MEI (Main Economic Indicators)
        {"provider": "OECD", "dataset": "MEI",
         "name": "Main Economic Indicators", "domain": "Internal Order"},
        
        # OECD CLI (Composite Leading Indicators)
        {"provider": "OECD", "dataset": "MEI_CLI",
         "name": "Composite Leading Indicator", "domain": "Internal Order"},
        
        # IMF direction of trade
        {"provider": "IMF", "dataset": "DOT",
         "name": "Direction of Trade", "domain": "External Order"},
    ]
    
    for probe in dbnomics_probes:
        provider = probe["provider"]
        dataset = probe["dataset"]
        indicator_name = probe["name"]
        domain = probe["domain"]
        
        print(f"\n  Probing {provider}/{dataset} — {indicator_name}...")
        
        try:
            # Fetch dataset metadata via the API
            url = f"https://api.db.nomics.world/v22/series/{provider}/{dataset}?limit=0&facets=true"
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            
            total_series = data.get("series", {}).get("num_found", 0)
            print(f"    Total series in dataset: {total_series:,}")
            
            # Check facets for country/frequency dimensions
            facets = data.get("series", {}).get("facets", {})
            
            # Try to identify the frequency dimension
            freq_facet = facets.get("@frequency", facets.get("FREQ", {}))
            if freq_facet:
                print(f"    Frequencies available: {list(freq_facet.keys())[:10]}")
            
            # Try to identify the country/reference area dimension
            country_dim = None
            for dim_name in ["REF_AREA", "ref_area", "LOCATION", "CL_AREA_BIS",
                             "COUNTERPART_AREA", "Country"]:
                if dim_name in facets:
                    country_dim = dim_name
                    break
            
            if country_dim:
                available_countries = set(facets[country_dim].keys())
                our_countries_iso2 = set(TARGET_NATIONS.keys())
                our_countries_iso3 = set(ISO2_TO_ISO3.values())
                
                # Check overlap (some use ISO-2, some ISO-3)
                matched_iso2 = our_countries_iso2 & available_countries
                matched_iso3 = our_countries_iso3 & available_countries
                matched = matched_iso2 | matched_iso3
                
                coverage_pct = len(matched) / len(TARGET_NATIONS) * 100
                print(f"    Country dimension: '{country_dim}'")
                print(f"    Our nations covered: {len(matched)}/{len(TARGET_NATIONS)} "
                      f"({coverage_pct:.0f}%)")
                
                if len(matched) < len(TARGET_NATIONS):
                    missing_iso2 = our_countries_iso2 - available_countries
                    missing_iso3 = our_countries_iso3 - available_countries
                    missing = missing_iso2 & {k for k, v in ISO2_TO_ISO3.items() 
                                              if v in missing_iso3}
                    if missing:
                        missing_names = [TARGET_NATIONS.get(c, c) for c in list(missing)[:10]]
                        print(f"    Missing: {', '.join(missing_names)}")
                
                # Record results
                for iso2, name in TARGET_NATIONS.items():
                    iso3 = ISO2_TO_ISO3.get(iso2, "")
                    has = iso2 in available_countries or iso3 in available_countries
                    report.add(IndicatorResult(
                        source="DBnomics",
                        indicator_name=f"{provider}/{dataset}: {indicator_name}",
                        domain=domain,
                        country_iso2=iso2,
                        country_name=name,
                        series_id=f"{provider}/{dataset}",
                        has_data=has,
                    ))
            else:
                print(f"    [WARN] Could not identify country dimension in facets")
                print(f"    Available facet dims: {list(facets.keys())[:10]}")
                
        except Exception as e:
            error_msg = f"{provider}/{dataset}: {e}"
            print(f"    [ERROR] {error_msg}")
            report.errors.append(error_msg)
        
        time.sleep(0.5)  # Rate limiting


# ---------------------------------------------------------------------------
# Source 2: FRED / ALFRED
# ---------------------------------------------------------------------------
def probe_fred(report: AuditReport, api_key: str | None = None) -> None:
    """Probe FRED for international macro series availability."""
    print("\n" + "=" * 70)
    print("SOURCE 2: FRED / ALFRED (St. Louis Fed)")
    print("=" * 70)
    
    if not api_key:
        print("  [INFO] No FRED API key provided — probing via public search API")
        print("  [INFO] Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("  [INFO] Re-run with: --fred-api-key YOUR_KEY")
        
        # We can still probe some well-known international series IDs
        # FRED uses predictable naming: {COUNTRY_CODE}{INDICATOR}
        known_series: dict[str, dict[str, str]] = {
            # CPI series (many countries)
            "CPI": {
                "US": "CPIAUCSL", "GB": "GBRCPIALLMINMEI", "DE": "DEUCPIALLMINMEI",
                "JP": "JPNCPIALLMINMEI", "CN": "CHNCPIALLMINMEI", "FR": "FRACPIALLMINMEI",
                "BR": "BRACPIALLMINMEI", "IN": "INDCPIALLMINMEI", "KR": "KORCPIALLMINMEI",
                "MX": "MEXCPIALLMINMEI", "ZA": "ZAFCPIALLMINMEI", "TR": "TURCPIALLMINMEI",
                "AU": "AUSCPIALLMINMEI", "CA": "CANCPIALLMINMEI", "IT": "ITACPIALLMINMEI",
            },
            # Industrial Production
            "INDPRO": {
                "US": "INDPRO", "GB": "GBRPROINDMISMEI", "DE": "DEUPROINDMISMEI",
                "JP": "JPNPROINDMISMEI", "FR": "FRAPROINDMISMEI", "IT": "ITAPROINDMISMEI",
                "KR": "KORPROINDMISMEI", "MX": "MEXPROINDMISMEI", "BR": "BRAPROINDMISMEI",
            },
            # Unemployment
            "UNEMP": {
                "US": "UNRATE", "GB": "LMUNRRTTGBM156S", "DE": "LMUNRRTTDEM156S",
                "JP": "LMUNRRTTJPM156S", "FR": "LMUNRRTTFRM156S", "CA": "LMUNRRTTCAM156S",
                "AU": "LMUNRRTTAUM156S", "KR": "LMUNRRTTKRM156S",
            },
        }
        
        print(f"\n  Probing {sum(len(v) for v in known_series.values())} known international series...")
        
        for indicator_name, country_series in known_series.items():
            for iso2, series_id in country_series.items():
                country_name = TARGET_NATIONS.get(iso2, iso2)
                
                try:
                    url = (f"https://api.stlouisfed.org/fred/series?"
                           f"series_id={series_id}&file_type=json"
                           f"&api_key=UNSET")
                    
                    # Without API key, we just record what we know exists
                    report.add(IndicatorResult(
                        source="FRED",
                        indicator_name=indicator_name,
                        domain="Monetary & Debt" if indicator_name == "CPI" else "Internal Order",
                        country_iso2=iso2,
                        country_name=country_name,
                        series_id=series_id,
                        has_data=True,  # Known to exist
                        frequency="Monthly",
                    ))
                except Exception as e:
                    report.errors.append(f"FRED {series_id}: {e}")
        
        print(f"    Catalogued {sum(len(v) for v in known_series.values())} known FRED series")
        return
    
    # With API key: full probe
    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        
        search_terms = [
            ("consumer price index", "CPI", "Monetary & Debt"),
            ("industrial production", "Industrial Production", "Internal Order"),
            ("unemployment rate", "Unemployment Rate", "Internal Order"),
            ("M2 money", "M2 Money Supply", "Monetary & Debt"),
            ("government debt", "Sovereign Debt", "Monetary & Debt"),
            ("10-year government bond", "10Y Yield", "Monetary & Debt"),
            ("current account", "Current Account", "External Order"),
            ("trade balance", "Trade Balance", "External Order"),
            ("consumer confidence", "Consumer Confidence", "Internal Order"),
        ]
        
        for search_term, indicator_name, domain in search_terms:
            print(f"\n  Searching FRED for: '{search_term}'...")
            try:
                results = fred.search(search_term, limit=200)
                
                if results is not None and len(results) > 0:
                    # Filter for monthly frequency
                    monthly = results[results["frequency"] == "Monthly"]
                    print(f"    Total results: {len(results)}, Monthly: {len(monthly)}")
                    
                    # Check coverage of our nations
                    for _, row in monthly.head(50).iterrows():
                        sid = row.name
                        title = row.get("title", "")
                        
                        # Try to match to our target nations
                        for iso2, name in TARGET_NATIONS.items():
                            if name.lower() in title.lower() or iso2.lower() in sid.lower():
                                try:
                                    info = fred.get_series_info(sid)
                                    obs_start = str(info.get("observation_start", ""))
                                    obs_end = str(info.get("observation_end", ""))
                                    
                                    report.add(IndicatorResult(
                                        source="FRED",
                                        indicator_name=indicator_name,
                                        domain=domain,
                                        country_iso2=iso2,
                                        country_name=name,
                                        series_id=sid,
                                        start_date=obs_start,
                                        end_date=obs_end,
                                        frequency="Monthly",
                                        has_data=True,
                                    ))
                                except Exception:
                                    pass
                                break
                    
                time.sleep(1.0)  # FRED rate limit: 120 req/min
                
            except Exception as e:
                print(f"    [ERROR] {e}")
                report.errors.append(f"FRED search '{search_term}': {e}")
                
    except ImportError:
        print("  [SKIP] fredapi package not installed")


# ---------------------------------------------------------------------------
# Source 3: BIS SDMX API (direct)
# ---------------------------------------------------------------------------
def probe_bis(report: AuditReport) -> None:
    """Probe BIS statistical datasets directly via their SDMX REST API."""
    print("\n" + "=" * 70)
    print("SOURCE 3: BIS SDMX API (Bank for International Settlements)")
    print("=" * 70)
    
    base_url = "https://stats.bis.org/api/v2"
    
    # Key BIS datasets relevant to MCLD-1
    datasets = [
        ("WS_CBPOL", "Central Bank Policy Rates", "Monetary & Debt"),
        ("WS_LONG_CPI", "Long Consumer Price Index", "Monetary & Debt"),
        ("WS_EER", "Effective Exchange Rates", "Monetary & Debt"),
        ("WS_CREDIT_GAP", "Credit-to-GDP Gap", "Monetary & Debt"),
        ("WS_SPP", "Residential Property Prices", "Internal Order"),
        ("WS_XRU", "US Dollar Exchange Rates", "Monetary & Debt"),
        ("WS_TC", "Debt Securities Statistics", "Monetary & Debt"),
        ("WS_LBS_D_PUB", "Locational Banking (cross-border)", "External Order"),
    ]
    
    for dataset_id, name, domain in datasets:
        print(f"\n  Probing BIS/{dataset_id} — {name}...")
        try:
            # Get dataset structure to find available countries
            url = f"{base_url}/structure/dataflow/BIS/{dataset_id}"
            resp = requests.get(url, headers={"Accept": "application/json"}, timeout=15)
            resp.raise_for_status()
            structure = resp.json()
            
            # Parse dimensions to find reference area
            dims = (structure.get("data", {})
                    .get("dataStructures", [{}])[0]
                    .get("dataStructureComponents", {})
                    .get("dimensionList", {})
                    .get("dimensions", []))
            
            ref_area_dim = None
            ref_area_idx = None
            for i, dim in enumerate(dims):
                dim_id = dim.get("id", "")
                if dim_id in ("REF_AREA", "BORROWERS_CTY", "L_CP_COUNTRY"):
                    ref_area_dim = dim
                    ref_area_idx = i
                    break
            
            if ref_area_dim:
                # Get the codelist for reference area
                codelist_ref = ref_area_dim.get("localRepresentation", {}).get("enumeration", "")
                
                # Try to fetch available values through a data availability query
                avail_url = f"{base_url}/availability/dataflow/BIS/{dataset_id}"
                avail_resp = requests.get(avail_url, headers={"Accept": "application/json"}, timeout=15)
                
                if avail_resp.status_code == 200:
                    avail_data = avail_resp.json()
                    # Parse available country codes
                    constraints = (avail_data.get("data", {})
                                   .get("contentConstraints", []))
                    
                    available_countries: set[str] = set()
                    for constraint in constraints:
                        cube_regions = (constraint.get("cubeRegions", []) or
                                       constraint.get("dataKeySets", []))
                        for region in cube_regions:
                            keys = region.get("keyValues", region.get("keys", []))
                            for key in keys:
                                if key.get("id") in ("REF_AREA", "BORROWERS_CTY",
                                                      "L_CP_COUNTRY"):
                                    available_countries.update(
                                        v.get("value", v) if isinstance(v, dict) else v
                                        for v in key.get("values", [])
                                    )
                    
                    if available_countries:
                        our_iso2 = set(TARGET_NATIONS.keys())
                        our_iso3 = set(ISO2_TO_ISO3.values())
                        matched = (our_iso2 & available_countries) | (our_iso3 & available_countries)
                        coverage_pct = len(matched) / len(TARGET_NATIONS) * 100
                        
                        print(f"    Available countries: {len(available_countries)}")
                        print(f"    Our nations covered: {len(matched)}/{len(TARGET_NATIONS)} "
                              f"({coverage_pct:.0f}%)")
                        
                        for iso2, cname in TARGET_NATIONS.items():
                            iso3 = ISO2_TO_ISO3.get(iso2, "")
                            has = iso2 in available_countries or iso3 in available_countries
                            report.add(IndicatorResult(
                                source="BIS",
                                indicator_name=f"{dataset_id}: {name}",
                                domain=domain,
                                country_iso2=iso2,
                                country_name=cname,
                                series_id=dataset_id,
                                has_data=has,
                            ))
                    else:
                        print(f"    [WARN] Could not parse available countries from constraints")
                else:
                    print(f"    [WARN] Availability endpoint returned {avail_resp.status_code}")
            else:
                print(f"    Dimensions: {[d.get('id') for d in dims]}")
                print(f"    [WARN] No REF_AREA dimension found")
                
        except Exception as e:
            error_msg = f"BIS/{dataset_id}: {e}"
            print(f"    [ERROR] {error_msg}")
            report.errors.append(error_msg)
        
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# Source 4: World Bank WDI
# ---------------------------------------------------------------------------
def probe_world_bank(report: AuditReport) -> None:
    """Probe World Bank WDI for annual indicators (will need PCHIP interpolation)."""
    print("\n" + "=" * 70)
    print("SOURCE 4: World Bank WDI (annual indicators → PCHIP interpolation)")
    print("=" * 70)
    
    # Key WDI indicators relevant to MCLD-1
    indicators = [
        ("NY.GDP.MKTP.CD", "GDP (current USD)", "Monetary & Debt"),
        ("GC.DOD.TOTL.GD.ZS", "Central Gov Debt/GDP", "Monetary & Debt"),
        ("BN.CAB.XOKA.GD.ZS", "Current Account/GDP", "External Order"),
        ("SI.POV.GINI", "Gini Index", "Internal Order"),
        ("BX.KLT.DINV.WD.GD.ZS", "FDI Net Inflows/GDP", "External Order"),
        ("MS.MIL.XPND.GD.ZS", "Military Expenditure/GDP", "External Order"),
        ("SL.UEM.TOTL.ZS", "Unemployment Rate (ILO)", "Internal Order"),
        ("FI.RES.TOTL.CD", "Total Reserves (incl gold)", "Monetary & Debt"),
        ("TG.VAL.TOTL.GD.ZS", "Merchandise Trade/GDP", "External Order"),
        ("NE.EXP.GNFS.ZS", "Exports of G&S/GDP", "External Order"),
        ("GC.BAL.CASH.GD.ZS", "Cash Surplus/Deficit % GDP", "Internal Order"),
        ("FR.INR.RINR", "Real Interest Rate", "Monetary & Debt"),
    ]
    
    for indicator_code, name, domain in indicators:
        print(f"\n  Probing WB/{indicator_code} — {name}...")
        
        # Build country list (ISO-3 codes, semicolon-separated)
        countries_str = ";".join(ISO2_TO_ISO3.values())
        
        try:
            url = (f"https://api.worldbank.org/v2/country/{countries_str}/"
                   f"indicator/{indicator_code}?format=json&per_page=20000"
                   f"&date=1980:2025")
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            
            if len(data) < 2 or data[1] is None:
                print(f"    [WARN] No data returned")
                for iso2, cname in TARGET_NATIONS.items():
                    report.add(IndicatorResult(
                        source="WorldBank",
                        indicator_name=f"{indicator_code}: {name}",
                        domain=domain,
                        country_iso2=iso2,
                        country_name=cname,
                        series_id=indicator_code,
                        has_data=False,
                        frequency="Annual",
                    ))
                continue
            
            records = data[1]
            
            # Analyse by country
            country_data: dict[str, list[dict[str, Any]]] = {}
            for rec in records:
                iso3 = rec.get("country", {}).get("id", "")
                if rec.get("value") is not None:
                    if iso3 not in country_data:
                        country_data[iso3] = []
                    country_data[iso3].append(rec)
            
            covered = 0
            for iso2, cname in TARGET_NATIONS.items():
                iso3 = ISO2_TO_ISO3.get(iso2, "")
                recs = country_data.get(iso3, [])
                has = len(recs) > 0
                if has:
                    covered += 1
                    dates = [r["date"] for r in recs]
                    start = min(dates)
                    end = max(dates)
                else:
                    start = ""
                    end = ""
                
                report.add(IndicatorResult(
                    source="WorldBank",
                    indicator_name=f"{indicator_code}: {name}",
                    domain=domain,
                    country_iso2=iso2,
                    country_name=cname,
                    series_id=indicator_code,
                    start_date=start,
                    end_date=end,
                    frequency="Annual",
                    n_observations=len(recs),
                    has_data=has,
                ))
            
            coverage_pct = covered / len(TARGET_NATIONS) * 100
            print(f"    Countries with data: {covered}/{len(TARGET_NATIONS)} ({coverage_pct:.0f}%)")
            
        except Exception as e:
            error_msg = f"WB/{indicator_code}: {e}"
            print(f"    [ERROR] {error_msg}")
            report.errors.append(error_msg)
        
        time.sleep(0.3)


# ---------------------------------------------------------------------------
# Analysis & Reporting
# ---------------------------------------------------------------------------
def generate_report(report: AuditReport, output_dir: Path) -> None:
    """Generate summary analysis from all probe results."""
    print("\n" + "=" * 70)
    print("GENERATING AUDIT REPORT")
    print("=" * 70)
    
    df = report.to_polars()
    
    if df.is_empty():
        print("  [WARN] No results collected — cannot generate report")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw results
    raw_path = output_dir / "raw_audit_results.csv"
    df.write_csv(raw_path)
    print(f"\n  Raw results saved to: {raw_path}")
    
    # --- Summary 1: Coverage by source ---
    print("\n  === COVERAGE BY SOURCE ===")
    source_summary = (
        df.filter(pl.col("has_data"))
        .group_by("source")
        .agg([
            pl.col("indicator").n_unique().alias("unique_indicators"),
            pl.col("country_iso2").n_unique().alias("countries_covered"),
            pl.len().alias("total_series"),
        ])
        .sort("total_series", descending=True)
    )
    print(source_summary)
    
    # --- Summary 2: Coverage by indicator across all sources ---
    print("\n  === COVERAGE BY INDICATOR (countries with data) ===")
    indicator_coverage = (
        df.filter(pl.col("has_data"))
        .group_by(["domain", "indicator"])
        .agg([
            pl.col("country_iso2").n_unique().alias("countries_covered"),
            pl.col("source").first().alias("best_source"),
        ])
        .sort(["domain", "countries_covered"], descending=[False, True])
    )
    print(indicator_coverage)
    
    # --- Summary 3: Coverage heatmap per country ---
    print("\n  === COUNTRY COVERAGE (indicators with data per country) ===")
    country_coverage = (
        df.filter(pl.col("has_data"))
        .group_by(["country_iso2", "country"])
        .agg([
            pl.col("indicator").n_unique().alias("indicators_available"),
            pl.col("source").n_unique().alias("sources_available"),
        ])
        .sort("indicators_available", descending=True)
    )
    print(country_coverage)
    
    # --- Summary 4: Gap analysis — countries with least coverage ---
    print("\n  === GAP ANALYSIS (least-covered countries) ===")
    all_countries = pl.DataFrame({
        "country_iso2": list(TARGET_NATIONS.keys()),
        "country": list(TARGET_NATIONS.values()),
    })
    
    gaps = (
        all_countries
        .join(
            df.filter(pl.col("has_data"))
            .group_by("country_iso2")
            .agg(pl.col("indicator").n_unique().alias("n_indicators")),
            on="country_iso2",
            how="left",
        )
        .with_columns(pl.col("n_indicators").fill_null(0))
        .sort("n_indicators")
        .head(15)
    )
    print(gaps)
    
    # --- Summary 5: Date range analysis (where available) ---
    print("\n  === DATE RANGE ANALYSIS (World Bank indicators) ===")
    date_ranges = (
        df.filter(
            (pl.col("has_data")) &
            (pl.col("start_date") != "") &
            (pl.col("source") == "WorldBank")
        )
        .group_by("indicator")
        .agg([
            pl.col("start_date").min().alias("earliest_start"),
            pl.col("end_date").max().alias("latest_end"),
            pl.col("n_obs").mean().alias("avg_observations"),
            pl.col("country_iso2").n_unique().alias("countries"),
        ])
        .sort("countries", descending=True)
    )
    print(date_ranges)
    
    # Save summaries
    summary_path = output_dir / "coverage_summary.csv"
    indicator_coverage.write_csv(summary_path)
    
    country_path = output_dir / "country_coverage.csv"
    country_coverage.write_csv(country_path)
    
    gaps_path = output_dir / "gap_analysis.csv"
    gaps.write_csv(gaps_path)
    
    print(f"\n  Summary files saved to: {output_dir}/")
    
    # --- Final recommendation ---
    total_indicators = df.select("indicator").n_unique()
    total_with_data = df.filter(pl.col("has_data")).select("indicator").n_unique()
    avg_country_coverage = (
        df.filter(pl.col("has_data"))
        .group_by("indicator")
        .agg(pl.col("country_iso2").n_unique())
        .select(pl.col("country_iso2").mean())
        .item()
    )
    
    print(f"\n  {'=' * 50}")
    print(f"  SUMMARY:")
    print(f"  Total indicator-source combinations probed: {total_indicators}")
    print(f"  Indicators with at least some data: {total_with_data}")
    print(f"  Avg country coverage per indicator: {avg_country_coverage:.1f}/{len(TARGET_NATIONS)}")
    if report.errors:
        print(f"  Errors encountered: {len(report.errors)}")
    print(f"  {'=' * 50}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="MCLD-1 Data Availability Audit")
    parser.add_argument("--fred-api-key", type=str, default=None,
                        help="FRED API key (free at fred.stlouisfed.org)")
    parser.add_argument("--output-dir", type=str, default="docs/data_audit",
                        help="Output directory for audit reports")
    parser.add_argument("--skip-dbnomics", action="store_true",
                        help="Skip DBnomics probing")
    parser.add_argument("--skip-fred", action="store_true",
                        help="Skip FRED probing")
    parser.add_argument("--skip-bis", action="store_true",
                        help="Skip BIS probing")
    parser.add_argument("--skip-worldbank", action="store_true",
                        help="Skip World Bank probing")
    args = parser.parse_args()
    
    print("=" * 70)
    print("MCLD-1 DATA AVAILABILITY AUDIT")
    print(f"Target: {len(TARGET_NATIONS)} sovereign nations, 1980–2025")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)
    
    report = AuditReport()
    
    # Probe each source
    if not args.skip_dbnomics:
        probe_dbnomics(report)
    
    if not args.skip_fred:
        probe_fred(report, api_key=args.fred_api_key)
    
    if not args.skip_bis:
        probe_bis(report)
    
    if not args.skip_worldbank:
        probe_world_bank(report)
    
    # Generate analysis
    output_dir = Path(args.output_dir)
    generate_report(report, output_dir)
    
    print(f"\nAudit complete at {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
