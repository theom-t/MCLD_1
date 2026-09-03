"""Stage 1: Multi-Source Bitemporal Data Ingestion Engine.

Ingests data from:
1. ALFRED (St. Louis Fed) - Real-time vintage timestamps.
2. DBnomics (BIS policy rates) - Conservative SDDS lag offsets.
3. BIS SDMX API - Credit-to-GDP Gap direct bulk CSV.
4. Economic Policy Uncertainty (EPU) - Baker, Bloom, Davis (monthly).
5. Geopolitical Risk (GPR) - Caldara & Iacoviello (monthly).

Features:
- Incremental updates: Skips existing (country, feature) pairs in Parquet.
- Causal integrity: All rows strictly have observation_date and release_date.
"""

import io
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import polars as pl
import requests
import yaml
from dbnomics import fetch_series
from fredapi import Fred

# Country ISO mappings
NATIONS = {
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
ISO3_TO_ISO2 = {v: k for k, v in NATIONS.items()}

PARQUET_PATH = Path("data/raw/bitemporal_panel.parquet")


def get_existing_pairs() -> set[tuple[str, str]]:
    """Return set of (country, feature) tuples already saved."""
    if not PARQUET_PATH.exists():
        return set()
    try:
        df = pl.read_parquet(PARQUET_PATH)
        pairs = df.select(["country", "feature"]).unique()
        return set(zip(pairs["country"].to_list(), pairs["feature"].to_list()))
    except Exception as e:
        print(f"Warning reading existing parquet: {e}")
        return set()


# ---------------------------------------------------------------------------
# Specialized Fetcher 1: BIS Credit-to-GDP Gap
# ---------------------------------------------------------------------------
def ingest_bis_credit_gap(existing_pairs: set[tuple[str, str]]) -> list[pl.DataFrame]:
    """Fetch BIS Credit-to-GDP Gap via direct SDMX CSV endpoint."""
    print("\n--- Ingesting BIS Credit-to-GDP Gap ---")
    url = "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CREDIT_GAP/1.0?format=csv"
    dfs = []
    try:
        r = requests.get(url, headers={"Accept": "text/csv"}, timeout=30)
        if r.status_code != 200:
            print(f"  [ERROR] BIS returned status {r.status_code}")
            return dfs

        df_raw = pd.read_csv(io.StringIO(r.text))
        # Filter for quarterly credit gap (CG_DTYPE == 'C' or 'A')
        # TC_BORROWERS == 'P' (Private non-financial sector)
        df_p = df_raw[(df_raw["TC_BORROWERS"] == "P") & (df_raw["CG_DTYPE"] == "A")].copy()

        # Parse quarter to date (e.g. 1964-Q4 -> 1964-10-01)
        def q_to_date(q_str):
            try:
                y, q = q_str.split("-Q")
                m = (int(q) - 1) * 3 + 1
                return datetime(int(y), m, 1)
            except Exception:
                return None

        df_p["obs_date"] = df_p["TIME_PERIOD"].apply(q_to_date)
        df_p = df_p.dropna(subset=["obs_date", "OBS_VALUE"])

        for iso2 in NATIONS.keys():
            if (iso2, "credit_to_gdp_gap") in existing_pairs:
                continue

            sub = df_p[df_p["BORROWERS_CTY"] == iso2]
            if sub.empty:
                continue

            print(f"  BIS Credit Gap: {iso2} ({len(sub)} obs)")
            # Credit-to-GDP gap is quarterly, published with ~90 days lag
            pldf = pl.DataFrame({
                "country": [iso2] * len(sub),
                "feature": ["credit_to_gdp_gap"] * len(sub),
                "observation_date": sub["obs_date"].tolist(),
                "release_date": [d + timedelta(days=90) for d in sub["obs_date"].tolist()],
                "value": sub["OBS_VALUE"].astype(float).tolist(),
            }).with_columns([
                pl.col("observation_date").cast(pl.Datetime),
                pl.col("release_date").cast(pl.Datetime),
                pl.col("value").cast(pl.Float64),
            ])
            dfs.append(pldf)

    except Exception as e:
        print(f"  [ERROR] BIS Credit Gap: {e}")
    return dfs


# ---------------------------------------------------------------------------
# Specialized Fetcher 2: Geopolitical Risk (GPR) Index
# ---------------------------------------------------------------------------
def ingest_gpr(existing_pairs: set[tuple[str, str]]) -> list[pl.DataFrame]:
    """Fetch Caldara & Iacoviello GPR country indices."""
    print("\n--- Ingesting Geopolitical Risk (GPR) Country Indices ---")
    url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
    dfs = []
    try:
        df_raw = pd.read_excel(url)
        # Dates are in 'month' column (can be Timestamp, int, or string)
        def parse_gpr_date(m):
            if isinstance(m, (datetime, pd.Timestamp)):
                return datetime(m.year, m.month, 1)
            try:
                s = str(int(m))
                return datetime(int(s[:4]), int(s[4:6]), 1)
            except Exception:
                return None

        df_raw["obs_date"] = df_raw["month"].apply(parse_gpr_date)
        df_raw = df_raw.dropna(subset=["obs_date"])

        for col in df_raw.columns:
            if not col.startswith("GPRC_"):
                continue
            iso3 = col.replace("GPRC_", "")
            iso2 = ISO3_TO_ISO2.get(iso3)
            if not iso2:
                continue

            if (iso2, "gpr_index") in existing_pairs:
                continue

            sub = df_raw.dropna(subset=[col])
            if sub.empty:
                continue

            print(f"  GPR Index: {iso2} ({len(sub)} obs)")
            # GPR is monthly, published at start of following month (+3 days lag)
            pldf = pl.DataFrame({
                "country": [iso2] * len(sub),
                "feature": ["gpr_index"] * len(sub),
                "observation_date": sub["obs_date"].tolist(),
                "release_date": [d + timedelta(days=3) for d in sub["obs_date"].tolist()],
                "value": sub[col].astype(float).tolist(),
            }).with_columns([
                pl.col("observation_date").cast(pl.Datetime),
                pl.col("release_date").cast(pl.Datetime),
                pl.col("value").cast(pl.Float64),
            ])
            dfs.append(pldf)

    except Exception as e:
        print(f"  [ERROR] GPR Ingestion: {e}")
    return dfs


# ---------------------------------------------------------------------------
# Specialized Fetcher 3: Economic Policy Uncertainty (EPU) Index
# ---------------------------------------------------------------------------
def ingest_epu(existing_pairs: set[tuple[str, str]]) -> list[pl.DataFrame]:
    """Fetch Baker, Bloom, Davis All Country EPU data."""
    print("\n--- Ingesting Economic Policy Uncertainty (EPU) ---")
    url = "https://www.policyuncertainty.com/media/All_Country_Data.xlsx"
    dfs = []
    try:
        df_raw = pd.read_excel(url)
        # Columns include: Year, Month, Australia, Brazil, Canada, China, etc.
        country_name_map = {
            "Australia": "AU", "Brazil": "BR", "Canada": "CA", "Chile": "CL",
            "China": "CN", "France": "FR", "Germany": "DE", "Greece": "GR",
            "India": "IN", "Ireland": "IE", "Italy": "IT", "Japan": "JP",
            "Korea": "KR", "Mexico": "MX", "Pakistan": "PK", "Russia": "RU",
            "Spain": "ES", "Singapore": "SG", "UK": "GB", "US": "US",
            "Mainland China": "CN",
        }

        # Build obs_date
        def make_date(row):
            try:
                return datetime(int(row["Year"]), int(row["Month"]), 1)
            except Exception:
                return None

        df_raw["obs_date"] = df_raw.apply(make_date, axis=1)
        df_raw = df_raw.dropna(subset=["obs_date"])

        for col_name, iso2 in country_name_map.items():
            if col_name not in df_raw.columns:
                continue
            if (iso2, "epu_index") in existing_pairs:
                continue

            sub = df_raw[["obs_date", col_name]].dropna()
            # In case multiple columns map to same iso2 (like Mainland China vs China)
            if any(df["country"][0] == iso2 for df in dfs if not df.is_empty()):
                continue

            print(f"  EPU Index: {iso2} ({len(sub)} obs)")
            # Monthly index released first week of following month (+7 days)
            pldf = pl.DataFrame({
                "country": [iso2] * len(sub),
                "feature": ["epu_index"] * len(sub),
                "observation_date": sub["obs_date"].tolist(),
                "release_date": [d + timedelta(days=7) for d in sub["obs_date"].tolist()],
                "value": pd.to_numeric(sub[col_name], errors="coerce").fillna(0.0).tolist(),
            }).with_columns([
                pl.col("observation_date").cast(pl.Datetime),
                pl.col("release_date").cast(pl.Datetime),
                pl.col("value").cast(pl.Float64),
            ])
            dfs.append(pldf)

    except Exception as e:
        print(f"  [ERROR] EPU Ingestion: {e}")
    return dfs


# ---------------------------------------------------------------------------
# Specialized Fetcher 4: World Bank WDI (Dalio Core Indicators)
# ---------------------------------------------------------------------------
def ingest_dalio_wdi(existing_pairs: set[tuple[str, str]]) -> list[pl.DataFrame]:
    """Fetch additional Dalio WDI Indicators."""
    print("\n--- Ingesting Dalio WDI Indicators ---")
    indicators = {
        "gov_debt_gdp": "GC.DOD.TOTL.GD.ZS",
        "current_account_gdp": "BN.CAB.XOKA.GD.ZS",
        "rnd_gdp": "GB.XPD.RSDV.GD.ZS",
        "working_age_pop": "SP.POP.1564.TO.ZS",
        "market_cap_gdp": "CM.MKT.LCAP.GD.ZS",
    }
    dfs = []
    
    for feature_name, ind_code in indicators.items():
        url = f"http://api.worldbank.org/v2/country/all/indicator/{ind_code}?format=json&per_page=10000"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                continue
            data = r.json()
            if len(data) == 2:
                records = data[1]
                df_raw = pd.DataFrame(records)
                df_raw["country_iso2"] = df_raw["countryiso3code"].map(ISO3_TO_ISO2)
                df_raw = df_raw.dropna(subset=["country_iso2", "value"])

                for iso2, group in df_raw.groupby("country_iso2"):
                    if (iso2, feature_name) in existing_pairs:
                        continue

                    print(f"  WDI {feature_name}: {iso2} ({len(group)} obs)")
                    pldf = pl.DataFrame({
                        "country": [iso2] * len(group),
                        "feature": [feature_name] * len(group),
                        "observation_date": [datetime(int(y), 12, 31) for y in group["date"]],
                        "release_date": [datetime(int(y) + 1, 6, 30) for y in group["date"]],
                        "value": group["value"].astype(float).tolist(),
                    }).with_columns([
                        pl.col("observation_date").cast(pl.Datetime),
                        pl.col("release_date").cast(pl.Datetime),
                        pl.col("value").cast(pl.Float64),
                    ]).sort("observation_date")
                    dfs.append(pldf)
        except Exception as e:
            print(f"  [ERROR] WDI {feature_name} Ingestion: {e}")
            
    return dfs

# ---------------------------------------------------------------------------
# Specialized Fetcher 5: SWIID Gini Index
# ---------------------------------------------------------------------------
def ingest_swiid(existing_pairs: set[tuple[str, str]]) -> list[pl.DataFrame]:
    """Fetch SWIID Gini Index."""
    print("\n--- Ingesting SWIID Gini Index ---")
    url = "https://raw.githubusercontent.com/fsolt/swiid/master/data/swiid_summary.csv"
    dfs = []
    
    swiid_map = {
        "China": "CN", "South Korea": "KR", "Japan": "JP", "Turkey": "TR", "Russia": "RU",
        "Brazil": "BR", "Argentina": "AR", "South Africa": "ZA", "United States": "US", 
        "United Kingdom": "GB", "Germany": "DE", "France": "FR", "Italy": "IT", "Spain": "ES", 
        "Netherlands": "NL", "Switzerland": "CH", "Australia": "AU", "Canada": "CA", 
        "India": "IN", "Vietnam": "VN", "Indonesia": "ID", "Poland": "PL", 
        "Czech Republic": "CZ", "Mexico": "MX", "Thailand": "TH", "Malaysia": "MY", 
        "Philippines": "PH", "Chile": "CL", "Greece": "GR", "Portugal": "PT", 
        "Nigeria": "NG", "Egypt": "EG", "Pakistan": "PK", "Colombia": "CO", "Peru": "PE", 
        "Hungary": "HU", "Romania": "RO", "Israel": "IL", "Saudi Arabia": "SA", 
        "United Arab Emirates": "AE", "Singapore": "SG", "Norway": "NO", "Sweden": "SE", 
        "Denmark": "DK", "New Zealand": "NZ", "Finland": "FI", "Ireland": "IE", "Belgium": "BE"
    }

    try:
        df_raw = pd.read_csv(url)
        df_raw["iso2"] = df_raw["country"].map(swiid_map)
        df_raw = df_raw.dropna(subset=["iso2", "gini_disp"])

        for iso2, group in df_raw.groupby("iso2"):
            if (iso2, "gini_disp") in existing_pairs:
                continue

            print(f"  SWIID Gini: {iso2} ({len(group)} obs)")
            pldf = pl.DataFrame({
                "country": [iso2] * len(group),
                "feature": ["gini_disp"] * len(group),
                "observation_date": [datetime(int(y), 12, 31) for y in group["year"]],
                "release_date": [datetime(int(y) + 1, 6, 30) for y in group["year"]],
                "value": group["gini_disp"].astype(float).tolist(),
            }).with_columns([
                pl.col("observation_date").cast(pl.Datetime),
                pl.col("release_date").cast(pl.Datetime),
                pl.col("value").cast(pl.Float64),
            ]).sort("observation_date")
            dfs.append(pldf)
    except Exception as e:
        print(f"  [ERROR] SWIID Ingestion: {e}")
    return dfs


# ---------------------------------------------------------------------------
# Specialized Fetcher 6: SIPRI Military Expenditure
# ---------------------------------------------------------------------------
def ingest_sipri(existing_pairs: set[tuple[str, str]]) -> list[pl.DataFrame]:
    """Fetch SIPRI Military Expenditure (Share of GDP)."""
    print("\n--- Ingesting SIPRI Military Expenditure ---")
    url = "https://www.sipri.org/sites/default/files/SIPRI-Milex-data-1949-2025_v1.2.xlsx"
    dfs = []
    
    sipri_map = {
        "China": "CN", "Korea, South": "KR", "Japan": "JP", "Türkiye": "TR", "Russia": "RU", "USSR/Russia": "RU",
        "Brazil": "BR", "Argentina": "AR", "South Africa": "ZA", "United States of America": "US", 
        "UK": "GB", "Germany": "DE", "France": "FR", "Italy": "IT", "Spain": "ES", 
        "Netherlands": "NL", "Switzerland": "CH", "Australia": "AU", "Canada": "CA", 
        "India": "IN", "Viet Nam": "VN", "Indonesia": "ID", "Poland": "PL", 
        "Czechia": "CZ", "Czechoslovakia/Czechia": "CZ", "Mexico": "MX", "Thailand": "TH", "Malaysia": "MY", 
        "Philippines": "PH", "Chile": "CL", "Greece": "GR", "Portugal": "PT", 
        "Nigeria": "NG", "Egypt": "EG", "Pakistan": "PK", "Colombia": "CO", "Peru": "PE", 
        "Hungary": "HU", "Romania": "RO", "Israel": "IL", "Saudi Arabia": "SA", 
        "United Arab Emirates": "AE", "Singapore": "SG", "Norway": "NO", "Sweden": "SE", 
        "Denmark": "DK", "New Zealand": "NZ", "Finland": "FI", "Ireland": "IE", "Belgium": "BE"
    }

    try:
        df_raw = pd.read_excel(url, sheet_name="Share of GDP", skiprows=5)
        df_raw["iso2"] = df_raw["Country"].map(sipri_map)
        df_raw = df_raw.dropna(subset=["iso2"])

        years = [c for c in df_raw.columns if isinstance(c, (int, str)) and str(c).isdigit()]
        
        for iso2, group in df_raw.groupby("iso2"):
            if (iso2, "milex_gdp") in existing_pairs:
                continue
                
            row = group.iloc[0]
            valid_years = []
            valid_vals = []
            for y in years:
                val = row[y]
                if pd.notna(val) and isinstance(val, (int, float)) and val not in ["xxx", "...", ". ."]:
                    valid_years.append(int(y))
                    valid_vals.append(float(val))
                    
            if not valid_years:
                continue

            print(f"  SIPRI Milex: {iso2} ({len(valid_years)} obs)")
            pldf = pl.DataFrame({
                "country": [iso2] * len(valid_years),
                "feature": ["milex_gdp"] * len(valid_years),
                "observation_date": [datetime(int(y), 12, 31) for y in valid_years],
                "release_date": [datetime(int(y) + 1, 4, 30) for y in valid_years],
                "value": valid_vals,
            }).with_columns([
                pl.col("observation_date").cast(pl.Datetime),
                pl.col("release_date").cast(pl.Datetime),
                pl.col("value").cast(pl.Float64),
            ]).sort("observation_date")
            dfs.append(pldf)
    except Exception as e:
        print(f"  [ERROR] SIPRI Ingestion: {e}")
    return dfs


# ---------------------------------------------------------------------------
# Specialized Fetcher 7: IMF IFS via DBnomics (FX Reserves)
# ---------------------------------------------------------------------------
def ingest_imf(existing_pairs: set[tuple[str, str]]) -> list[pl.DataFrame]:
    """Fetch IMF IFS Total Reserves (excluding gold) via DBnomics."""
    print("\n--- Ingesting IMF IFS (FX Reserves) ---")
    dfs = []
    try:
        for iso2 in NATIONS.keys():
            if (iso2, "fx_reserves") in existing_pairs:
                continue
                
            series_id = f"IMF/IFS/M.{iso2}.RAXG_USD"
            try:
                df_pandas = fetch_series(series_id)
                if df_pandas is not None and not df_pandas.empty:
                    df_pandas = df_pandas.dropna(subset=["value"])
                    if df_pandas.empty:
                        continue
                        
                    print(f"  IMF FX Reserves: {iso2} ({len(df_pandas)} obs)")
                    pldf = pl.from_pandas(df_pandas).rename({
                        "period": "observation_date"
                    }).with_columns([
                        pl.lit(iso2).alias("country"),
                        pl.lit("fx_reserves").alias("feature"),
                        pl.col("observation_date").cast(pl.Datetime),
                        pl.col("value").cast(pl.Float64, strict=False),
                    ]).with_columns(
                        (pl.col("observation_date") + pl.duration(days=30)).alias("release_date")
                    ).drop_nulls(subset=["value"]).select([
                        "country", "feature", "observation_date", "release_date", "value"
                    ])
                    dfs.append(pldf)
            except Exception as e:
                pass
    except Exception as e:
        print(f"  [ERROR] IMF Ingestion: {e}")
    return dfs


# ---------------------------------------------------------------------------
# Specialized Fetcher 8: BIS REER via DBnomics
# ---------------------------------------------------------------------------
def ingest_bis_reer(existing_pairs: set[tuple[str, str]]) -> list[pl.DataFrame]:
    """Fetch BIS Real Effective Exchange Rate (Broad) via DBnomics."""
    print("\n--- Ingesting BIS REER (Broad) ---")
    dfs = []
    try:
        for iso2 in NATIONS.keys():
            if (iso2, "reer") in existing_pairs:
                continue
                
            series_id = f"BIS/WS_EER/M.R.B.{iso2}"
            try:
                df_pandas = fetch_series(series_id)
                if df_pandas is not None and not df_pandas.empty:
                    df_pandas = df_pandas.dropna(subset=["value"])
                    if df_pandas.empty:
                        continue
                        
                    print(f"  BIS REER: {iso2} ({len(df_pandas)} obs)")
                    pldf = pl.from_pandas(df_pandas).rename({
                        "period": "observation_date"
                    }).with_columns([
                        pl.lit(iso2).alias("country"),
                        pl.lit("reer").alias("feature"),
                        pl.col("observation_date").cast(pl.Datetime),
                        pl.col("value").cast(pl.Float64, strict=False),
                    ]).with_columns(
                        (pl.col("observation_date") + pl.duration(days=15)).alias("release_date")
                    ).drop_nulls(subset=["value"]).select([
                        "country", "feature", "observation_date", "release_date", "value"
                    ])
                    dfs.append(pldf)
            except Exception as e:
                pass
    except Exception as e:
        print(f"  [ERROR] BIS REER Ingestion: {e}")
    return dfs


# ---------------------------------------------------------------------------
# Specialized Fetcher 9: IMF Gold Reserves via DBnomics
# ---------------------------------------------------------------------------
def ingest_imf_gold(existing_pairs: set[tuple[str, str]]) -> list[pl.DataFrame]:
    """Fetch IMF Gold Reserves (Volume) via DBnomics."""
    print("\n--- Ingesting IMF Gold Reserves ---")
    dfs = []
    try:
        for iso2 in NATIONS.keys():
            if (iso2, "gold_reserves") in existing_pairs:
                continue
                
            # Gold volume in million fine troy ounces
            series_id = f"IMF/IFS/M.{iso2}.RAXGF_OZT"
            try:
                df_pandas = fetch_series(series_id)
                if df_pandas is not None and not df_pandas.empty:
                    df_pandas = df_pandas.dropna(subset=["value"])
                    if df_pandas.empty:
                        continue
                        
                    print(f"  IMF Gold Reserves: {iso2} ({len(df_pandas)} obs)")
                    pldf = pl.from_pandas(df_pandas).rename({
                        "period": "observation_date"
                    }).with_columns([
                        pl.lit(iso2).alias("country"),
                        pl.lit("gold_reserves").alias("feature"),
                        pl.col("observation_date").cast(pl.Datetime),
                        pl.col("value").cast(pl.Float64, strict=False),
                    ]).with_columns(
                        (pl.col("observation_date") + pl.duration(days=30)).alias("release_date")
                    ).drop_nulls(subset=["value"]).select([
                        "country", "feature", "observation_date", "release_date", "value"
                    ])
                    dfs.append(pldf)
            except Exception as e:
                pass
    except Exception as e:
        print(f"  [ERROR] IMF Gold Ingestion: {e}")
    return dfs


# ---------------------------------------------------------------------------
# Standard Config-Based Ingestion (ALFRED & DBnomics with Caching)
# ---------------------------------------------------------------------------
def ingest_from_config(existing_pairs: set[tuple[str, str]]) -> list[pl.DataFrame]:
    """Ingests series specified in configs/data.yaml, skipping existing pairs."""
    print("\n--- Ingesting from configs/data.yaml (ALFRED & DBnomics) ---")
    config_path = "configs/data.yaml"
    if not os.path.exists(config_path):
        return []

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    lag_config = config.get("sdds_lag_offsets_days", {})
    api_key = os.getenv("FRED_API_KEY")
    fred = Fred(api_key=api_key) if api_key else None

    dfs = []

    for feature_id, feature_meta in config.get("features", {}).items():
        sdds_category = feature_meta.get("sdds_category", "cpi")
        sources = feature_meta.get("sources", {})

        for country, source_meta in sources.items():
            # Check skip
            if (country, feature_id) in existing_pairs:
                continue

            src_type = source_meta.get("source")
            series_id = source_meta.get("series_id")

            if src_type == "FRED" and fred:
                print(f"Fetching ALFRED: {series_id} ({feature_id} - {country})")
                time.sleep(0.6)
                try:
                    df_pandas = fred.get_series_all_releases(series_id)
                    if not df_pandas.empty:
                        pldf = pl.from_pandas(df_pandas).rename({
                            "date": "observation_date",
                            "realtime_start": "release_date",
                        }).with_columns([
                            pl.lit(country).alias("country"),
                            pl.lit(feature_id).alias("feature"),
                            pl.col("observation_date").cast(pl.Datetime),
                            pl.col("release_date").cast(pl.Datetime),
                            pl.col("value").cast(pl.Float64),
                        ]).select([
                            "country", "feature", "observation_date", "release_date", "value"
                        ])
                        dfs.append(pldf)
                except Exception as e:
                    if "The series does not exist in ALFRED" in str(e) or "Bad Request" in str(e):
                        print(f"  [INFO] ALFRED vintage missing. Falling back to standard FRED + SDDS Lag for {series_id}")
                        time.sleep(0.6)
                        try:
                            df_pandas = fred.get_series(series_id)
                            if not df_pandas.empty:
                                lag_days = lag_config.get(sdds_category, 30)
                                pldf = pl.DataFrame({
                                    "observation_date": df_pandas.index,
                                    "value": df_pandas.values
                                }).with_columns([
                                    pl.lit(country).alias("country"),
                                    pl.lit(feature_id).alias("feature"),
                                    pl.col("observation_date").cast(pl.Datetime),
                                    pl.col("value").cast(pl.Float64, strict=False),
                                ]).with_columns(
                                    (pl.col("observation_date") + pl.duration(days=lag_days)).alias("release_date")
                                ).drop_nulls(subset=["value"]).select([
                                    "country", "feature", "observation_date", "release_date", "value"
                                ])
                                dfs.append(pldf)
                        except Exception as inner_e:
                            print(f"  [ERROR] FRED fallback {series_id}: {inner_e}")
                    else:
                        print(f"  [ERROR] ALFRED {series_id}: {e}")

            elif src_type == "DBNOMICS":
                print(f"Fetching DBnomics: {series_id} ({feature_id} - {country})")
                try:
                    df_pandas = fetch_series(series_id)
                    if df_pandas is not None and not df_pandas.empty:
                        lag_days = lag_config.get(sdds_category, 30)
                        pldf = pl.from_pandas(df_pandas).rename({
                            "period": "observation_date"
                        }).with_columns([
                            pl.lit(country).alias("country"),
                            pl.lit(feature_id).alias("feature"),
                            pl.col("observation_date").cast(pl.Datetime),
                            pl.col("value").cast(pl.Float64, strict=False),
                        ]).with_columns(
                            (pl.col("observation_date") + pl.duration(days=lag_days)).alias("release_date")
                        ).drop_nulls(subset=["value"]).select([
                            "country", "feature", "observation_date", "release_date", "value"
                        ])
                        dfs.append(pldf)
                except Exception as e:
                    print(f"  [ERROR] DBnomics {series_id}: {e}")

    return dfs


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def run_full_pipeline():
    existing_pairs = get_existing_pairs()
    print(f"Existing populated (country, feature) pairs: {len(existing_pairs)}")

    all_new_dfs = []

    # 1. Specialized datasets
    all_new_dfs.extend(ingest_bis_credit_gap(existing_pairs))
    all_new_dfs.extend(ingest_gpr(existing_pairs))
    all_new_dfs.extend(ingest_epu(existing_pairs))
    all_new_dfs.extend(ingest_dalio_wdi(existing_pairs))
    all_new_dfs.extend(ingest_swiid(existing_pairs))
    all_new_dfs.extend(ingest_sipri(existing_pairs))
    all_new_dfs.extend(ingest_imf(existing_pairs))
    all_new_dfs.extend(ingest_bis_reer(existing_pairs))
    all_new_dfs.extend(ingest_imf_gold(existing_pairs))

    # 2. Config-based ALFRED & DBnomics
    all_new_dfs.extend(ingest_from_config(existing_pairs))

    if all_new_dfs:
        new_data = pl.concat(all_new_dfs)
        if PARQUET_PATH.exists():
            existing_df = pl.read_parquet(PARQUET_PATH)
            master_df = pl.concat([existing_df, new_data])
        else:
            master_df = new_data

        master_df = (
            master_df
            .unique(subset=["country", "feature", "observation_date", "release_date"])
            .sort(["country", "feature", "observation_date", "release_date"])
        )

        PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
        master_df.write_parquet(PARQUET_PATH)
        print(f"\nPipeline Complete. Master panel now holds {len(master_df):,} bitemporal rows.")
    else:
        print("\nNo new data to append; all series already populated.")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_full_pipeline()
