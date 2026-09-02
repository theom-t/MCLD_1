"""Stage 1: Bitemporal Data Ingestion.

Fetches macroeconomic data from FRED/ALFRED and DBnomics.
Crucially, it preserves causal integrity by computing `release_date`.
- For ALFRED: Uses true `realtime_start` vintage dates.
- For DBnomics: Computes `release_date = observation_date + SDDS_LAG`.
"""

import os
from datetime import timedelta
import yaml
import polars as pl
from fredapi import Fred
from dbnomics import fetch_series


def load_config(config_path: str = "configs/data.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class FredIngestor:
    def __init__(self, api_key: str):
        self.fred = Fred(api_key=api_key)

    def fetch_bitemporal(self, series_id: str, country: str, feature_id: str) -> pl.DataFrame:
        """Fetch ALFRED bitemporal vintages."""
        print(f"Fetching ALFRED: {series_id} ({feature_id} - {country})")
        try:
            # get_series_all_releases returns: date, realtime_start, value
            df_pandas = self.fred.get_series_all_releases(series_id)
            if df_pandas.empty:
                return pl.DataFrame()

            df = pl.from_pandas(df_pandas)
            
            # Format to our schema: [country, feature, obs_date, release_date, value]
            df = df.rename({
                "date": "observation_date",
                "realtime_start": "release_date"
            }).with_columns([
                pl.lit(country).alias("country"),
                pl.lit(feature_id).alias("feature"),
                pl.col("observation_date").cast(pl.Datetime),
                pl.col("release_date").cast(pl.Datetime),
                pl.col("value").cast(pl.Float64)
            ]).select([
                "country", "feature", "observation_date", "release_date", "value"
            ])
            return df
        except Exception as e:
            print(f"  [ERROR] FRED {series_id}: {e}")
            return pl.DataFrame()


class DBnomicsIngestor:
    def __init__(self, lag_config: dict):
        self.lag_config = lag_config

    def fetch_bitemporal(self, series_id: str, country: str, feature_id: str, sdds_category: str) -> pl.DataFrame:
        """Fetch DBnomics and apply SDDS lag for release date."""
        print(f"Fetching DBnomics: {series_id} ({feature_id} - {country})")
        try:
            # fetch_series returns: period, value
            df_pandas = fetch_series(series_id)
            if df_pandas is None or df_pandas.empty:
                return pl.DataFrame()

            df = pl.from_pandas(df_pandas)
            
            # Ensure period and value exist
            if "period" not in df.columns or "value" not in df.columns:
                return pl.DataFrame()

            # Apply SDDS Lag
            lag_days = self.lag_config.get(sdds_category, 30)  # Default 30 days
            
            df = df.rename({"period": "observation_date"}).with_columns([
                pl.lit(country).alias("country"),
                pl.lit(feature_id).alias("feature"),
                pl.col("observation_date").cast(pl.Datetime),
                pl.col("value").cast(pl.Float64, strict=False)
            ])
            
            # Compute conservative release_date
            df = df.with_columns(
                (pl.col("observation_date") + pl.duration(days=lag_days)).alias("release_date")
            )
            
            # Drop nulls (DBnomics sometimes returns NA rows for future dates)
            df = df.drop_nulls(subset=["value"])
            
            return df.select([
                "country", "feature", "observation_date", "release_date", "value"
            ])
        except Exception as e:
            print(f"  [ERROR] DBnomics {series_id}: {e}")
            return pl.DataFrame()


def run_ingestion(limit_features: list[str] = None):
    """Main ingestion loop based on config."""
    config = load_config()
    lag_config = config.get("sdds_lag_offsets_days", {})
    
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED_API_KEY environment variable not set.")
        
    fred_api = FredIngestor(api_key=api_key)
    dbn_api = DBnomicsIngestor(lag_config=lag_config)
    
    all_data = []
    
    for feature_id, feature_meta in config.get("features", {}).items():
        if limit_features and feature_id not in limit_features:
            continue
            
        sdds_category = feature_meta.get("sdds_category", "cpi")
        sources = feature_meta.get("sources", {})
        
        for country, source_meta in sources.items():
            src_type = source_meta.get("source")
            series_id = source_meta.get("series_id")
            
            if src_type == "FRED":
                df = fred_api.fetch_bitemporal(series_id, country, feature_id)
            elif src_type == "DBNOMICS":
                df = dbn_api.fetch_bitemporal(series_id, country, feature_id, sdds_category)
            else:
                continue
                
            if not df.is_empty():
                all_data.append(df)
                
    if all_data:
        master_panel = pl.concat(all_data)
        
        # Sort for cleanliness
        master_panel = master_panel.sort(["country", "feature", "observation_date", "release_date"])
        
        # Ensure directory exists
        os.makedirs("data/raw", exist_ok=True)
        master_panel.write_parquet("data/raw/bitemporal_panel.parquet")
        print(f"\nSuccessfully ingested {len(master_panel)} bitemporal observations.")
        print(f"Saved to data/raw/bitemporal_panel.parquet")
        
        # Display sample
        print("\nSample (Bitemporal view):")
        print(master_panel.head())
        return master_panel
    else:
        print("No data collected.")
        return None

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_ingestion()
