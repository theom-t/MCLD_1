import polars as pl
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
import os

def interpolate_panel():
    print("--- Running PCHIP Interpolation ---")
    input_path = "data/raw/bitemporal_panel_curated.parquet"
    out_tensor_path = "data/processed/monthly_interpolated.parquet"
    out_mask_path = "data/processed/mask_tensor.parquet"
    
    os.makedirs("data/processed", exist_ok=True)
    
    if not os.path.exists(input_path):
        print(f"[ERROR] {input_path} not found.")
        return
        
    df = pl.read_parquet(input_path).to_pandas()
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    
    countries = df["country"].unique()
    features = df["feature"].unique()
    
    # Create master monthly grid (end of month)
    # Most macro data is anchored to the 1st of the month in our DB
    # Let's align to the 1st of every month from 1950 to 2025
    master_dates = pd.date_range(start="1950-01-01", end="2025-01-01", freq="MS")
    master_grid = pd.DataFrame({"date": master_dates})
    
    interpolated_records = []
    mask_records = []
    
    print(f"Interpolating {len(countries)} countries x {len(features)} features...")
    
    for country in countries:
        c_df = df[df["country"] == country]
        
        for feature in features:
            f_df = c_df[c_df["feature"] == feature].sort_values("observation_date")
            
            if len(f_df) < 2:
                # Need at least 2 points for interpolation
                continue
                
            # Drop duplicates on observation_date and drop NaNs
            f_df = f_df.dropna(subset=["value"]).drop_duplicates(subset=["observation_date"], keep="last")
            
            if len(f_df) < 2:
                continue
            
            # Map observed dates to numerical for PCHIP (scale to days to prevent float64 overflow)
            # Convert to days since 1970 using pandas safely
            epoch = pd.Timestamp("1970-01-01")
            x_obs = (f_df["observation_date"] - epoch).dt.total_seconds().values / 86400.0
            y_obs = f_df["value"].values
            
            # Create PCHIP interpolator
            try:
                interpolator = PchipInterpolator(x_obs, y_obs)
            except Exception as e:
                print(f"Failed PCHIP for {country} {feature}: {e}")
                continue
                
            # Restrict grid to only interpolate between min and max observed dates
            min_date = f_df["observation_date"].min()
            max_date = f_df["observation_date"].max()
            
            valid_grid = master_grid[(master_grid["date"] >= min_date) & (master_grid["date"] <= max_date)]
            x_target = (valid_grid["date"] - epoch).dt.total_seconds().values / 86400.0
            
            if len(x_target) == 0:
                continue
                
            y_interp = interpolator(x_target)
            
            # Create mask: 1 if the target month had an exact observation (year & month match), else 0
            obs_year_months = set((d.year, d.month) for d in f_df["observation_date"])
            
            for t_date, t_val in zip(valid_grid["date"], y_interp):
                interpolated_records.append({
                    "date": t_date,
                    "country": country,
                    "feature": feature,
                    "value": t_val
                })
                
                # Mask logic
                is_real = 1 if (t_date.year, t_date.month) in obs_year_months else 0
                mask_records.append({
                    "date": t_date,
                    "country": country,
                    "feature": feature,
                    "mask_val": is_real
                })
                
    interp_df = pl.DataFrame(interpolated_records)
    mask_df = pl.DataFrame(mask_records)
    
    interp_df.write_parquet(out_tensor_path)
    mask_df.write_parquet(out_mask_path)
    
    print(f"Generated {interp_df.height:,} interpolated monthly points.")
    print(f"Saved to {out_tensor_path} and {out_mask_path}")

if __name__ == "__main__":
    interpolate_panel()
