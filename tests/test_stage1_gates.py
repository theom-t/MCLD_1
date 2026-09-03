import polars as pl
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def validate_stage1():
    print("--- Stage 1 Validation Gate Audit ---\n")
    
    try:
        df = pl.read_parquet("data/processed/stationary_tensor.parquet").to_pandas()
    except Exception as e:
        print(f"Failed to load stationary tensor: {e}")
        return
        
    total_series = 0
    stationary_count = 0
    correlations = []
    
    countries = df["country"].unique()
    features = df["feature"].unique()
    
    print("Evaluating ADF Stationarity & Memory Retention...")
    
    for country in countries:
        for feature in features:
            sub = df[(df["country"] == country) & (df["feature"] == feature)]
            if len(sub) < 30:
                continue
                
            raw_vals = sub["value"].astype(float).values
            stat_vals = sub["value_stationary"].astype(float).values
            
            is_log = sub["is_log_transformed"].iloc[0] if "is_log_transformed" in sub.columns else False
            if is_log:
                min_val = np.nanmin(raw_vals)
                if min_val <= 0:
                    raw_vals = raw_vals - min_val + 1e-6
                raw_vals = np.log(raw_vals)
            
            # Find common non-nan mask
            mask = ~np.isnan(raw_vals) & ~np.isnan(stat_vals)
            if mask.sum() < 30:
                continue
                
            total_series += 1
            
            # 1. ADF Stationarity Test
            try:
                adf_res = adfuller(stat_vals[mask])
                if adf_res[1] < 0.01:
                    stationary_count += 1
            except:
                pass
                
            # 2. Memory Retention (Correlation)
            try:
                corr = np.corrcoef(raw_vals[mask], stat_vals[mask])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
            except:
                pass
                
    # Metrics
    stationarity_pct = (stationary_count / total_series) * 100 if total_series > 0 else 0
    avg_corr = np.mean(correlations) if correlations else 0
    
    print(f"Total Series Evaluated: {total_series}")
    
    # Gate 1: ADF
    print(f"\n[Gate 1] ADF Stationarity (Target: >= 98.0%)")
    print(f"Result: {stationarity_pct:.2f}% (Passed: {stationary_count}/{total_series})")
    if stationarity_pct >= 98.0:
        print("✅ PASSED")
    else:
        print("❌ FAILED")
        
    # Gate 2: Memory Retention
    print(f"\n[Gate 2] Memory Retention Correlation (Target: >= 0.70)")
    print(f"Result: {avg_corr:.3f}")
    if avg_corr >= 0.70:
        print("✅ PASSED")
    else:
        print("❌ FAILED")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    validate_stage1()
