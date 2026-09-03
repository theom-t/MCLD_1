import polars as pl
import pandas as pd
import numpy as np

# Map countries to Dalio Archetypes
ARCHETYPES = {
    # Core Developed / Declining (Late Cycle)
    "US": "1_Core_Developed", "GB": "1_Core_Developed", "JP": "1_Core_Developed", 
    "DE": "1_Core_Developed", "FR": "1_Core_Developed", "IT": "1_Core_Developed", 
    "CA": "1_Core_Developed", "AU": "1_Core_Developed", "CH": "1_Core_Developed", 
    "NL": "1_Core_Developed", "SE": "1_Core_Developed", "ES": "1_Core_Developed",
    "BE": "1_Core_Developed", "DK": "1_Core_Developed", "FI": "1_Core_Developed",
    "IE": "1_Core_Developed", "AT": "1_Core_Developed", "NZ": "1_Core_Developed",
    "NO": "1_Core_Developed", "PT": "1_Core_Developed", "GR": "1_Core_Developed",
    "IL": "1_Core_Developed", "SG": "1_Core_Developed", "KR": "1_Core_Developed",
    "CZ": "1_Core_Developed", "PL": "1_Core_Developed", "HU": "1_Core_Developed",
    "RO": "1_Core_Developed",
    
    # Rising Challengers (Mid/Late Cycle)
    "CN": "2_Rising_Challengers", "IN": "2_Rising_Challengers", 
    "RU": "2_Rising_Challengers", "BR": "2_Rising_Challengers",
    
    # Emerging / Early Cycle
    "VN": "3_Emerging", "ID": "3_Emerging", "PH": "3_Emerging", 
    "NG": "3_Emerging", "PK": "3_Emerging", "EG": "3_Emerging",
    "TH": "3_Emerging", "MY": "3_Emerging", "TR": "3_Emerging",
    
    # Resource / Volatile (Boom/Bust)
    "SA": "4_Resource_Volatile", "AE": "4_Resource_Volatile", 
    "CL": "4_Resource_Volatile", "PE": "4_Resource_Volatile", 
    "ZA": "4_Resource_Volatile", "AR": "4_Resource_Volatile", 
    "MX": "4_Resource_Volatile", "CO": "4_Resource_Volatile"
}

def profile_dataset():
    df = pl.read_parquet("data/raw/bitemporal_panel.parquet")
    
    results = []
    countries = df["country"].unique().to_list()
    
    for country in countries:
        sub = df.filter(pl.col("country") == country)
        total_obs = sub.height
        
        # Calculate start year for each feature
        feature_starts = sub.group_by("feature").agg([
            pl.col("observation_date").min().alias("first_date"),
            pl.len().alias("count")
        ]).to_pandas()
        
        # Filter features with > 50 obs
        valid_features = feature_starts[feature_starts["count"] > 50]
        feature_count = len(valid_features)
        
        if feature_count > 0:
            median_start_year = int(valid_features["first_date"].dt.year.median())
            earliest_year = int(valid_features["first_date"].dt.year.min())
        else:
            median_start_year = 9999
            earliest_year = 9999
            
        results.append({
            "Country": country,
            "Archetype": ARCHETYPES.get(country, "Unknown"),
            "Total_Obs": total_obs,
            "Valid_Features_Count": feature_count,
            "Median_Start_Year": median_start_year,
            "Earliest_Year": earliest_year
        })
        
    res_df = pd.DataFrame(results)
    
    # Sort by Archetype, then by Feature Count (desc), then by Median Start Year (asc)
    res_df = res_df.sort_values(["Archetype", "Valid_Features_Count", "Median_Start_Year"], 
                                ascending=[True, False, True])
    
    res_df.to_csv("analysis/country_profiles.csv", index=False)
    
    print("# Dalio Dataset Profiling Report\n")
    for archetype, group in res_df.groupby("Archetype"):
        print(f"## Archetype: {archetype}")
        print(group[["Country", "Valid_Features_Count", "Median_Start_Year", "Earliest_Year", "Total_Obs"]].to_markdown(index=False))
        print("\n")
        
if __name__ == "__main__":
    profile_dataset()
