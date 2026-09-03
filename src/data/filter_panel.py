import polars as pl
import os

def filter_panel():
    print("--- Running Dataset Curation (Filtering) ---")
    input_path = "data/raw/bitemporal_panel.parquet"
    output_path = "data/raw/bitemporal_panel_curated.parquet"
    
    if not os.path.exists(input_path):
        print(f"[ERROR] {input_path} not found.")
        return
        
    df = pl.read_parquet(input_path)
    total_initial = df.height
    countries_initial = df["country"].unique().to_list()
    print(f"Initial: {total_initial:,} rows, {len(countries_initial)} countries.")
    
    # Calculate valid features (>50 obs) per country
    feature_counts = df.group_by(["country", "feature"]).agg([
        pl.len().alias("count")
    ]).filter(pl.col("count") > 50).group_by("country").agg([
        pl.len().alias("valid_features")
    ])
    
    # Identify retained countries
    retained_df = feature_counts.filter(pl.col("valid_features") > 10)
    retained_countries = retained_df["country"].to_list()
    
    print(f"Retaining {len(retained_countries)} countries with > 10 valid features.")
    
    # Filter the master dataframe
    curated_df = df.filter(pl.col("country").is_in(retained_countries))
    
    curated_df.write_parquet(output_path)
    
    total_final = curated_df.height
    print(f"Final: {total_final:,} rows, {len(retained_countries)} countries.")
    print(f"Dropped {total_initial - total_final:,} noisy rows.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    filter_panel()
