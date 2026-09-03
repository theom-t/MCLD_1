import streamlit as st
import polars as pl
import plotly.express as px
import pandas as pd
import os

st.set_page_config(page_title="MCLD-1 Raw Data Dashboard", layout="wide")

@st.cache_data
def load_data():
    # Handle being run from root or from inside analysis/
    path = "data/raw/bitemporal_panel.parquet"
    if not os.path.exists(path):
        path = "../data/raw/bitemporal_panel.parquet"
        
    if not os.path.exists(path):
        return None
    return pl.read_parquet(path)

st.title("🌍 MCLD-1: Bitemporal Raw Data Dashboard")
st.markdown("Explore the raw macro-cycle data ingested from various sources before Stage 1 transformations (interpolation and fractional differencing).")

df = load_data()

if df is None:
    st.error("Parquet file not found. Please run the ingestion pipeline first.")
    st.stop()

# Basic Stats
total_obs = df.height
countries = df["country"].unique().to_list()
features = df["feature"].unique().to_list()
min_date = df["observation_date"].min()
max_date = df["observation_date"].max()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Observations", f"{total_obs:,}")
col2.metric("Distinct Countries", len(countries))
col3.metric("Distinct Features", len(features))
col4.metric("Date Range", f"{min_date.year} - {max_date.year}")

st.divider()

# Data Availability Heatmap
st.subheader("Data Coverage Matrix")
coverage = df.group_by(["country", "feature"]).agg(pl.len().alias("count")).to_pandas()
coverage_pivot = coverage.pivot(index="country", columns="feature", values="count").fillna(0)
fig_heat = px.imshow(coverage_pivot, text_auto=False, aspect="auto", 
                     color_continuous_scale="Viridis",
                     title="Observation Count by Country and Feature")
st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# Time Series Explorer
st.subheader("Time Series Explorer")
c_col, f_col = st.columns(2)
selected_country = c_col.selectbox("Select Country", sorted(countries), index=sorted(countries).index("US") if "US" in countries else 0)
selected_feature = f_col.selectbox("Select Feature", sorted(features))

# Filter data
filtered_df = df.filter(
    (pl.col("country") == selected_country) & 
    (pl.col("feature") == selected_feature)
).to_pandas()

if not filtered_df.empty:
    fig_ts = px.line(filtered_df, x="observation_date", y="value", 
                     title=f"{selected_feature} for {selected_country}",
                     markers=True)
    st.plotly_chart(fig_ts, use_container_width=True)
    
    # Show release lag
    filtered_df['lag_days'] = (filtered_df['release_date'] - filtered_df['observation_date']).dt.days
    fig_lag = px.histogram(filtered_df, x="lag_days", nbins=30, 
                           title=f"Publication Lag Distribution (Days) for {selected_feature} ({selected_country})")
    st.plotly_chart(fig_lag, use_container_width=True)
    
    with st.expander("View Raw Data Snippet"):
        st.dataframe(filtered_df.tail(10))
else:
    st.warning("No data found for this combination.")
