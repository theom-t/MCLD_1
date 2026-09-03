#!/bin/bash
cd "$(dirname "$0")/.."
uv run streamlit run analysis/dashboard_raw_data.py
