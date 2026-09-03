"""Generates a massive data.yaml config by applying known FRED/DBnomics naming conventions."""

import yaml
from pathlib import Path

# The 48 Target Nations
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

# The Euro Area mapping for ECB policy rates in BIS
EURO_AREA = {"DE", "FR", "IT", "ES", "NL", "GR", "PT", "IE", "BE", "FI", "AT"}

def generate_config():
    config = {
        "sdds_lag_offsets_days": {
            "policy_rate": 1, "m2_money": 15, "cpi": 30, "unemployment": 30,
            "industrial_production": 45, "trade_balance": 60, "fx_reserves": 30,
            "10y_yield": 1
        },
        "nations": list(NATIONS.keys()),
        "features": {}
    }

    # Helper to add feature
    def add_feature(feature_id, domain, sdds_cat, transform, source_logic):
        sources = {}
        for iso2, iso3 in NATIONS.items():
            src = source_logic(iso2, iso3)
            if src:
                sources[iso2] = src
        config["features"][feature_id] = {
            "domain": domain,
            "sdds_category": sdds_cat,
            "transform": transform,
            "sources": sources
        }

    # 1. CPI (OECD MEI standard in FRED)
    add_feature("cpi_yoy", "Monetary & Debt", "cpi", "pct_change_12m",
                lambda i2, i3: {"source": "FRED", "series_id": f"{i3}CPIALLMINMEI"} if i3 != "USA" else {"source": "FRED", "series_id": "CPIAUCSL"})

    # 2. Industrial Production (OECD MEI standard)
    add_feature("industrial_production", "Internal Order", "industrial_production", "log_diff",
                lambda i2, i3: {"source": "FRED", "series_id": f"{i3}PROINDMISMEI"} if i3 != "USA" else {"source": "FRED", "series_id": "INDPRO"})

    # 3. Unemployment (OECD standard)
    add_feature("unemployment", "Internal Order", "unemployment", "level",
                lambda i2, i3: {"source": "FRED", "series_id": f"LMUNRRTT{i2}M156S"} if i3 != "USA" else {"source": "FRED", "series_id": "UNRATE"})

    # 4. Central Bank Policy Rate (BIS)
    add_feature("policy_rate", "Monetary & Debt", "policy_rate", "level",
                lambda i2, i3: {"source": "DBNOMICS", "series_id": f"BIS/WS_CBPOL/M.{i2 if i2 not in EURO_AREA else 'XM'}"})

    # 5. M2 / Broad Money (FRED typical patterns)
    add_feature("m2_money", "Monetary & Debt", "m2_money", "log_diff",
                lambda i2, i3: {"source": "FRED", "series_id": f"MYAGM2{i2}M189N"} if i3 != "USA" else {"source": "FRED", "series_id": "WM2NS"})

    # 6. FX Reserves (FRED)
    add_feature("fx_reserves", "Monetary & Debt", "fx_reserves", "log_diff",
                lambda i2, i3: {"source": "FRED", "series_id": f"TRESEG{i2}M052N"})

    # 7. Real Effective Exchange Rate (BIS dataset in DBnomics)
    add_feature("reer", "Monetary & Debt", "cpi", "log_diff",
                lambda i2, i3: {"source": "DBNOMICS", "series_id": f"BIS/WS_EER/M.R.N.B.{i2}.02"})

    # 8. 10Y Yield (FRED IRLTLT01 pattern)
    add_feature("10y_yield", "Monetary & Debt", "10y_yield", "level",
                lambda i2, i3: {"source": "FRED", "series_id": f"IRLTLT01{i2}M156N"})

    # 9. Trade Balance (FRED Exports - Imports proxies)
    add_feature("exports", "External Order", "trade_balance", "log_diff",
                lambda i2, i3: {"source": "FRED", "series_id": f"XTEXVA01{i2}M667S"})
    add_feature("imports", "External Order", "trade_balance", "log_diff",
                lambda i2, i3: {"source": "FRED", "series_id": f"XTIMVA01{i2}M667S"})

    # Save to configs/data.yaml
    with open("configs/data.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False, indent=2)

    print(f"Generated YAML with {len(config['nations'])} nations and {len(config['features'])} features.")
    print("Ready for ingestion.")

if __name__ == "__main__":
    generate_config()
