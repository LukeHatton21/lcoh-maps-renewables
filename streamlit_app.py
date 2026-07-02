# streamlit_app.py
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import xarray as xr

st.set_page_config(page_title="Green H2 Cost & Carbon Explorer", layout="wide")
st.title("Green Hydrogen Cost & Carbon Explorer")
st.caption("Explore spatial variation in LCOH, capacity, and carbon intensity for Alkaline and PEM pathways.")

# =========================
# Secrets / URLs
# =========================
PEM_URL = os.getenv("PEM_URL", "")
ALK_URL = os.getenv("ALK_URL", "")

PEM_URL = "https://www.dropbox.com/scl/fi/vzcuxj5znytwuhzz207iv/PEM_COLLATED_RESULTS.nc?rlkey=z9heh3eums23uxcrgw9pspzck&st=v8ku4dpx&dl=1"
ALK_URL = "https://www.dropbox.com/scl/fi/fy0c1qp70r64cjy3hxe1k/ALK_COLLATED_RESULTS.nc?rlkey=5pt5amo1ocbsanzne1is4nq2x&st=bghmbym3&dl=1"

try:
    PEM_URL = st.secrets.get("PEM_URL", PEM_URL)
    ALK_URL = st.secrets.get("ALK_URL", ALK_URL)
except Exception:
    pass

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Professional variable labels
# =========================
VAR_LABELS = {
    "levelised_cost": "LCOH (USD/kg)",
    "levelised_cost_ren": "Renewables LCOH component (USD/kg)",
    "levelised_cost_elec": "Electrolyser LCOH component (USD/kg)",
    "carbon_intensity": "Carbon intensity (kgCO2/kgH2)",
    "carbon_intensity_low": "Carbon intensity low (kgCO2/kgH2)",
    "carbon_intensity_high": "Carbon intensity high (kgCO2/kgH2)",
    "hydrogen_technical_potential": "Hydrogen technical potential",
    "hydrogen_production": "Hydrogen production (TPA/MW)",
    "electrolyser_capacity": "Electrolyser capacity (%, MW)",
    "renewable_electricity": "Annual electricity output (MWh per MW)",
    "solar_costs": "Solar CAPEX for 1 MW (USD/MW)",
    "electrolyser_costs": "ElectrolyserCAPEX for 1 MW (USD/kW)",
    "renewables_costs": "Renewables CAPEX for 1 MW (USD/kW)",
}
LABEL_TO_VAR = {v: k for k, v in VAR_LABELS.items()}

# =========================
# Download + load helpers
# =========================
def _download_if_missing(url: str, out_path: Path) -> Path:
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    if not url:
        raise ValueError(f"Missing URL for {out_path.name}. Set in st.secrets or env vars.")

    with requests.get(url, stream=True, timeout=300, allow_redirects=True) as r:
        r.raise_for_status()
        ctype = r.headers.get("content-type", "").lower()
        if "text/html" in ctype:
            raise ValueError(f"URL returned HTML, not NetCDF: {url}")
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    with open(out_path, "rb") as f:
        sig = f.read(4)
    if sig[:3] != b"CDF" and sig != b"\x89HDF":
        raise ValueError(f"Downloaded file is not NetCDF/HDF5: {out_path}")

    return out_path


@st.cache_resource(show_spinner=False)
def _load_nc(tech: str) -> xr.Dataset:
    tech = tech.lower()
    if tech == "PEM":
        fp = _download_if_missing(PEM_URL, DATA_DIR / "pem.nc")
    else:
        fp = _download_if_missing(ALK_URL, DATA_DIR / "alkaline.nc")

    return xr.open_dataset(
        fp,
        engine="netcdf4",
        chunks={"solar_fraction": 1, "latitude": 180, "longitude": 180},
    )

@st.cache_data(show_spinner=False)
def load_country_mapping() -> pd.DataFrame:
    fp = DATA_DIR / "country_mapping_regions.csv"
    if not fp.exists():
        return pd.DataFrame(columns=["country_id", "country_name", "region"])

    cm = pd.read_csv(fp)

    # flexible column names
    cols = {c.lower(): c for c in cm.columns}
    id_col = cols.get("index") or cols.get("country") or cols.get("country_id") or cols.get("country_number")
    name_col = cols.get("country_name") or cols.get("name")
    region_col = cols.get("region")

    if not id_col or not name_col or not region_col:
        return pd.DataFrame(columns=["country_id", "country_name", "region"])

    cm = cm[[id_col, name_col, region_col]].copy()
    cm.columns = ["country_id", "country_name", "region"]

    cm["country_id"] = pd.to_numeric(cm["country_id"], errors="coerce").astype("Int64")
    cm["country_name"] = cm["country_name"].astype("string")
    cm["region"] = cm["region"].astype("string")

    cm = cm.dropna(subset=["country_id"]).drop_duplicates(subset=["country_id"])
    return cm

@st.cache_data(show_spinner=False)
def load_cost_mapping() -> pd.DataFrame:
    fp = DATA_DIR / "IRENA_Country_Costs_2024.csv"
    if not fp.exists():
        return pd.DataFrame(columns=["index", "solar_costs_usd_kw", "wind_costs_usd_kw"])

    cm = pd.read_csv(fp)

    # flexible column names
    cols = {c.lower(): c for c in cm.columns}
    id_col = cols.get("index") 
    solar_col = cols.get("solar_costs_usd_kw")
    wind_col = cols.get("wind_costs_usd_kw")

    if not id_col or not solar_col or not wind_col:
        return pd.DataFrame(columns=["index", "solar_costs_usd_kw", "wind_costs_usd_kw"])

    cm = cm[[id_col, solar_col, wind_col]].copy()
    cm.columns = ["country_id", "solar_costs_usd_kw", "wind_costs_usd_kw"]
    inflation_2025 = 1.05

    cm["country_id"] = pd.to_numeric(cm["country_id"], errors="coerce").astype("Int64")
    cm["solar_costs_usd_kw"] = pd.to_numeric(cm["solar_costs_usd_kw"], errors="coerce").astype("Int64")
    cm["wind_costs_usd_kw"] = pd.to_numeric(cm["wind_costs_usd_kw"], errors="coerce").astype("Int64")

    # Convert to 2025 terms
    cm["solar_costs_usd_kw"] = cm["solar_costs_usd_kw"]*inflation_2025
    cm["wind_costs_usd_kw"] = cm["wind_costs_usd_kw"]*inflation_2025

    cm = cm.dropna(subset=["country_id"]).drop_duplicates(subset=["country_id"])
    return cm

def regional_lcoh_component_breakdown(
    df: pd.DataFrame,
    lifetime_years: int = 20,
    elec_om_frac: float = 0.03,
    solar_om_frac: float = 0.023,
    wind_om_frac: float = 0.026,
) -> pd.DataFrame:
    """
    Regional-average LCOH breakdown:
      - renewables_undiscounted_lcoh
      - electrolyser_undiscounted_lcoh
      - residual_finance_other_lcoh

    Assumptions:
      - solar_costs, wind_costs, electrolyser_costs are CAPEX terms per site basis
      - annual O&M = fraction * CAPEX
      - lifetime = 20 years
      - hydrogen_production is annual production (same basis as your LCOH denominator)
    """

    req = [
        "region",
        "levelised_cost",
        "hydrogen_production",
        "solar_costs",
        "wind_costs",
        "renewables_costs",
        "electrolyser_costs",
    ]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d = df.copy()
    d = d.dropna(subset=["region", "levelised_cost", "hydrogen_production"])
    d = d[d["hydrogen_production"] > 0]

    # Fill missing CAPEX terms with 0 for component calc
    for c in ["solar_costs", "wind_costs", "electrolyser_costs"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    #d["wind_costs"] = d["renewables_costs"] - d["solar_costs"]

    # Regional averages first (as requested)
    reg = (
        d.groupby("region", as_index=False)
        .agg(
            lcoh_mean=("levelised_cost", "mean"),
            hydrogen_production_mean=("hydrogen_production", "mean"),
            solar_costs_mean=("solar_costs", "mean"),
            wind_costs_mean=("wind_costs", "mean"),
            electrolyser_costs_mean=("electrolyser_costs", "mean"),
        )
    )

    # Undiscounted lifetime costs
    reg["renewables_capex_mean"] = reg["solar_costs_mean"] + reg["wind_costs_mean"]
    reg["renewables_annual_om_mean"] = (
        solar_om_frac * reg["solar_costs_mean"] + wind_om_frac * reg["wind_costs_mean"]
    )
    reg["renewables_total_lifetime_cost_mean"] = (
        reg["renewables_capex_mean"] + lifetime_years * reg["renewables_annual_om_mean"]
    )

    reg["electrolyser_annual_om_mean"] = elec_om_frac * reg["electrolyser_costs_mean"]
    reg["electrolyser_total_lifetime_cost_mean"] = (
        reg["electrolyser_costs_mean"] + lifetime_years * reg["electrolyser_annual_om_mean"]
    )

    reg["hydrogen_lifetime_mean"] = lifetime_years * reg["hydrogen_production_mean"]

    # Convert to LCOH components
    reg["renewables_undiscounted_lcoh"] = (
        reg["renewables_total_lifetime_cost_mean"] / reg["hydrogen_lifetime_mean"]
    )
    reg["electrolyser_undiscounted_lcoh"] = (
        reg["electrolyser_total_lifetime_cost_mean"] / reg["hydrogen_lifetime_mean"]
    )

    reg["undiscounted_sum_lcoh"] = (
        reg["renewables_undiscounted_lcoh"] + reg["electrolyser_undiscounted_lcoh"]
    )

    # residual to observed mean LCOH
    reg["residual_finance_other_lcoh"] = reg["lcoh_mean"] - reg["undiscounted_sum_lcoh"]

    # shares
    reg["renewables_share"] = reg["renewables_undiscounted_lcoh"] / reg["lcoh_mean"]
    reg["electrolyser_share"] = reg["electrolyser_undiscounted_lcoh"] / reg["lcoh_mean"]
    reg["residual_share"] = reg["residual_finance_other_lcoh"] / reg["lcoh_mean"]

    return reg.sort_values("lcoh_mean")

def apply_country_region_mapping(df: pd.DataFrame, cm: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["country_name"] = pd.Series(dtype="string")
        out["region"] = pd.Series(dtype="string")
        return out

    # country index from NetCDF expected in 'country'
    if "country" not in out.columns:
        out["country_id"] = pd.Series([pd.NA] * len(out), dtype="Int64")
    else:
        out["country_id"] = pd.to_numeric(out["country"], errors="coerce").astype("Int64")

    if cm.empty:
        out["country_name"] = "Unknown"
        out["region"] = "Unknown"
        return out

    out = out.merge(cm, on="country_id", how="left")
    out["country_name"] = out["country_name"].fillna("Unknown").astype("string")
    out["region"] = out["region"].fillna("Unknown").astype("string")
    return out

def _grid_resolution(vals: pd.Series, default: float = 0.5) -> float:
    v = np.sort(pd.to_numeric(vals.dropna().unique(), errors="coerce"))
    if len(v) < 2:
        return default
    d = np.diff(v)
    d = d[d > 0]
    return float(np.median(d)) if len(d) else default

def build_cell_geojson(
    df: pd.DataFrame,
    value_col: str = "levelised_cost",
    id_col: str = "cell_id",
) -> tuple[dict, pd.DataFrame]:
    d = df.dropna(subset=["latitude", "longitude", value_col]).copy()
    if d.empty:
        return {"type": "FeatureCollection", "features": []}, d

    lat_step = _grid_resolution(d["latitude"], default=0.5)
    lon_step = _grid_resolution(d["longitude"], default=0.5)
    half_lat = lat_step / 2.0
    half_lon = lon_step / 2.0

    d[id_col] = (
        d["latitude"].round(6).astype(str) + "_" + d["longitude"].round(6).astype(str)
    )

    features = []
    for _, r in d.iterrows():
        lat = float(r["latitude"])
        lon = float(r["longitude"])

        # bounds
        lat_min = max(-90.0, lat - half_lat)
        lat_max = min(90.0, lat + half_lat)
        lon_min = max(-180.0, lon - half_lon)
        lon_max = min(180.0, lon + half_lon)

        poly = [
            [lon_min, lat_min],
            [lon_max, lat_min],
            [lon_max, lat_max],
            [lon_min, lat_max],
            [lon_min, lat_min],
        ]

        features.append(
            {
                "type": "Feature",
                "id": r[id_col],
                "properties": {
                    id_col: r[id_col],
                    "region": str(r.get("region", "")),
                    "country_name": str(r.get("country_name", "")),
                    value_col: None if pd.isna(r[value_col]) else float(r[value_col]),
                },
                "geometry": {"type": "Polygon", "coordinates": [poly]},
            }
        )

    geojson = {"type": "FeatureCollection", "features": features}
    return geojson, d

def regional_lcoh_component_breakdown(
    df: pd.DataFrame,
    lifetime_years: int = 20,
    elec_om_frac: float = 0.03,
    solar_om_frac: float = 0.023,
    wind_om_frac: float = 0.026,
) -> pd.DataFrame:
    """
    Regional-average LCOH breakdown:
      - renewables_undiscounted_lcoh
      - electrolyser_undiscounted_lcoh
      - residual_finance_other_lcoh

    Assumptions:
      - solar_costs, wind_costs, electrolyser_costs are CAPEX terms per site basis
      - annual O&M = fraction * CAPEX
      - lifetime = 20 years
      - hydrogen_production is annual production (same basis as your LCOH denominator)
    """

    req = [
        "region",
        "levelised_cost",
        "hydrogen_production",
        "solar_costs",
        "wind_costs",
        "electrolyser_costs",
    ]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d = df.copy()
    d = d.dropna(subset=["region", "levelised_cost", "hydrogen_production"])
    d = d[d["hydrogen_production"] > 0]

    # Fill missing CAPEX terms with 0 for component calc
    for c in ["solar_costs", "wind_costs", "electrolyser_costs"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)

    # Regional averages first (as requested)
    reg = (
        d.groupby("region", as_index=False)
        .agg(
            lcoh_mean=("levelised_cost", "mean"),
            hydrogen_production_mean=("hydrogen_production", "mean"),
            solar_costs_mean=("solar_costs", "mean"),
            wind_costs_mean=("wind_costs", "mean"),
            electrolyser_costs_mean=("electrolyser_costs", "mean"),
        )
    )

    # Undiscounted lifetime costs
    reg["renewables_capex_mean"] = reg["solar_costs_mean"] + reg["wind_costs_mean"]
    reg["renewables_annual_om_mean"] = (
        solar_om_frac * reg["solar_costs_mean"] + wind_om_frac * reg["wind_costs_mean"]
    )
    reg["renewables_total_lifetime_cost_mean"] = (
        reg["renewables_capex_mean"] + lifetime_years * reg["renewables_annual_om_mean"]
    )

    reg["electrolyser_annual_om_mean"] = elec_om_frac * reg["electrolyser_costs_mean"]
    reg["electrolyser_total_lifetime_cost_mean"] = (
        reg["electrolyser_costs_mean"] + lifetime_years * reg["electrolyser_annual_om_mean"]
    )

    reg["hydrogen_lifetime_mean"] = lifetime_years * reg["hydrogen_production_mean"]

    # Convert to LCOH components
    reg["renewables_undiscounted_lcoh"] = (
        reg["renewables_total_lifetime_cost_mean"] / reg["hydrogen_lifetime_mean"] / 1000
    )
    reg["electrolyser_undiscounted_lcoh"] = (
        reg["electrolyser_total_lifetime_cost_mean"] / reg["hydrogen_lifetime_mean"] / 1000
    )

    reg["undiscounted_sum_lcoh"] = (
        reg["renewables_undiscounted_lcoh"] + reg["electrolyser_undiscounted_lcoh"]
    )

    # residual to observed mean LCOH
    reg["residual_finance_other_lcoh"] = reg["lcoh_mean"] - reg["undiscounted_sum_lcoh"]

    # shares
    reg["renewables_share"] = reg["renewables_undiscounted_lcoh"] / reg["lcoh_mean"]
    reg["electrolyser_share"] = reg["electrolyser_undiscounted_lcoh"] / reg["lcoh_mean"]
    reg["residual_share"] = reg["residual_finance_other_lcoh"] / reg["lcoh_mean"]

    return reg.sort_values("lcoh_mean")

def apply_country_mapping(df: pd.DataFrame, cm: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    """
    Map numeric country ID in NetCDF to country_name + region + costs from CSV.
    """
    out = df.copy()
    if out.empty or cm.empty:
        # ensure expected columns exist
        if "country_name" not in out.columns:
            out["country_name"] = pd.Series(["Unknown"] * len(out), dtype="string")
        return out

    # Find country-id column in data
    candidate_cols = ["country", "country_id", "country_number"]
    data_country_col = next((c for c in candidate_cols if c in out.columns), None)

    if data_country_col is None:
        if "country_name" not in out.columns:
            out["country_name"] = pd.Series(["Unknown"] * len(out), dtype="string")
        return out

    out["country_id"] = pd.to_numeric(out[data_country_col], errors="coerce").astype("Int64")

    out = out.merge(cm, on="country_id", how="left", suffixes=("", "_map"))

    # region priority: mapped region, else existing region
    if "region_map" in out.columns:
        out["region"] = out["region_map"].combine_first(out.get("region"))
        out = out.drop(columns=["region_map"])

    # country_name final
    if "country_name" not in out.columns:
        out["country_name"] = "Unknown"
    out["country_name"] = out["country_name"].fillna("Unknown").astype("string")

    # Merge cost onto output
    out = out.merge(costs[["country_id", "solar_costs_usd_kw", "wind_costs_usd_kw"]], how="left", suffixes=("", "_costs"))

    return out


@st.cache_data(show_spinner=False)
def load_points_slice_from_nc(tech: str, solar_fraction: int, columns: list[str]) -> pd.DataFrame:
    ds = _load_nc(tech)
    ds_sf = ds.sel(solar_fraction=solar_fraction)

    available = [c for c in columns if (c in ds_sf.data_vars or c in ds_sf.coords)]
    if len(available) == 0:
        return pd.DataFrame()

    df = ds_sf[available].to_dataframe().reset_index()

    defaults = {
        "country": "Unknown",
        "carbon_intensity": np.nan,
        "carbon_intensity_low": np.nan,
        "carbon_intensity_high": np.nan,
        "hydrogen_technical_potential": np.nan,
        "hydrogen_production": np.nan,
        "renewable_electricity": np.nan,
        "electrolyser_capacity": np.nan,
        "solar_costs": np.nan,
        "electrolyser_costs": np.nan,
        "renewables_costs": np.nan,
        "wind_costs": np.nan,
        "levelised_cost": np.nan,
        "levelised_cost_ren": np.nan,
        "levelised_cost_elec": np.nan,
    }
    for c, d in defaults.items():
        if c not in df.columns:
            df[c] = d

    # Convert electrolyser ratio
    df["electrolyser_capacity"] = df["electrolyser_capacity"] * 100

    numeric_cols = [
        "latitude", "longitude", "levelised_cost", "levelised_cost_ren", "levelised_cost_elec",
        "solar_costs", "wind_costs", "renewables_costs", "carbon_intensity", "carbon_intensity_low",
        "carbon_intensity_high", "hydrogen_technical_potential", "electrolyser_capacity", "renewable_electricity", "hydrogen_production",
        "solar_costs_usd_kw", "wind_costs_usd_kw",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    for c in ["region", "country"]:
        if c in df.columns:
            df[c] = df[c].astype("string")

    return df


def change_capex_absolute_df(
    df: pd.DataFrame,
    solar_capex: float,
    solar_capex_original: float,
    wind_capex: float,
    wind_capex_original: float,
    elec_capex: float,
    initial_capex: float,
) -> pd.DataFrame:
    out = df.copy()

    ren_lcoh = out["levelised_cost_ren"]
    elec_lcoh = out["levelised_cost_elec"]
    total_lcoh = out["levelised_cost"]

    solar_costs_frac = out["solar_costs"] / out["renewables_costs"]
    solar_costs_frac = solar_costs_frac.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    new_ren_lcoh = (
        (1 - solar_costs_frac) * ren_lcoh * (1 + (wind_capex - wind_capex_original) / wind_capex_original)
        + solar_costs_frac * ren_lcoh * (1 + (solar_capex - solar_capex_original) / solar_capex_original)
    )
    new_elec_lcoh = elec_lcoh * (1 + (elec_capex - initial_capex) / initial_capex)

    out["Calculated_LCOH"] = total_lcoh - ren_lcoh + new_ren_lcoh - elec_lcoh + new_elec_lcoh
    out["delta_lcoh"] = out["Calculated_LCOH"] - out["levelised_cost"]
    return out


def summarize_group(df: pd.DataFrame, group_col: str, lcoh_col: str = "levelised_cost") -> pd.DataFrame:
    valid = df.dropna(subset=[group_col, lcoh_col]).copy()
    if valid.empty:
        return pd.DataFrame()

    # average CAPEX in USD/kW at group level
    summary = (
        valid.groupby(group_col, as_index=False)
        .agg(
            n_points=(group_col, "size"),
            lcoh_mean=(lcoh_col, "mean"),
            lcoh_p10=(lcoh_col, lambda x: x.quantile(0.1)),
            lcoh_p90=(lcoh_col, lambda x: x.quantile(0.9)),
            carbon_mean=("carbon_intensity", "mean"),
            electrolyser_capacity_mean_mw=("electrolyser_capacity", "mean"),
            solar_capex_usd_kw_mean=("solar_costs_usd_kw", "mean"),
            wind_capex_usd_kw_mean=("wind_costs_usd_kw", "mean"),
            elec_capex_component_mean=("levelised_cost_elec", "mean"),
        )
        .sort_values("lcoh_mean")
    )
    return summary


# ---------------------------
# Sidebar controls (global)
# ---------------------------
with st.sidebar:
    st.header("Global Controls")
    tech = st.selectbox("Electrolyser Technology", ["Alkaline", "PEM"], index=0)
    solar_fraction = st.select_slider(
        "Solar Fraction (%)",
        options=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        value=50,
    )

needed_cols = [
    "solar_fraction",
    "latitude",
    "longitude",
    "country",   # numeric code from NetCDF
    "levelised_cost",
    "levelised_cost_ren",
    "levelised_cost_elec",
    "solar_costs",
    "wind_costs",
    "renewables_costs",
    "renewable_electricity",
    "hydrogen_production",
    "electrolyser_costs",
    "carbon_intensity",
    "carbon_intensity_low",
    "carbon_intensity_high",
    "hydrogen_technical_potential",
    "electrolyser_capacity",
]

with st.spinner("Downloading/loading underlying data..."):
    df_sf = load_points_slice_from_nc(tech=tech, solar_fraction=solar_fraction, columns=needed_cols)

country_map = load_country_mapping()
cost_mapping = load_cost_mapping()
df_sf = apply_country_mapping(df_sf, country_map, cost_mapping)

if not df_sf.empty:
    df_sf = df_sf.dropna(subset=["levelised_cost"], how="all")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🗺️ Map Explorer", "🧮 CAPEX Sensitivity (Regional)", "🌍 Regional Summary", "🏳️ Country Summary"]
)

# ==========================================================
# TAB 1: MAP EXPLORER
# ==========================================================
with tab1:
    st.subheader("Spatial Explorer")

    if df_sf.empty:
        st.warning("No data found for this technology / solar fraction selection.")
    else:
        metric_label = st.selectbox(
            "Map metric",
            options=[
                VAR_LABELS["levelised_cost"],
                VAR_LABELS["hydrogen_production"],
                VAR_LABELS["renewable_electricity"],
                VAR_LABELS["electrolyser_capacity"],  # requested addition + default,
                VAR_LABELS["carbon_intensity"],
                VAR_LABELS["hydrogen_technical_potential"],
            ],
            index=0,
        )
        metric = LABEL_TO_VAR[metric_label]

        c1, c2 = st.columns([1, 1])
        with c1:
            color_scale = st.selectbox("Color scale", ["Viridis", "Turbo", "Plasma", "Cividis"], index=0)
        with c2:
            # default to maximum points
            max_points = st.slider("Max points on map", 2000, 50000, 50000, step=1000)

        df_map = df_sf.dropna(subset=[metric]).copy()
        if len(df_map) > max_points:
            df_map = df_map.sample(max_points, random_state=42)

        if df_map.empty:
            st.warning("No data available for this metric.")
        else:
            q_low = df_map[metric].quantile(0.02)
            q_hi = df_map[metric].quantile(0.98)
            if np.isclose(q_low, q_hi):
                q_low, q_hi = df_map[metric].min(), df_map[metric].max()

            fig = px.scatter_mapbox(
                df_map,
                lat="latitude",
                lon="longitude",
                color=metric,
                color_continuous_scale=color_scale,
                range_color=(0, q_hi) if np.isfinite(q_low) and np.isfinite(q_hi) else None,
                labels={metric: metric_label},
                hover_data={
                    "region": True,
                    "country": True if "country" in df_map.columns else False,
                     metric: ":.3f",
                    "latitude": ":.2f",
                    "longitude": ":.2f",
                },
                zoom=1,
                height=650,
            )
            metric_label_wrapped = metric_label.replace(" ", "<br>")
            fig.update_layout(
                coloraxis_colorbar=dict(
                    title=dict(
                        text=metric_label_wrapped
                    )
                )
            )
            fig.update_layout(mapbox_style="carto-positron", margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# TAB 2: CAPEX SENSITIVITY (REGIONAL ONLY)
# ==========================================================
with tab2:
    st.subheader("Regional CAPEX Sensitivity")
    st.caption("Exploration is restricted to a selected region, with defaults from average regional CAPEX.")

    if df_sf.empty:
        st.warning("No data found for this technology / solar fraction selection.")
    else:
        regions = sorted([r for r in df_sf["region"].dropna().unique().tolist() if r != "Unknown"])
        if not regions:
            st.warning("No region data available.")
        else:
            selected_region = st.selectbox("Select region", regions, index=0)
            df_reg = df_sf[df_sf["region"] == selected_region].copy()

            if df_reg.empty:
                st.warning("No data for selected region.")
            else:
                # defaults from regional averages (USD/kW for solar/wind)
                reg_solar_kw = float(df_reg["solar_costs_usd_kw"].mean()) if "solar_costs_usd_kw" in df_reg else 990
                reg_wind_kw = float(df_reg["wind_costs_usd_kw"].mean()) if "wind_costs_usd_kw" in df_reg else 1500

                # convert back to USD/MW for existing CAPEX function baseline math
                reg_solar_mw = reg_solar_kw 
                reg_wind_mw = reg_wind_kw 

                # proxy regional electrolyser default from elec component (or tech default if preferred)
                default_elec = 1700.0 if tech == "Alkaline" else 2000

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    solar_capex = st.number_input("Solar CAPEX (USD/kW)", min_value=100.0, max_value=6000.0, value=float(np.nan_to_num(reg_solar_mw, nan=990.0)), step=10.0)
                with c2:
                    wind_capex = st.number_input("Wind CAPEX (USD/kW)", min_value=100.0, max_value=8000.0, value=float(np.nan_to_num(reg_wind_mw, nan=1500.0)), step=10.0)
                with c3:
                    elec_capex = st.number_input("Electrolyser CAPEX (USD/kW)", min_value=100.0, max_value=6000.0, value=default_elec, step=10.0)
                with c4:
                    initial_capex = st.number_input("Initial electrolyser CAPEX baseline (USD/kW)", min_value=100.0, max_value=6000.0, value=default_elec, step=10.0)

                st.markdown(
                    f"Regional baseline CAPEX averages: "
                    f"Solar = **{reg_solar_kw:.0f} USD/kW**, Wind = **{reg_wind_kw:.0f} USD/kW**"
                )

                if st.button("Run Regional CAPEX Recalculation", type="primary"):
                    out = change_capex_absolute_df(
                        df=df_reg,
                        solar_capex=solar_capex,
                        solar_capex_original=float(np.nan_to_num(reg_solar_mw, nan=990.0)),
                        wind_capex=wind_capex,
                        wind_capex_original=float(np.nan_to_num(reg_wind_mw, nan=1500.0)),
                        elec_capex=elec_capex,
                        initial_capex=initial_capex,
                    )
                    st.metric("Regional mean ΔLCOH", f"{out['delta_lcoh'].mean():.3f}")

                    # --- mini regional map of recalculated LCOH ---
                    # ---------------------------------------
                    # ---------------------------------------
                    VAR_MAP = {
                        "levelised_cost": {
                            "label": "LCOH (USD/kg)",
                            "fmt": ":.3f",
                        },
                        "Calculated_LCOH": {
                            "label": "Recalculated LCOH (USD/kg)",
                            "fmt": ":.3f",
                        },
                        "delta_lcoh": {
                            "label": "ΔLCOH (USD/kg)",
                            "fmt": ":.3f",
                        },
                        "carbon_intensity": {
                            "label": "Carbon intensity (kgCO2/kgH2)",
                            "fmt": ":.3f",
                        },
                        "electrolyser_capacity": {
                            "label": "Electrolyser capacity (MW)",
                            "fmt": ":.2f",
                        },
                        "hydrogen_technical_potential": {
                            "label": "Hydrogen technical potential",
                            "fmt": ":.2f",
                        },
                    }

                    LABEL_TO_VAR = {v["label"]: k for k, v in VAR_MAP.items()}
                    map_metric = "Calculated_LCOH"
                    metric_label = VAR_MAP[map_metric]["label"]
                    map_df = out.dropna(subset=[map_metric, "latitude", "longitude"]).copy()

                    if map_df.empty:
                        st.warning("No mappable points for selected region.")
                    else:
                        max_region_points = 6000
                        if len(map_df) > max_region_points:
                            map_df = map_df.sample(max_region_points, random_state=42)

                        q_hi = map_df[map_metric].quantile(0.98)
                        if not np.isfinite(q_hi):
                            q_hi = map_df[map_metric].max()
                        if not np.isfinite(q_hi) or q_hi <= 0:
                            q_hi = 1e-6

                        hover_data = {
                            "region": True,
                            "country_name": True if "country_name" in map_df.columns else False,
                            "latitude": ":.2f",
                            "longitude": ":.2f",
                            "levelised_cost": VAR_MAP["levelised_cost"]["fmt"] if "levelised_cost" in map_df.columns else False,
                            map_metric: VAR_MAP[map_metric]["fmt"],
                        }

                        fig = px.scatter_mapbox(
                            map_df,
                            lat="latitude",
                            lon="longitude",
                            color=map_metric,
                            color_continuous_scale="Turbo",
                            range_color=(0, q_hi),
                            labels={k: VAR_MAP[k]["label"] for k in VAR_MAP if k in map_df.columns},
                            hover_data=hover_data,
                            zoom=3,
                            height=420,
                        )

                        fig.update_layout(
                            mapbox_style="carto-positron",
                            coloraxis_colorbar=dict(title=metric_label),
                            margin=dict(l=0, r=0, t=30, b=0),
                            title=f"{metric_label} — {selected_region}",
                        )

                        st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# TAB 3: REGIONAL SUMMARY
# ==========================================================
with tab3:
    st.subheader("Regional Summary (grouped by region)")
    if df_sf.empty:
        st.warning("No data found.")
    else:
        lcoh_field = st.selectbox(
            "LCOH field for summary",
            options=[VAR_LABELS["levelised_cost"]],
            index=0,
            key="region_lcoh_field",
        )
        lcoh_var = LABEL_TO_VAR[lcoh_field]
        reg_summary = summarize_group(df_sf, group_col="region", lcoh_col=lcoh_var)
        if reg_summary.empty:
            st.warning("No regional summary available.")
        else:
            st.markdown("### Regional LCOH comparison")

            # Bar: mean LCOH by region
            fig_reg_lcoh = px.bar(
                reg_summary.sort_values("lcoh_mean"),
                x="region",
                y="lcoh_mean",
                title="Mean LCOH by region",
                labels={"lcoh_mean": "Mean LCOH (USD/kg)", "region": "Region"},
            )
            fig_reg_lcoh.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig_reg_lcoh, use_container_width=True)

            st.markdown("### Cost component differences by region")

            # Build component summary from df_sf directly
            reg_break = regional_lcoh_component_breakdown(df_sf)

            st.markdown("### Regional LCOH component breakdown (regional averages)")

            # Stacked absolute components
            comp_long = reg_break.melt(
                id_vars="region",
                value_vars=[
                    "residual_finance_other_lcoh",
                    "renewables_undiscounted_lcoh",
                    "electrolyser_undiscounted_lcoh",
                                    ],
                var_name="component",
                value_name="lcoh_component",
            )
            comp_long["component"] = comp_long["component"].map({
                "renewables_undiscounted_lcoh": "Renewables (undiscounted)",
                "electrolyser_undiscounted_lcoh": "Electrolyser (undiscounted)",
                "residual_finance_other_lcoh": "Residual (financing/other)",
            })

            fig_abs = px.bar(
                comp_long,
                x="region",
                y="lcoh_component",
                color="component",
                barmode="stack",
                title="Regional LCOH decomposition (USD/kg)",
                labels={"lcoh_component": "LCOH component (USD/kg)", "region": "Region"},
            )
            fig_abs.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig_abs, use_container_width=True)

            # 100% stacked shares
            share_long = reg_break.melt(
                id_vars="region",
                value_vars=["residual_share", "renewables_share", "electrolyser_share"],
                var_name="component",
                value_name="share",
            )
            share_long["component"] = share_long["component"].map({
                "renewables_share": "Renewables (undiscounted)",
                "electrolyser_share": "Electrolyser (undiscounted)",
                "residual_share": "Residual (financing/other)",
            })

            fig_share = px.bar(
                share_long,
                x="region",
                y="share",
                color="component",
                barmode="stack",
                title="Regional LCOH decomposition share",
                labels={"share": "Share of LCOH", "region": "Region"},
            )
            fig_share.update_layout(xaxis_tickangle=-35, yaxis_tickformat=".0%")
            st.plotly_chart(fig_share, use_container_width=True)

# ==========================================================
# TAB 4: COUNTRY SUMMARY (new)
# ==========================================================
with tab4:
    st.subheader("Country Summary (grouped by country)")
    if df_sf.empty:
        st.warning("No data found.")
    else:
        if "country" not in df_sf.columns or df_sf["country"].isna().all():
            st.info("Country column not yet available in dataset. Add 'country' variable to enable this tab.")
        else:
            st.markdown("### Country deep-dive")

            country_options = sorted(df_sf["country_name"].dropna().unique().tolist())
            selected_country = st.selectbox("Select country", country_options, key="country_select")

            df_country = df_sf[df_sf["country_name"] == selected_country].copy()

            if df_country.empty:
                st.warning("No data for selected country.")
            else:
                # ---- metrics ----
                lcoh_min = df_country["levelised_cost"].min()
                lcoh_p10 = df_country["levelised_cost"].quantile(0.10)
                lcoh_p50 = df_country["levelised_cost"].median()
                lcoh_p90 = df_country["levelised_cost"].quantile(0.90)
                lcoh_max = df_country["levelised_cost"].max()

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("LCOH min", f"{lcoh_min:.2f}")
                c2.metric("LCOH p10", f"{lcoh_p10:.2f}")
                c3.metric("LCOH median", f"{lcoh_p50:.2f}")
                c4.metric("LCOH p90", f"{lcoh_p90:.2f}")
                c5.metric("LCOH max", f"{lcoh_max:.2f}")

                # ---- subset map (auto-zoom to country extent) ----
                lat_c = float(df_country["latitude"].mean())
                lon_c = float(df_country["longitude"].mean())

                value_col = "levelised_cost"

                geojson_cells, df_cells = build_cell_geojson(df_country, value_col=value_col, id_col="cell_id")

                if len(df_cells) == 0:
                    st.warning("No data for cell map.")
                else:
                    lat_c = float(df_cells["latitude"].mean())
                    lon_c = float(df_cells["longitude"].mean())

                    LCOH_MIN = 0.0
                    LCOH_MAX = 20.0

                    fig_cells = px.choropleth_mapbox(
                        df_cells,
                        geojson=geojson_cells,
                        locations="cell_id",
                        featureidkey="properties.cell_id",
                        color=value_col,
                        color_continuous_scale="Viridis",
                        range_color=(LCOH_MIN, LCOH_MAX),
                        hover_data={
                            "latitude": ":.2f",
                            "longitude": ":.2f",
                            "region": True,
                            "country_name": True,
                            "levelised_cost": ":.3f",
                        },
                        center={"lat": lat_c, "lon": lon_c},
                        zoom=4,
                        opacity=0.65,
                        title=f"{selected_country}: exact lat/lon grid cells",
                        height=700,
                    )
                    fig_cells.update_layout(mapbox_style="carto-positron", margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(fig_cells, use_container_width=True)

                # ---- cumulative cost-technical potential curve ----
                curve = df_country.dropna(subset=["levelised_cost", "hydrogen_technical_potential"]).copy()
                curve = curve[curve["hydrogen_technical_potential"] > 0].copy()

                if curve.empty:
                    st.info("No positive technical potential data available for cumulative curve.")
                else:
                    curve = curve.sort_values("levelised_cost")
                    curve["cum_potential"] = curve["hydrogen_technical_potential"].cumsum()
                    curve["cum_potential_share"] = curve["cum_potential"] / curve["hydrogen_technical_potential"].sum()

                    fig_curve = px.line(
                        curve,
                        x="cum_potential",
                        y="levelised_cost",
                        title=f"{selected_country}: cumulative cost-technical potential curve",
                        labels={
                            "cum_potential": "Cumulative technical potential",
                            "levelised_cost": "LCOH (USD/kg)",
                        },
                    )
                    st.plotly_chart(fig_curve, use_container_width=True)

                    fig_curve_share = px.line(
                        curve,
                        x="cum_potential_share",
                        y="levelised_cost",
                        title=f"{selected_country}: cumulative curve (normalised potential)",
                        labels={
                            "cum_potential_share": "Cumulative potential share",
                            "levelised_cost": "LCOH (USD/kg)",
                        },
                    )
                    st.plotly_chart(fig_curve_share, use_container_width=True)