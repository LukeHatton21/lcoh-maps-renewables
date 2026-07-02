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
    "electrolyser_capacity": "Electrolyser capacity (MW)",
    "solar_costs": "Solar CAPEX for 1 MW (USD/MW)",
    "wind_costs": "Wind CAPEX for 1 MW (USD/MW)",
    "renewables_costs": "Renewables CAPEX for 1 MW (USD/MW)",
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
    if tech == "pem":
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
    """
    Expects data/country_mapping_regions.csv with at least:
    - index (or country / country_id)
    - country_name
    - region
    """
    fp = DATA_DIR / "country_mapping_regions.csv"
    if not fp.exists():
        return pd.DataFrame()

    cm = pd.read_csv(fp)

    # normalize common column names
    rename_map = {}
    if "index" in cm.columns:
        rename_map["index"] = "country_id"
    elif "country" in cm.columns:
        rename_map["country"] = "country_id"
    elif "country_number" in cm.columns:
        rename_map["country_number"] = "country_id"

    if "country_name" not in cm.columns:
        # fallback if called name
        if "name" in cm.columns:
            rename_map["name"] = "country_name"

    cm = cm.rename(columns=rename_map)

    required = {"country_id", "country_name", "region"}
    if not required.issubset(set(cm.columns)):
        return pd.DataFrame()

    cm = cm[["country_id", "country_name", "region"]].copy()
    cm["country_id"] = pd.to_numeric(cm["country_id"], errors="coerce").astype("Int64")
    cm["country_name"] = cm["country_name"].astype("string")
    cm["region"] = cm["region"].astype("string")
    cm = cm.dropna(subset=["country_id"]).drop_duplicates(subset=["country_id"])
    return cm

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


def apply_country_mapping(df: pd.DataFrame, cm: pd.DataFrame) -> pd.DataFrame:
    """
    Map numeric country ID in NetCDF to country_name + region from CSV.
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
        "region": "Unknown",
        "country": "Unknown",
        "carbon_intensity": np.nan,
        "carbon_intensity_low": np.nan,
        "carbon_intensity_high": np.nan,
        "hydrogen_technical_potential": np.nan,
        "electrolyser_capacity": np.nan,
        "solar_costs": np.nan,
        "wind_costs": np.nan,
        "renewables_costs": np.nan,
        "levelised_cost": np.nan,
        "levelised_cost_ren": np.nan,
        "levelised_cost_elec": np.nan,
    }
    for c, d in defaults.items():
        if c not in df.columns:
            df[c] = d

    # unit conversion: USD/MW -> USD/kW
    if "solar_costs" in df.columns:
        df["solar_costs_usd_kw"] = pd.to_numeric(df["solar_costs"], errors="coerce") / 1000.0
    if "wind_costs" in df.columns:
        df["wind_costs_usd_kw"] = pd.to_numeric(df["wind_costs"], errors="coerce") / 1000.0

    numeric_cols = [
        "latitude", "longitude", "levelised_cost", "levelised_cost_ren", "levelised_cost_elec",
        "solar_costs", "wind_costs", "renewables_costs", "carbon_intensity", "carbon_intensity_low",
        "carbon_intensity_high", "hydrogen_technical_potential", "electrolyser_capacity",
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
    wind_capex: float,
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
        (1 - solar_costs_frac) * ren_lcoh * (1 + (wind_capex - 1500) / 1500)
        + solar_costs_frac * ren_lcoh * (1 + (solar_capex - 990) / 990)
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
    tech = st.selectbox("Electrolyser Technology", ["alkaline", "pem"], index=0)
    solar_fraction = st.select_slider(
        "Solar Fraction (%)",
        options=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        value=50,
    )

needed_cols = [
    "solar_fraction",
    "latitude",
    "longitude",
    "region",
    "country",   # numeric code from NetCDF
    "levelised_cost",
    "levelised_cost_ren",
    "levelised_cost_elec",
    "solar_costs",
    "wind_costs",
    "renewables_costs",
    "carbon_intensity",
    "carbon_intensity_low",
    "carbon_intensity_high",
    "hydrogen_technical_potential",
    "electrolyser_capacity",
]

with st.spinner("Downloading/loading underlying data..."):
    df_sf = load_points_slice_from_nc(tech=tech, solar_fraction=solar_fraction, columns=needed_cols)

country_map = load_country_mapping()
df_sf = apply_country_mapping(df_sf, country_map)

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
                VAR_LABELS["electrolyser_capacity"],  # requested addition + default
                VAR_LABELS["levelised_cost"],
                VAR_LABELS["carbon_intensity"],
                VAR_LABELS["levelised_cost_ren"],
                VAR_LABELS["levelised_cost_elec"],
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
                range_color=(q_low, q_hi) if np.isfinite(q_low) and np.isfinite(q_hi) else None,
                hover_data={
                    "region": True,
                    "country": True if "country" in df_map.columns else False,
                    "electrolyser_capacity": ":.3f",
                    "levelised_cost": ":.3f",
                    "carbon_intensity": ":.3f",
                    "latitude": ":.2f",
                    "longitude": ":.2f",
                },
                zoom=1,
                height=650,
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
                reg_solar_kw = float(df_reg["solar_costs_usd_kw"].mean()) if "solar_costs_usd_kw" in df_reg else 0.99
                reg_wind_kw = float(df_reg["wind_costs_usd_kw"].mean()) if "wind_costs_usd_kw" in df_reg else 1.50

                # convert back to USD/MW for existing CAPEX function baseline math
                reg_solar_mw = reg_solar_kw 
                reg_wind_mw = reg_wind_kw 

                # proxy regional electrolyser default from elec component (or tech default if preferred)
                default_elec = 1200.0 if tech == "alkaline" else 1400.0

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
                    f"Solar = **{reg_solar_kw:.2f} USD/kW**, Wind = **{reg_wind_kw:.2f} USD/kW**"
                )

                if st.button("Run Regional CAPEX Recalculation", type="primary"):
                    out = change_capex_absolute_df(
                        df=df_reg,
                        solar_capex=solar_capex,
                        wind_capex=wind_capex,
                        elec_capex=elec_capex,
                        initial_capex=initial_capex,
                    )
                    st.metric("Regional mean ΔLCOH", f"{out['delta_lcoh'].mean():.3f}")

                    fig = px.histogram(out.dropna(subset=["delta_lcoh"]), x="delta_lcoh", nbins=50, title=f"ΔLCOH distribution - {selected_region}")
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
            options=[VAR_LABELS["levelised_cost"], VAR_LABELS["levelised_cost_ren"], VAR_LABELS["levelised_cost_elec"]],
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
            comp = (
                df_sf.dropna(subset=["region"])
                .groupby("region", as_index=False)
                .agg(
                    lcoh_total=("levelised_cost", "mean"),
                    lcoh_ren=("levelised_cost_ren", "mean"),
                    lcoh_elec=("levelised_cost_elec", "mean"),
                )
            )

            comp_long = comp.melt(
                id_vars="region",
                value_vars=["lcoh_ren", "lcoh_elec"],
                var_name="component",
                value_name="value",
            )
            comp_long["component"] = comp_long["component"].map({
                "lcoh_ren": "Renewables component",
                "lcoh_elec": "Electrolyser component",
            })

            fig_comp = px.bar(
                comp_long.sort_values("region"),
                x="region",
                y="value",
                color="component",
                barmode="stack",
                title="Average LCOH component breakdown by region",
                labels={"value": "LCOH component (USD/kg)", "region": "Region"},
            )
            fig_comp.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig_comp, use_container_width=True)

            # Optional: relative to best region
            best = comp["lcoh_total"].min()
            comp["delta_vs_best"] = comp["lcoh_total"] - best

            fig_delta = px.bar(
                comp.sort_values("delta_vs_best"),
                x="region",
                y="delta_vs_best",
                title="LCOH premium vs best-performing region",
                labels={"delta_vs_best": "Δ LCOH vs best (USD/kg)", "region": "Region"},
            )
            fig_delta.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig_delta, use_container_width=True)

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