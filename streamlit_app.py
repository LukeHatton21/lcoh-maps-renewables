import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.data_loader import load_points_slice
from src.aggregations import country_summary
from src.calculations import change_capex_absolute_df



st.set_page_config(page_title="Green H2 Cost & Carbon Explorer", layout="wide")

st.title("Green Hydrogen Cost & Carbon Explorer")
st.caption("Explore spatial variation in LCOH and carbon intensity for Alkaline and PEM pathways.")

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

# Load only selected slice from partitioned parquet
needed_cols = [
    "latitude",
    "longitude",
    "region",
    "levelised_cost",
    "levelised_cost_ren",
    "levelised_cost_elec",
    "solar_costs",
    "renewables_costs",
    "carbon_intensity",
    "carbon_intensity_low",
    "carbon_intensity_high",
    "hydrogen_technical_potential",
]
df_sf = load_points_slice(tech=tech, solar_fraction=solar_fraction, columns=needed_cols)

# remove rows where both key metrics missing
if not df_sf.empty:
    df_sf = df_sf.dropna(subset=["levelised_cost"], how="all")

for col, default in {
"region": "Unknown",
"carbon_intensity": np.nan,
"carbon_intensity_low": np.nan,
"carbon_intensity_high": np.nan,
"hydrogen_technical_potential": np.nan,
}.items():
    if col not in df_sf.columns:
        df_sf[col] = default

tab1, tab2, tab3 = st.tabs(["🗺️ Map Explorer", "🧮 CAPEX Sensitivity", "🌍 Country Summary"])


# ==========================================================
# TAB 1: MAP EXPLORER
# ==========================================================
with tab1:
    st.subheader("Spatial Explorer")

    if df_sf.empty:
        st.warning("No data found for this technology / solar fraction selection.")
    else:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            metric = st.selectbox(
                "Map metric",
                [
                    "levelised_cost",
                    "carbon_intensity",
                    "levelised_cost_ren",
                    "levelised_cost_elec",
                    "hydrogen_technical_potential",
                ],
                index=0,
            )
        with c2:
            color_scale = st.selectbox("Color scale", ["Viridis", "Turbo", "Plasma", "Cividis"], index=0)
        with c3:
            max_points = st.slider("Max points on map", 2000, 50000, 12000, step=1000)

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
                    "levelised_cost": ":.3f",
                    "carbon_intensity": ":.3f",
                    "hydrogen_technical_potential": ":.3f",
                    "latitude": ":.2f",
                    "longitude": ":.2f",
                },
                zoom=1,
                height=650,
            )
            fig.update_layout(mapbox_style="carto-positron", margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Quick stats")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Grid points", f"{len(df_map):,}")
            s2.metric("Mean LCOH", f"{df_map['levelised_cost'].mean():.3f}" if "levelised_cost" in df_map else "NA")
            s3.metric("Mean carbon intensity", f"{df_map['carbon_intensity'].mean():.3f}" if "carbon_intensity" in df_map else "NA")
            s4.metric("Median selected metric", f"{df_map[metric].median():.3f}")

            with st.expander("Show filtered data table"):
                st.dataframe(df_map.head(1000), use_container_width=True)


# ==========================================================
# TAB 2: CAPEX SENSITIVITY
# ==========================================================
with tab2:
    st.subheader("LCOH Sensitivity to CAPEX Assumptions")
    st.write("Adjust CAPEX assumptions and compare baseline vs recalculated LCOH.")

    if df_sf.empty:
        st.warning("No data found for this technology / solar fraction selection.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            solar_capex = st.number_input("Solar CAPEX", min_value=100.0, max_value=4000.0, value=990.0, step=10.0)
        with c2:
            wind_capex = st.number_input("Wind CAPEX", min_value=100.0, max_value=6000.0, value=1500.0, step=10.0)
        with c3:
            default_elec = 1200.0 if tech == "alkaline" else 1400.0
            elec_capex = st.number_input("Electrolyser CAPEX", min_value=100.0, max_value=6000.0, value=default_elec, step=10.0)
        with c4:
            initial_capex = st.number_input(
                "Initial electrolyser CAPEX (baseline)",
                min_value=100.0,
                max_value=6000.0,
                value=default_elec,
                step=10.0,
            )

        run = st.button("Run CAPEX Recalculation", type="primary")

        if run:
            out = change_capex_absolute_df(
                df=df_sf,
                solar_capex=solar_capex,
                wind_capex=wind_capex,
                elec_capex=elec_capex,
                initial_capex=initial_capex,
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Baseline mean LCOH", f"{out['levelised_cost'].mean():.3f}")
            m2.metric("Scenario mean LCOH", f"{out['Calculated_LCOH'].mean():.3f}")
            m3.metric("Mean ΔLCOH", f"{out['delta_lcoh'].mean():.3f}")
            m4.metric("Median ΔLCOH", f"{out['delta_lcoh'].median():.3f}")

            map_metric = st.radio("Display map of:", ["Calculated_LCOH", "delta_lcoh", "levelised_cost"], horizontal=True)

            df_plot = out.dropna(subset=[map_metric]).copy()
            if len(df_plot) > 15000:
                df_plot = df_plot.sample(15000, random_state=42)

            fig = px.scatter_mapbox(
                df_plot,
                lat="latitude",
                lon="longitude",
                color=map_metric,
                color_continuous_scale="Turbo",
                hover_data={
                    "region": True,
                    "levelised_cost": ":.3f",
                    "Calculated_LCOH": ":.3f",
                    "delta_lcoh": ":.3f",
                    "latitude": ":.2f",
                    "longitude": ":.2f",
                },
                zoom=1,
                height=600,
            )
            fig.update_layout(mapbox_style="carto-positron", margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

            c_left, c_right = st.columns(2)
            with c_left:
                hist1 = px.histogram(
                    out.dropna(subset=["levelised_cost", "Calculated_LCOH"]),
                    x=["levelised_cost", "Calculated_LCOH"],
                    nbins=60,
                    barmode="overlay",
                    opacity=0.6,
                    title="Baseline vs Scenario LCOH distribution",
                )
                st.plotly_chart(hist1, use_container_width=True)
            with c_right:
                hist2 = px.histogram(
                    out.dropna(subset=["delta_lcoh"]),
                    x="delta_lcoh",
                    nbins=60,
                    title="ΔLCOH distribution (Scenario - Baseline)",
                )
                st.plotly_chart(hist2, use_container_width=True)

            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download scenario results (CSV)",
                data=csv,
                file_name=f"{tech}_sf{solar_fraction}_capex_sensitivity.csv",
                mime="text/csv",
            )
        else:
            st.info("Set assumptions and click **Run CAPEX Recalculation**.")


# ==========================================================
# TAB 3: COUNTRY SUMMARY
# ==========================================================
with tab3:
    st.subheader("Country / Region Summary")

    if df_sf.empty:
        st.warning("No data found for this technology / solar fraction selection.")
    else:
        lcoh_field = st.selectbox(
            "LCOH field for summary",
            ["levelised_cost", "levelised_cost_ren", "levelised_cost_elec"],
            index=0,
        )

        summary = country_summary(df_sf, lcoh_col=lcoh_field, ci_col="carbon_intensity")

        if summary.empty:
            st.warning("No country-level data available for this selection.")
        else:
            c1, c2 = st.columns([2, 1])

            with c1:
                st.markdown("#### Country ranking table")
                st.dataframe(summary, use_container_width=True, hide_index=True)

            with c2:
                st.markdown("#### Top/Bottom performers")
                top_n = st.slider("N countries", 5, 30, 10)
                best = summary.nsmallest(top_n, "lcoh_mean")[["region", "lcoh_mean"]]
                worst = summary.nlargest(top_n, "lcoh_mean")[["region", "lcoh_mean"]]
                st.write("**Lowest mean LCOH**")
                st.dataframe(best, use_container_width=True, hide_index=True)
                st.write("**Highest mean LCOH**")
                st.dataframe(worst, use_container_width=True, hide_index=True)

            st.markdown("#### Country comparison charts")
            cc1, cc2 = st.columns(2)

            with cc1:
                fig_lcoh = px.bar(
                    summary.sort_values("lcoh_mean").head(30),
                    x="region",
                    y="lcoh_mean",
                    title="Lowest mean LCOH countries (top 30)",
                )
                fig_lcoh.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_lcoh, use_container_width=True)

            with cc2:
                fig_ci = px.bar(
                    summary.sort_values("ci_mean").head(30),
                    x="region",
                    y="ci_mean",
                    title="Lowest mean carbon intensity countries (top 30)",
                )
                fig_ci.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_ci, use_container_width=True)

            csv_sum = summary.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download country summary (CSV)",
                data=csv_sum,
                file_name=f"{tech}_sf{solar_fraction}_country_summary.csv",
                mime="text/csv",
            )