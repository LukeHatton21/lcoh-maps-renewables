from pathlib import Path
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "cache_partitioned"

TECH_FILES = {
    "alkaline": DATA_DIR / "alkaline.nc",
    "pem": DATA_DIR / "pem.nc",
}

KEEP_COLS = [
    "solar_fraction",
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

def cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    float_cols = [
        "latitude",
        "longitude",
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
    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].astype("float32")

    if "solar_fraction" in df.columns:
        df["solar_fraction"] = df["solar_fraction"].astype("int16")

    if "region" in df.columns:
        df["region"] = df["region"].astype("category")

    return df

def build_for_tech(tech: str, nc_path: Path):
    print(f"\nBuilding cache for {tech}: {nc_path}")
    ds = xr.open_dataset(nc_path)

    vars_present = [v for v in KEEP_COLS if (v in ds.data_vars or v in ds.coords)]
    dss = ds[vars_present]

    df = dss.to_dataframe().reset_index()

    # keep rows with at least one key metric
    key = [c for c in ["levelised_cost", "carbon_intensity"] if c in df.columns]
    if key:
        df = df.dropna(subset=key, how="all")

    df = cast_dtypes(df)

    out_path = OUT_DIR / f"tech={tech}"
    out_path.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        out_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
        partition_cols=["solar_fraction"],
    )
    print(f"Saved partitioned parquet under: {out_path}")

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for tech, path in TECH_FILES.items():
        build_for_tech(tech, path)
    print("\nDone.")