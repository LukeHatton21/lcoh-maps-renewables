from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st
from .config import CACHE_PARTITIONED_DIR

def _find_partition_path(tech: str, solar_fraction: int) -> Path | None:
    tech = str(tech).strip().lower()
    tech_dir = CACHE_PARTITIONED_DIR / f"tech={tech}"
    if not tech_dir.exists():
        return None

    # Try common names first
    candidates = [
        tech_dir / f"solar_fraction={int(solar_fraction)}",
        tech_dir / f"solar_fraction={float(solar_fraction)}",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Fallback scan
    for d in tech_dir.iterdir():
        if d.is_dir() and d.name.startswith("solar_fraction="):
            raw = d.name.split("=", 1)[1]
            try:
                if int(float(raw)) == int(solar_fraction):
                    return d
            except Exception:
                pass
    return None

def _available_columns_in_partition(partition_path: Path) -> list[str]:
    files = sorted(partition_path.glob("*.parquet"))
    if not files:
        return []
    schema = pq.read_schema(files[0])
    return schema.names

@st.cache_data(show_spinner=False)
def load_points_slice(tech: str, solar_fraction: int, columns: list[str] | None = None) -> pd.DataFrame:
    part = _find_partition_path(tech, solar_fraction)
    if part is None:
        return pd.DataFrame()

    avail = _available_columns_in_partition(part)
    if not avail:
        # Try reading directory directly as fallback
        try:
            df = pd.read_parquet(part)
            return df
        except Exception:
            return pd.DataFrame()

    read_cols = None
    if columns is not None:
        read_cols = [c for c in columns if c in avail]
        if len(read_cols) == 0:
            return pd.DataFrame()

    files = sorted(part.glob("*.parquet"))
    if not files:
        try:
            df = pd.read_parquet(part, columns=read_cols)
            return df
        except Exception:
            return pd.DataFrame()

    dfs = [pd.read_parquet(f, columns=read_cols) for f in files]
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    if "region" in df.columns and str(df["region"].dtype) != "category":
        df["region"] = df["region"].astype("category")

    return df