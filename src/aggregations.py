import pandas as pd

def country_summary(df: pd.DataFrame, lcoh_col: str = "levelised_cost", ci_col: str = "carbon_intensity") -> pd.DataFrame:
    valid = df.dropna(subset=["region", lcoh_col, ci_col]).copy()

    summary = (
        valid.groupby("region", as_index=False)
        .agg(
            n_points=("region", "size"),
            lcoh_mean=(lcoh_col, "mean"),
            lcoh_p10=(lcoh_col, lambda x: x.quantile(0.1)),
            lcoh_p90=(lcoh_col, lambda x: x.quantile(0.9)),
            ci_mean=(ci_col, "mean"),
            ci_p10=(ci_col, lambda x: x.quantile(0.1)),
            ci_p90=(ci_col, lambda x: x.quantile(0.9)),
            h2_potential_sum=("hydrogen_technical_potential", "sum"),
        )
        .sort_values("lcoh_mean")
    )
    return summary