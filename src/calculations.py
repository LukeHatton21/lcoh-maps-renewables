import numpy as np
import pandas as pd

def change_capex_absolute_df(
    df: pd.DataFrame,
    solar_capex: float,
    wind_capex: float,
    elec_capex: float,
    initial_capex: float,
) -> pd.DataFrame:
    """
    Pandas version of your CAPEX sensitivity method.
    """
    out = df.copy()

    needed = [
        "levelised_cost",
        "levelised_cost_ren",
        "levelised_cost_elec",
        "solar_costs",
        "renewables_costs",
    ]
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns for CAPEX calculation: {missing}")

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