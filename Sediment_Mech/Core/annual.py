# core/annual.py

from __future__ import annotations
import numpy as np
import pandas as pd

from .hydraulics import depth_from_manning, shear_stress
from .ashida_michiue import compute_ashida_michiue
from .parker1990 import compute_parker1990
from .wilcock_crowe2003 import compute_wilcock_crowe2003


SECONDS_PER_YEAR = 365 * 24 * 3600


def compute_annual_bedload(
    df_psd: pd.DataFrame,
    df_fdc: pd.DataFrame,
    model: str,
    rho: float,
    rho_s: float,
    g: float,
    S: float,
    width: float,
    n_manning: float,
    D_sand: float = 0.002
) -> dict:
    """
    Compute annual bedload transport using flow duration curve.

    model: "ashida", "parker", or "wilcock"
    """

    Delta = (rho_s - rho) / rho
    results = []
    annual_total = 0.0

    for _, row in df_fdc.iterrows():

        Q = float(row["Q_m3s"])
        p_time = float(row["p_time"])

        # compute flow depth from Manning (wide channel assumption)
        h = depth_from_manning(Q, n_manning, width, S)

        # bed shear stress
        tau = shear_stress(rho, g, h, S)

        if model == "ashida":
            # compute tau_star per class
            tau_star_i = {}
            for idx, r_psd in df_psd.iterrows():
                D_i = float(r_psd["D_i_m"])
                tau_star_i[idx] = tau / ((rho_s - rho) * g * D_i)

            res_df, q_b_tot = compute_ashida_michiue(
                df_psd, tau_star_i, g, Delta
            )

        elif model == "parker":
            # use tau_star_sg computed with D_sg inside function
            # need tau_star_sg:
            # approximate with geometric mean first
            from .active_layer import geometric_mean_Dsg
            D_sg = geometric_mean_Dsg(df_psd)
            tau_star_sg = tau / ((rho_s - rho) * g * D_sg)

            out = compute_parker1990(
                df_psd=df_psd,
                tau_star_sg=tau_star_sg,
                tau_star_i=None,
                g=g,
                Delta=Delta,
                remove_sand=True,
                D_sand=D_sand
            )

            res_df = out["results_df"]
            q_b_tot = out["q_b_tot"]

        elif model == "wilcock":
            out = compute_wilcock_crowe2003(
                df_psd=df_psd,
                tau=tau,
                rho=rho,
                rho_s=rho_s,
                g=g,
                D_sand=D_sand,
                extended_low_f=True
            )

            res_df = out["results_df"]
            q_b_tot = out["q_b_tot"]

        else:
            raise ValueError("Model must be 'ashida', 'parker', or 'wilcock'.")

        # integrate annual load
        annual_contribution = q_b_tot * p_time * SECONDS_PER_YEAR
        annual_total += annual_contribution

        results.append({
            "Q_m3s": Q,
            "h_m": h,
            "tau_Pa": tau,
            "q_b_tot_m2s": q_b_tot,
            "p_time": p_time,
            "annual_contribution_m3_per_m": annual_contribution
        })

    results_df = pd.DataFrame(results)

    return {
        "results_df": results_df,
        "annual_bedload_m3_per_m": annual_total
    }
