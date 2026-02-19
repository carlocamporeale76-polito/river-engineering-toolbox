# core/ashida_michiue.py

from __future__ import annotations
import numpy as np
import pandas as pd
from .transport_common import qb_from_Phi
from .active_layer import geometric_mean_Dsg

def tau_star_c_egiazaroff(D_i: float, D_sg: float) -> float:
    """
    Modified Egiazaroff (1965) hiding function used by Ashida & Michiue (1972).
    """
    ratio = D_i / D_sg
    if ratio < 0.4:
        return 0.0421 * (D_sg / D_i)
    else:
        return 0.433 / (np.log(19.0 * ratio) ** 2)

def Phi_ashida_michiue(tau_star_i: float, tau_star_ci: float) -> float:
    """
    Einstein number according to Ashida & Michiue (1972).
    """
    if tau_star_i <= tau_star_ci:
        return 0.0
    return 17.0 * (tau_star_i - tau_star_ci) * (
        np.sqrt(tau_star_i) - np.sqrt(tau_star_ci)
    )

def compute_ashida_michiue(
    df_psd: pd.DataFrame,
    tau_star_dict: dict,
    g: float,
    Delta: float
) -> tuple[pd.DataFrame, float]:
    """
    Compute class-specific and total bedload transport.

    Parameters
    ----------
    df_psd : DataFrame with columns D_i_m, f_i
    tau_star_dict : dict mapping class index to tau*_i
    g : gravitational acceleration
    Delta : submerged specific density

    Returns
    -------
    results_df : DataFrame with D_i, f_i, tau*_i, tau*_ci, Phi_i, q_bi
    q_b_tot : total bedload transport per unit width (m^2/s)
    """

    D_sg = geometric_mean_Dsg(df_psd)

    results = []
    q_b_tot = 0.0

    for idx, row in df_psd.iterrows():
        D_i = float(row["D_i_m"])
        f_i = float(row["f_i"])
        tau_star_i = float(tau_star_dict[idx])

        tau_star_ci = tau_star_c_egiazaroff(D_i, D_sg)
        Phi_i = Phi_ashida_michiue(tau_star_i, tau_star_ci)

        q_bi = qb_from_Phi(Phi_i, f_i, g, Delta, D_i)
        q_b_tot += q_bi

        results.append({
            "D_i_m": D_i,
            "f_i": f_i,
            "tau_star_i": tau_star_i,
            "tau_star_ci": tau_star_ci,
            "Phi_i": Phi_i,
            "q_bi_m2s": q_bi
        })

    results_df = pd.DataFrame(results)
    return results_df, q_b_tot
