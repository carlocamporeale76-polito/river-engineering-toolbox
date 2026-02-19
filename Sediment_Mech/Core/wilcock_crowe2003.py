# core/wilcock_crowe2003.py

from __future__ import annotations
import numpy as np
import pandas as pd

from .active_layer import geometric_mean_Dsg
from .transport_common import qb_from_Phi


def surface_sand_fraction(df_psd: pd.DataFrame, D_sand: float = 0.002) -> float:
    """Fs = sum f_i for D_i < 2 mm (default)."""
    return float(df_psd.loc[df_psd["D_i_m"] < D_sand, "f_i"].sum())


def tau_star_rm(Fs: float) -> float:
    """
    Wilcock & Crowe (2003), Eq. (6):
      tau*_rm = 0.021 + 0.015 exp(-20 Fs)
    Fs must be a fraction in [0,1], not percent.
    """
    if Fs < 0 or Fs > 1:
        raise ValueError("Fs must be a fraction in [0,1].")
    return float(0.021 + 0.015 * np.exp(-20.0 * Fs))


def hiding_exponent_b(D_i: float, D_sm: float) -> float:
    """
    Wilcock & Crowe (2003), Eq. (4):
      b = 0.67 / (1 + exp(1.5 - D_i/D_sm))
    """
    x = D_i / D_sm
    return float(0.67 / (1.0 + np.exp(1.5 - x)))


def tau_ri(tau_rm: float, D_i: float, D_sm: float) -> float:
    """
    Hiding function:
      tau_ri = tau_rm * (D_i/D_sm)^b
    with b from Eq. (4).
    """
    b = hiding_exponent_b(D_i, D_sm)
    return float(tau_rm * (D_i / D_sm) ** b)


def Wstar_transport(f: float, extended_low_f: bool = True) -> float:
    """
    Wilcock & Crowe (2003), Eq. (7):
      W* = 0.002 f^7.5                  for f < 1.35
      W* = 14 (1 - 0.894 / f^0.5)^4.5   for f > 1.35

    Your slide set adds an extra low-f branch (for f < 1):
      W* = 0.00218 f^14.2               for f < 1
    which can be enabled via extended_low_f=True (default).
    """
    if f <= 0:
        return 0.0

    if extended_low_f and f < 1.0:
        return float(0.00218 * f ** 14.2)

    if f < 1.35:
        return float(0.002 * f ** 7.5)

    return float(14.0 * (1.0 - 0.894 / np.sqrt(f)) ** 4.5)


def compute_wilcock_crowe2003(
    df_psd: pd.DataFrame,
    tau: float,
    rho: float,
    rho_s: float,
    g: float,
    D_sand: float = 0.002,
    extended_low_f: bool = True
) -> dict:
    """
    Full Wilcock & Crowe (2003) surface-based bedload model.

    Inputs
    ------
    df_psd : DataFrame with columns D_i_m, f_i (surface distribution; includes sand)
    tau    : bed shear stress (Pa) (same for all sizes; from hydraulics)
    rho, rho_s, g : physical constants
    D_sand : sand threshold (m), default 0.002 m
    extended_low_f : include extra branch for f<1 as used in your slides

    Returns
    -------
    dict with keys:
      - results_df
      - q_b_tot
      - D_sm (geometric mean size used)
      - Fs (surface sand fraction)
      - tau_star_rm
      - tau_rm (Pa)
    """
    if tau < 0:
        raise ValueError("tau must be >= 0.")
    if rho <= 0 or rho_s <= rho:
        raise ValueError("Require rho > 0 and rho_s > rho.")

    Delta = (rho_s - rho) / rho

    # In WC2003 the model uses the full surface distribution (including sand)
    D_sm = geometric_mean_Dsg(df_psd)

    Fs = surface_sand_fraction(df_psd, D_sand=D_sand)
    tstar_rm = tau_star_rm(Fs)
    tau_rm = tstar_rm * (rho_s - rho) * g * D_sm

    results = []
    q_tot = 0.0

    for idx, row in df_psd.iterrows():
        D_i = float(row["D_i_m"])
        f_i = float(row["f_i"])

        tau_ref_i = tau_ri(tau_rm, D_i, D_sm)
        f = tau / tau_ref_i if tau_ref_i > 0 else 0.0

        Wstar = Wstar_transport(f, extended_low_f=extended_low_f)

        # Use the same structure as Einstein number: Phi_i := W*_i
        q_bi = qb_from_Phi(Wstar, f_i, g, Delta, D_i)
        q_tot += q_bi

        results.append({
            "D_i_m": D_i,
            "f_i": f_i,
            "tau_Pa": float(tau),
            "D_sm_m": float(D_sm),
            "Fs": float(Fs),
            "tau_star_rm": float(tstar_rm),
            "tau_rm_Pa": float(tau_rm),
            "tau_ri_Pa": float(tau_ref_i),
            "f_ratio": float(f),
            "Wstar_i": float(Wstar),
            "q_bi_m2s": float(q_bi),
        })

    results_df = pd.DataFrame(results)

    return {
        "results_df": results_df,
        "q_b_tot": float(q_tot),
        "D_sm": float(D_sm),
        "Fs": float(Fs),
        "tau_star_rm": float(tstar_rm),
        "tau_rm": float(tau_rm),
    }
