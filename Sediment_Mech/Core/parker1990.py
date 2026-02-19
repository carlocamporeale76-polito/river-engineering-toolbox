# core/parker1990.py

from __future__ import annotations
import numpy as np
import pandas as pd

from .active_layer import geometric_mean_Dsg, sorting_sigma_s_log2
from .transport_common import qb_from_Phi


# ---------------------------------------------------------------------
# Parker (1990) strain functions table (from the Excel sheet Strain_Functions)
# Columns: fsgo = tau_*sg / 0.0386, omega0, sigma0
# ---------------------------------------------------------------------
_FSGO = np.array([
    0.6684, 0.7639, 0.8601, 0.9096, 0.9615, 1.0000, 1.0550, 1.1080, 1.1970,
    1.3020, 1.4070, 1.5290, 1.6410, 1.7020, 1.8320, 2.0610, 2.3490, 2.6390,
    3.1320, 3.6290, 4.4930, 5.3240, 6.1880, 7.0120, 8.0600, 9.0850, 10.29,
    11.60, 13.16, 15.20, 17.40, 21.73, 38.57, 68.74, 91.95, 231.20, 2320.0
], dtype=float)

_OMEGA0 = np.array([
    1.0110, 1.0110, 1.0100, 1.0080, 1.0040, 0.9997, 0.9903, 0.9789, 0.9567,
    0.9273, 0.8964, 0.8604, 0.8287, 0.8123, 0.7796, 0.7262, 0.6645, 0.6145,
    0.5442, 0.5043, 0.4716, 0.4632, 0.4595, 0.4576, 0.4562, 0.4553, 0.4547,
    0.4544, 0.4541, 0.4539, 0.4537, 0.4532, 0.4620, 0.4578, 0.4564, 0.4541,
    0.4527
], dtype=float)

_SIGMA0 = np.array([
    0.8157, 0.8157, 0.8182, 0.8233, 0.8333, 0.8439, 0.8621, 0.8825, 0.9214,
    0.9723, 1.0250, 1.0830, 1.1300, 1.1530, 1.1960, 1.2580, 1.3020, 1.3380,
    1.3850, 1.4130, 1.4400, 1.4510, 1.4590, 1.4650, 1.4720, 1.4760, 1.4800,
    1.4840, 1.4870, 1.4900, 1.4920, 1.4960, 1.4930, 1.4970, 1.4980, 1.4990,
    1.5000
], dtype=float)


def _interp_strain_functions(tau_star_sg: float) -> tuple[float, float]:
    """
    Interpolate Parker's reference strain functions omega0 and sigma0 as a function of
    fsgo = tau_*sg / 0.0386, using linear interpolation with clamping.
    """
    if tau_star_sg < 0:
        raise ValueError("tau_star_sg must be >= 0.")

    fsgo = tau_star_sg / 0.0386
    # Clamp to table bounds
    fsgo_clamped = float(np.clip(fsgo, _FSGO.min(), _FSGO.max()))

    omega0 = float(np.interp(fsgo_clamped, _FSGO, _OMEGA0))
    sigma0 = float(np.interp(fsgo_clamped, _FSGO, _SIGMA0))
    return omega0, sigma0


def compute_omega(sigma_s: float, tau_star_sg: float) -> float:
    """
    Parker (1990): omega = 1 + (sigma_s / sigma0(tau_*sg)) * (omega0(tau_*sg) - 1)
    """
    omega0, sigma0 = _interp_strain_functions(tau_star_sg)
    return float(1.0 + (sigma_s / sigma0) * (omega0 - 1.0))


def phi_i(D_i: float, D_sg: float, tau_star_sg: float, omega: float) -> float:
    """
    phi_i = omega * (tau_*sg / 0.0386) * (D_i / D_sg)^(-0.0951)
    """
    if D_i <= 0 or D_sg <= 0:
        raise ValueError("D_i and D_sg must be > 0.")
    return float(omega * (tau_star_sg / 0.0386) * (D_i / D_sg) ** (-0.0951))


def Phi_parker(tau_star_i: float, phi: float) -> float:
    """
    Parker (1990) surface-based gravel relation (as in your slides):
      Phi_i = tau_*i^3 * f(phi_i)
    """
    if tau_star_i < 0:
        raise ValueError("tau_star_i must be >= 0.")
    if phi <= 1.0:
        return 0.0

    if phi > 1.59:
        return float(tau_star_i**3 * 1.2513 * (1.0 - 0.853 / phi) ** 4.5)

    # 1 < phi < 1.59
    x = (phi - 1.0)
    return float(tau_star_i**3 * np.exp(14.2 * x - 9.28 * x**2))


def remove_sand_and_renormalize(df_psd: pd.DataFrame, D_sand: float = 0.002) -> tuple[pd.DataFrame, float]:
    """
    Remove sand from surface PSD (D < D_sand) and renormalize remaining fractions.
    Returns (df_gravel, sand_fraction_removed).
    """
    df = df_psd.copy()
    sand_mask = df["D_i_m"] < D_sand
    sand_fraction = float(df.loc[sand_mask, "f_i"].sum())

    df_gravel = df.loc[~sand_mask].copy()
    if df_gravel.empty:
        raise ValueError("All classes are sand (<2 mm). Parker gravel relation not applicable.")

    s = df_gravel["f_i"].sum()
    df_gravel["f_i"] = df_gravel["f_i"] / s
    return df_gravel, sand_fraction


def compute_parker1990(
    df_psd: pd.DataFrame,
    tau_star_sg: float,
    tau_star_i: pd.Series | None,
    g: float,
    Delta: float,
    remove_sand: bool = True,
    D_sand: float = 0.002
) -> dict:
    """
    Compute Parker (1990) class transport and totals.

    Inputs
    ------
    df_psd: DataFrame with columns D_i_m, f_i (surface PSD)
    tau_star_sg: reference Shields stress computed with D_sg (or provided)
    tau_star_i: optional Series of tau_*i per class. If None, tau_*i = tau_*sg for all classes.
    remove_sand: if True, remove D < 2 mm from PSD and renormalize before applying Parker gravel relation.

    Returns
    -------
    dict with keys:
      - results_df (per-class table)
      - q_b_tot (total gravel bedload, m^2/s)
      - D_sg (geometric mean used)
      - sigma_s (sorting measure used)
      - omega (mobility correction)
      - sand_fraction_removed (if remove_sand True else 0)
    """
    if tau_star_sg < 0:
        raise ValueError("tau_star_sg must be >= 0.")

    if remove_sand:
        df_use, sand_frac = remove_sand_and_renormalize(df_psd, D_sand=D_sand)
    else:
        df_use = df_psd.copy()
        sand_frac = 0.0

    D_sg = geometric_mean_Dsg(df_use)
    sigma_s = sorting_sigma_s_log2(df_use)
    omega = compute_omega(sigma_s, tau_star_sg)

    if tau_star_i is None:
        tau_star_i = pd.Series([tau_star_sg] * len(df_use), index=df_use.index, dtype=float)
    else:
        tau_star_i = tau_star_i.astype(float)

    results = []
    q_tot = 0.0

    for idx, row in df_use.iterrows():
        D_i = float(row["D_i_m"])
        f_i = float(row["f_i"])
        tsi = float(tau_star_i.loc[idx])

        phi = phi_i(D_i, D_sg, tau_star_sg, omega)
        Phi = Phi_parker(tsi, phi)
        q_bi = qb_from_Phi(Phi, f_i, g, Delta, D_i)
        q_tot += q_bi

        results.append({
            "D_i_m": D_i,
            "f_i": f_i,
            "tau_star_i": tsi,
            "tau_star_sg": float(tau_star_sg),
            "phi_i": phi,
            "Phi_i": Phi,
            "q_bi_m2s": q_bi
        })

    results_df = pd.DataFrame(results)

    return {
        "results_df": results_df,
        "q_b_tot": float(q_tot),
        "D_sg": float(D_sg),
        "sigma_s": float(sigma_s),
        "omega": float(omega),
        "sand_fraction_removed": float(sand_frac),
    }
