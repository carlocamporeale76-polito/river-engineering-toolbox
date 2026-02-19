from __future__ import annotations
import numpy as np
import pandas as pd

def geometric_mean_Dsg(df_psd: pd.DataFrame) -> float:
    """D_sg = exp(sum f_i ln D_i)."""
    D = df_psd["D_i_m"].to_numpy(dtype=float)
    f = df_psd["f_i"].to_numpy(dtype=float)
    return float(np.exp(np.sum(f * np.log(D))))

def sorting_sigma_s_log2(df_psd: pd.DataFrame) -> float:
    """
    Surface sorting measure as std dev in psi scale:
    psi = log2(D / 1mm). Here we use D in meters, reference 1e-3 m.
    """
    D = df_psd["D_i_m"].to_numpy(dtype=float)
    f = df_psd["f_i"].to_numpy(dtype=float)
    psi = np.log2(D / 1e-3)
    mu = np.sum(f * psi)
    var = np.sum(f * (psi - mu) ** 2)
    return float(np.sqrt(var))
