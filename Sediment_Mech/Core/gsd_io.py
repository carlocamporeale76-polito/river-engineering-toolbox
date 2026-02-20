# Sediment_Mech/Core/gsd_io.py
from __future__ import annotations
from typing import Tuple, Dict, Any
import os
import numpy as np
import pandas as pd

def read_gsd_xlsx(path_or_buffer) -> pd.DataFrame:
    """
    Read a GSD Excel file and return DataFrame with columns:
    ['class_id', 'D_i_m', 'f_i'] where f_i are normalized to sum=1.
    Accepts various column names (D_mm, D_i_m, diameter, percent, fraction, f_i).
    """
    xls = pd.ExcelFile(path_or_buffer, engine="openpyxl")
    # prefer a sheet named like 'GSD' or 'Calculator' or first sheet
    names_lower = [s.lower() for s in xls.sheet_names]
    preferred = None
    for candidate in ("gsd", "surface", "psd", "calculator", "sheet1"):
        if candidate in names_lower:
            preferred = xls.sheet_names[names_lower.index(candidate)]
            break
    sheet = preferred or xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet)

    # normalize column names heuristically
    colmap = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in ("class", "class_id", "id", "bin"):
            colmap[c] = "class_id"
        elif lc in ("d", "di", "di_m", "d_i_m", "diameter_m", "diameter"):
            colmap[c] = "D_i_m"
        elif "mm" in lc and ("d" in lc or "diameter" in lc):
            colmap[c] = "D_i_mm"
        elif lc in ("f", "fi", "f_i", "fraction", "surface_fraction", "percent", "percentile"):
            colmap[c] = "f_i"

    df = df.rename(columns=colmap)

    # convert mm to meters if needed
    if "D_i_mm" in df.columns and "D_i_m" not in df.columns:
        df["D_i_m"] = pd.to_numeric(df["D_i_mm"], errors="coerce") / 1000.0

    # require at least diameter and fraction
    if not {"D_i_m", "f_i"}.issubset(df.columns):
        raise ValueError(f"GSD file must contain diameter and fraction columns. Found: {df.columns.tolist()}")

    # coerce numeric, drop NaNs
    df = df[["D_i_m", "f_i"]].copy()
    df["D_i_m"] = pd.to_numeric(df["D_i_m"], errors="coerce")
    df["f_i"] = pd.to_numeric(df["f_i"], errors="coerce")
    df = df.dropna(subset=["D_i_m", "f_i"]).reset_index(drop=True)

    # normalize fractions
    s = df["f_i"].sum()
    if s <= 0:
        raise ValueError("Sum of fractions <= 0 in GSD file.")
    df["f_i"] = df["f_i"] / s

    # sort by diameter ascending (fine -> coarse)
    df = df.sort_values("D_i_m", ascending=True).reset_index(drop=True)

    # assign class ids
    df.insert(0, "class_id", range(1, len(df) + 1))

    return df


def cumulative_from_gsd(df: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    """
    Given df with columns ['class_id','D_i_m','f_i'], return a copy with:
    - cumulative (from fine to coarse if ascending=True) as cumulative fraction [0..1]
    - cumulative_percent (0..100)
    """
    dfc = df.copy().reset_index(drop=True)
    # ensure sorted ascending by D (fine to coarse)
    dfc = dfc.sort_values("D_i_m", ascending=ascending).reset_index(drop=True)
    dfc["cum_f"] = dfc["f_i"].cumsum()
    dfc["cum_percent"] = dfc["cum_f"] * 100.0
    return dfc


def interp_percentile(df: pd.DataFrame, p: float, diameter_col: str = "D_i_m") -> float:
    """
    Interpolate to find D_p: the diameter at percentile p (0..100).
    Uses linear interpolation on the cumulative percent curve.
    p is in percent (e.g., 50 for D50).
    Returns diameter in same units as diameter_col.
    """
    if "cum_percent" not in df.columns:
        dfc = cumulative_from_gsd(df)
    else:
        dfc = df.copy().sort_values(diameter_col, ascending=True).reset_index(drop=True)

    x = dfc["cum_percent"].values
    y = dfc[diameter_col].values

    # edge cases
    if p <= x.min():
        return float(y[0])
    if p >= x.max():
        return float(y[-1])

    # numpy.interp requires x to be increasing; cum_percent is non-decreasing
    return float(np.interp(p, x, y))


def diameters_at_percentiles(df: pd.DataFrame, percents=(16, 50, 84, 75, 90, 10, 5, 95)) -> Dict[int, float]:
    """
    Return a dict {percent: D_percent} for the requested percents.
    """
    dfc = cumulative_from_gsd(df)
    results = {}
    for p in percents:
        try:
            results[p] = interp_percentile(dfc, p)
        except Exception:
            results[p] = float("nan")
    return results


def phi_from_d(d_m: float) -> float:
    """Convert diameter in meters to phi scale: phi = -log2(D) with D in mm convention we usually use phi = -log2(D_mm) but here D_m -> convert to mm first."""
    if d_m <= 0 or np.isnan(d_m):
        return float("nan")
    d_mm = d_m * 1000.0
    return -np.log2(d_mm)


def d_from_phi(phi: float) -> float:
    """Convert phi back to meters (D_m)."""
    d_mm = 2 ** (-phi)
    return d_mm / 1000.0


def stats_in_phi(df: pd.DataFrame, percents=(16, 50, 84, 5, 95)) -> Dict[str, Any]:
    """
    Compute common granulometric stats in phi-space:
    - D16, D50, D84 (meters)
    - phi_mean (weighted by f_i)
    - phi_std (standard deviation in phi)
    - geometric_mean (meters)
    - folk_ward_sorting (graphic standard deviation using Folk & Ward formula)
    Returns a dict.
    """
    # cumulative needed for percentile interpolation
    dfc = cumulative_from_gsd(df)

    # get diameters at percentiles
    per_vals = diameters_at_percentiles(dfc, percents)
    D16 = per_vals.get(16, np.nan)
    D50 = per_vals.get(50, np.nan)
    D84 = per_vals.get(84, np.nan)
    D5 = per_vals.get(5, np.nan)
    D95 = per_vals.get(95, np.nan)

    # convert to phi
    phi_vals = {}
    for k, v in (("D16", D16), ("D50", D50), ("D84", D84), ("D5", D5), ("D95", D95)):
        phi_vals[k.replace("D", "phi")] = phi_from_d(v) if not np.isnan(v) else np.nan

    # weighted mean phi (weights = f_i)
    # compute phi for each class center and weight by f_i
    df_phi = df.copy()
    df_phi["phi"] = df_phi["D_i_m"].apply(phi_from_d)
    # drop NaN phis if any
    df_phi = df_phi.dropna(subset=["phi", "f_i"])
    if df_phi["f_i"].sum() <= 0:
        phi_mean = float("nan")
        phi_std = float("nan")
    else:
        w = df_phi["f_i"].values
        phi_vals_arr = df_phi["phi"].values
        phi_mean = float(np.average(phi_vals_arr, weights=w))
        # unbiased weighted std
        # compute variance = sum(w*(x - mean)^2) / sum(w)
        phi_var = float(np.sum(w * (phi_vals_arr - phi_mean) ** 2) / np.sum(w))
        phi_std = float(np.sqrt(phi_var))

    # geometric mean Dg = exp( sum(w * ln(D)) ) where w fractions sum to 1
    D_vals = df["D_i_m"].values
    w_vals = df["f_i"].values
    # protective checks
    mask = (D_vals > 0) & (~np.isnan(D_vals)) & (~np.isnan(w_vals))
    if mask.sum() == 0 or w_vals.sum() <= 0:
        geometric_mean = float("nan")
    else:
        lnD = np.log(D_vals[mask])
        geometric_mean = float(np.exp(np.sum(w_vals[mask] * lnD) / np.sum(w_vals[mask])))

    # Folk & Ward graphic sorting estimate:
    # sigma_phi_FW = (phi84 - phi16)/4 + (phi95 - phi5)/6.6
    phi16 = phi_vals.get("phi16", np.nan)
    phi84 = phi_vals.get("phi84", np.nan)
    phi5 = phi_vals.get("phi5", np.nan)
    phi95 = phi_vals.get("phi95", np.nan)
    folk_ward_sort = float(np.nan)
    try:
        folk_ward_sort = float(((phi84 - phi16) / 4.0) + ((phi95 - phi5) / 6.6))
    except Exception:
        folk_ward_sort = float("nan")

    return {
        "D16_m": D16,
        "D50_m": D50,
        "D84_m": D84,
        "phi16": phi16,
        "phi50": phi_vals.get("phi50", np.nan),
        "phi84": phi84,
        "phi5": phi5,
        "phi95": phi95,
        "phi_mean": phi_mean,
        "phi_std": phi_std,
        "geometric_mean_m": geometric_mean,
        "folk_ward_sort": folk_ward_sort,
    }


def gsd_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with the original classes and added derived columns:
    ['class_id','D_i_m','f_i','cum_f','cum_percent','phi']
    """
    dfc = cumulative_from_gsd(df)
    dfc = dfc.rename(columns={"cum_f": "cum_f", "cum_percent": "cum_percent"})
    dfc["phi"] = dfc["D_i_m"].apply(phi_from_d)
    return dfc[["class_id", "D_i_m", "f_i", "cum_f", "cum_percent", "phi"]].copy()


def compute_all_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience wrapper to compute summary table and stats dictionary.
    """
    table = gsd_summary_table(df)
    stats = stats_in_phi(df)
    # attach Dxx values as well
    percentiles = diameters_at_percentiles(df, percents=(5,10,16,50,75,84,90,95))
    # convert percentiles to phi too
    p_phi = {f"phi_p{p}": (phi_from_d(percentiles[p]) if not np.isnan(percentiles[p]) else np.nan) for p in percentiles}
    out = {"table": table, "stats": stats, "percentiles_m": percentiles, "percentiles_phi": p_phi}
    return out
