from __future__ import annotations
import pandas as pd

def read_psd_surface_csv(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    required = {"D_i_m", "f_i"}
    if not required.issubset(df.columns):
        raise ValueError(f"PSD CSV must contain columns {sorted(required)}")
    df = df.copy()
    if (df["D_i_m"] <= 0).any():
        raise ValueError("All D_i_m must be > 0.")
    if (df["f_i"] < 0).any():
        raise ValueError("All f_i must be >= 0.")
    s = df["f_i"].sum()
    if s <= 0:
        raise ValueError("Sum of f_i must be > 0.")
    if abs(s - 1.0) > 1e-6:
        df["f_i"] = df["f_i"] / s
        df.attrs["normalized_f_i"] = True
        df.attrs["original_sum_f_i"] = float(s)
    else:
        df.attrs["normalized_f_i"] = False
        df.attrs["original_sum_f_i"] = float(s)
    return df

def read_fdc_csv(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    if "Q_m3s" not in df.columns:
        raise ValueError("FDC CSV must contain column Q_m3s.")
    if (df["Q_m3s"] < 0).any():
        raise ValueError("All Q_m3s must be >= 0.")
    df = df.copy()

    if "p_time" in df.columns:
        if (df["p_time"] < 0).any():
            raise ValueError("All p_time must be >= 0.")
        s = df["p_time"].sum()
        if s <= 0:
            raise ValueError("Sum of p_time must be > 0.")
        df["p_time"] = df["p_time"] / s  # normalize
        return df[["Q_m3s", "p_time"]]

    if "days_per_year" in df.columns:
        if (df["days_per_year"] < 0).any():
            raise ValueError("All days_per_year must be >= 0.")
        total_days = df["days_per_year"].sum()
        if total_days <= 0:
            raise ValueError("Sum of days_per_year must be > 0.")
        df["p_time"] = df["days_per_year"] / total_days
        return df[["Q_m3s", "p_time"]]

    raise ValueError("FDC CSV must contain either p_time or days_per_year.")
