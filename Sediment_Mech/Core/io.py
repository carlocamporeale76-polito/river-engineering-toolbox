from __future__ import annotations
from typing import Union, IO
import pandas as pd
import numpy as np
import io

# ---------- PSD / GSD readers ----------------------------------------------

def _ensure_df_has_columns(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def read_gsd_xlsx(path_or_buffer: Union[str, bytes, IO]) -> pd.DataFrame:
    """
    Read an Excel workbook exported by the original GSD tool and return a
    surface PSD DataFrame with columns: D_i_m, f_i (fractions normalized to sum=1).
    Accepts: filename, file-like, or bytes (as returned by Streamlit uploader).
    Behaviour:
      - Looks for a sheet named 'GSD' or 'Sheet1' or uses the first sheet.
      - Accepts size column named D_i_m (metres) or D_i_mm (millimetres).
      - Ensures columns D_i_m (m) and f_i (fraction) are present and normalized.
    """
    # Accept bytes buffer from streamlit file_uploader
    if isinstance(path_or_buffer, (bytes, bytearray)):
        buf = io.BytesIO(path_or_buffer)
    else:
        buf = path_or_buffer

    xls = pd.read_excel(buf, sheet_name=None)
    # Choose sheet
    sheet_keys = list(xls.keys())
    preferred = None
    for name in ["GSD", "Gsd", "surface", "Sheet1"]:
        if name in sheet_keys:
            preferred = name
            break
    if preferred is None:
        preferred = sheet_keys[0]
    df = pd.read_excel(buf, sheet_name=preferred)

    # normalize column names (strip / lower)
    df.columns = [c.strip() for c in df.columns]

    # Accept D_i_m or D_i_mm
    if "D_i_m" in df.columns:
        Dcol = "D_i_m"
    elif "D_i_mm" in df.columns:
        # convert mm to m
        df["D_i_m"] = df["D_i_mm"].astype(float) / 1000.0
        Dcol = "D_i_m"
    elif "D(mm)" in df.columns:
        df["D_i_m"] = df["D(mm)"].astype(float) / 1000.0
        Dcol = "D_i_m"
    else:
        raise ValueError("GSD spreadsheet must contain a diameter column named 'D_i_m' or 'D_i_mm' or 'D(mm)'")

    # fraction column
    if "f_i" not in df.columns and "f" in df.columns:
        df = df.rename(columns={"f": "f_i"})
    _ensure_df_has_columns(df, ["D_i_m", "f_i"])

    df = df[["D_i_m", "f_i"]].copy()
    df["D_i_m"] = df["D_i_m"].astype(float)
    df["f_i"] = df["f_i"].astype(float)

    if (df["D_i_m"] <= 0).any():
        raise ValueError("All D_i_m must be > 0")
    if (df["f_i"] < 0).any():
        raise ValueError("All f_i must be >= 0")

    s = df["f_i"].sum()
    if s <= 0:
        raise ValueError("Sum of f_i must be > 0")
    # normalize
    df["f_i"] = df["f_i"] / s

    # sort ascending by D
    df = df.sort_values("D_i_m").reset_index(drop=True)
    return df

def read_psd_surface_csv(path_or_buffer: Union[str, IO, bytes]) -> pd.DataFrame:
    """
    Read a CSV containing surface PSD. Required columns: D_i_m (m) and f_i (fraction).
    If D_i_mm exists it will be converted to meters.
    The returned dataframe has columns D_i_m (float, m) and f_i (float, normalized).
    """
    # handle bytes (streamlit)
    if isinstance(path_or_buffer, (bytes, bytearray)):
        buf = io.BytesIO(path_or_buffer)
    else:
        buf = path_or_buffer

    df = pd.read_csv(buf)
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    if "D_i_m" not in df.columns and "D_i_mm" in df.columns:
        df["D_i_m"] = df["D_i_mm"].astype(float) / 1000.0

    _ensure_df_has_columns(df, ["D_i_m", "f_i"])

    df = df[["D_i_m", "f_i"]].copy()
    df["D_i_m"] = df["D_i_m"].astype(float)
    df["f_i"] = df["f_i"].astype(float)

    if (df["D_i_m"] <= 0).any():
        raise ValueError("All D_i_m must be > 0")
    if (df["f_i"] < 0).any():
        raise ValueError("All f_i must be >= 0")

    s = df["f_i"].sum()
    if s <= 0:
        raise ValueError("Sum of f_i must be > 0")

    df["f_i"] = df["f_i"] / s
    df = df.sort_values("D_i_m").reset_index(drop=True)
    return df

def read_fdc_csv(path_or_buffer: Union[str, IO, bytes]) -> pd.DataFrame:
    """
    Read a Flow Duration Curve CSV and return a normalized DataFrame with columns:
      - Q_m3s : discharge (m3/s)
      - p_time: probability (fraction of time or fraction of year, sums to 1)
    Accepted input formats:
      - columns Q_m3s and p_time (p_time already normalized or not)
      - columns Q_m3s and days_per_year (integers or floats; will be converted)
    """
    if isinstance(path_or_buffer, (bytes, bytearray)):
        buf = io.BytesIO(path_or_buffer)
    else:
        buf = path_or_buffer

    df = pd.read_csv(buf)
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    _ensure_df_has_columns(df, ["Q_m3s"])

    if "p_time" in df.columns:
        if (df["p_time"] < 0).any():
            raise ValueError("All p_time must be >= 0.")
        s = df["p_time"].sum()
        if s <= 0:
            raise ValueError("Sum of p_time must be > 0.")
        df["p_time"] = df["p_time"] / s
        return df[["Q_m3s", "p_time"]]

    if "days_per_year" in df.columns:
        if (df["days_per_year"] < 0).any():
            raise ValueError("All days_per_year must be >= 0.")
        total_days = df["days_per_year"].sum()
        if total_days <= 0:
            raise ValueError("Sum of days_per_year must be > 0.")
        df["p_time"] = df["days_per_year"] / total_days
        return df[["Q_m3s", "p_time"]]

    raise ValueError("FDC CSV must contain either 'p_time' or 'days_per_year' column.")
