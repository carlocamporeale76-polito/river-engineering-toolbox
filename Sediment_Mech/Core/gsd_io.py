# Sediment_Mech/core/gsd_io.py
from typing import Tuple
import pandas as pd

def read_gsd_xlsx(path_or_buffer) -> pd.DataFrame:
    """
    Read a GSDCalculator.xlsx file (uploaded file-like or path) and return
    a DataFrame with columns: class_id, D_i_m, f_i (normalized).
    The function attempts common sheet/column names and does validations.
    """
    # read all sheets (small file)
    xls = pd.ExcelFile(path_or_buffer, engine="openpyxl")
    # prefer a sheet named 'GSD' or first sheet otherwise
    possible = [s for s in xls.sheet_names if s.lower() in ("gsd", "surface", "psd", "sheet1")]
    sheet = possible[0] if possible else xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet)

    # common column name mappings
    colmap = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc in ("class", "class_id", "id"):
            colmap[c] = "class_id"
        elif lc in ("d", "di", "di_m", "d_i_m", "diameter_m", "diameter"):
            colmap[c] = "D_i_m"
        elif lc in ("f", "fi", "f_i", "fraction", "surface_fraction", "percent"):
            colmap[c] = "f_i"
        elif lc in ("d_mm", "d_mm", "diameter_mm"):
            # convert mm to m
            colmap[c] = "D_i_mm"

    df = df.rename(columns=colmap)

    # if D_i_mm present convert to meters
    if "D_i_mm" in df.columns and "D_i_m" not in df.columns:
        df["D_i_m"] = df["D_i_mm"].astype(float) / 1000.0

    # keep only needed cols
    if not {"D_i_m", "f_i"}.issubset(df.columns):
        raise ValueError("GSD file must contain diameter and fraction columns. Found: {}".format(df.columns.tolist()))

    df = df[["D_i_m", "f_i"]].copy()
    df["D_i_m"] = df["D_i_m"].astype(float)
    df["f_i"] = df["f_i"].astype(float)

    # normalize fractions to sum=1 (if sum > 0)
    s = df["f_i"].sum()
    if s <= 0:
        raise ValueError("Sum of fractions <= 0 in GSD file.")
    df["f_i"] = df["f_i"] / s

    # add a class id if missing
    df = df.reset_index(drop=True)
    df.insert(0, "class_id", range(1, len(df) + 1))

    return df
