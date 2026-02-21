# Sediment_Mech/core/settling_velocity.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PQM:
    P: float
    Q: float
    m: float


# Table 1.3 (for naturally worn sediment particles; Sp ≈ 0.7)
# See Table 1.3:
TABLE_13: Dict[str, PQM] = {
    "Rubey (1933)": PQM(P=24.0, Q=2.1, m=1.0),      # note: P changes for dn > 1 mm (not handled here)
    "Zhang (1961)": PQM(P=34.0, Q=1.2, m=1.0),
    "Zanke (1977)": PQM(P=24.0, Q=1.1, m=1.0),      # note: P changes for dn > 1 mm
    "Raudkivi (1990)": PQM(P=32.0, Q=1.2, m=1.0),
    "Fredsøe & Deigaard (1992)": PQM(P=36.0, Q=1.4, m=1.0),
    "Julien (1998)": PQM(P=24.0, Q=1.5, m=1.0),
    "Cheng (1997)": PQM(P=32.0, Q=1.0, m=1.5),
    "Soulsby (1997)": PQM(P=26.4, Q=1.27, m=1.0),
    "She et al. (2005)": PQM(P=35.0, Q=1.56, m=1.0),
    "Camenen (2007)": PQM(P=24.6, Q=0.96, m=1.53),
}


def pqm_wu_wang_2006(Sp: float) -> PQM:
    """
    Table 1.3 entry for Wu & Wang (2006):
    P = 53.5 exp(-0.65 Sp), Q = 5.65 exp(-2.5 Sp), m = 0.7 + 0.9 Sp
    
    """
    P = 53.5 * np.exp(-0.65 * Sp)
    Q = 5.65 * np.exp(-2.5 * Sp)
    m = 0.7 + 0.9 * Sp
    return PQM(P=float(P), Q=float(Q), m=float(m))


def nominal_diameter_from_median_sieve(d50_m: float) -> float:
    """
    dn ≈ d / 0.9 (median sieve diameter d -> nominal diameter dn)
    
    """
    return float(d50_m / 0.9)


def D_star(dn_m: float, s: float, g: float, nu_m2s: float) -> float:
    """
    D* = dn * ( ( (s-1) g ) / nu^2 )^(1/3)
    
    """
    return float(dn_m * (((s - 1.0) * g) / (nu_m2s ** 2)) ** (1.0 / 3.0))


def ws_eq_140(dn_m: float, s: float, g: float, nu_m2s: float, P: float, Q: float, m: float) -> float:
    """
    Terminal fall velocity from Eq. (1.40) derived from Eq. (1.39):
    ws = (P/Q) * (nu/dn) * [ sqrt(1/4 + (4Q/(3 P^2)) * D*^3) - 1/2 ]^m
    (notation as in text; D* defined above)
    
    """
    Dst = D_star(dn_m, s, g, nu_m2s)
    term = np.sqrt(0.25 + (4.0 * Q / (3.0 * (P ** 2))) * (Dst ** 3)) - 0.5
    ws = (P / Q) * (nu_m2s / dn_m) * (term ** m)
    return float(ws)


# Table 1.4 (alternative ws(D*) expressions)
# 
def ws_hallermeier_1981(dn_m: float, s: float, g: float, nu_m2s: float) -> float:
    Dst = D_star(dn_m, s, g, nu_m2s)
    if Dst <= 3.42:
        ws = (nu_m2s / dn_m) * (Dst ** 3) / 18.0
    elif Dst <= 21.54:
        ws = (nu_m2s / dn_m) * (Dst ** 2.1) / 6.0
    else:
        ws = 1.05 * (nu_m2s / dn_m) * (Dst ** 1.5)
    return float(ws)


def ws_chang_liou_2001(dn_m: float, s: float, g: float, nu_m2s: float) -> float:
    Dst = D_star(dn_m, s, g, nu_m2s)
    ws = 1.68 * (nu_m2s / dn_m) * (Dst ** 1.389) / (1.0 + 30.22 * (Dst ** 1.611))
    return float(ws)


def ws_guo_2002(dn_m: float, s: float, g: float, nu_m2s: float) -> float:
    Dst = D_star(dn_m, s, g, nu_m2s)
    ws = (nu_m2s / dn_m) * (Dst ** 3) / (24.0 + 0.866 * (Dst ** 1.5))
    return float(ws)


def ws_dietrich_1982_fit(dn_m: float, s: float, g: float, nu_m2s: float) -> float:
    """
    Eq. (1.41) polynomial in log(D*), as reported in the text.
    
    """
    Dst = D_star(dn_m, s, g, nu_m2s)
    logD = np.log10(Dst)
    c1, c2, c3, c4, c5 = 1.25572, 2.92944, 0.29445, 0.05175, 0.01512
    expo = c1 + c2 * logD - c3 * (logD ** 2) - c4 * (logD ** 3) + c5 * (logD ** 4)
    ws = (nu_m2s / dn_m) * (10.0 ** expo)
    return float(ws)


def ws_jimenez_madsen_2003(dn_m: float, s: float, g: float, nu_m2s: float) -> float:
    """
    Eq. (1.43) gives W* as function of S* (Jiménez & Madsen, 2003), as reported:
    W* = 0.954 + 20.48/S* - 1
    with W* = ws / sqrt((s-1) g dn), S* = dn * sqrt((s-1) g dn) / nu
    
    """
    sqrt_term = np.sqrt((s - 1.0) * g * dn_m)
    S_star = dn_m * sqrt_term / nu_m2s
    W_star = 0.954 + (20.48 / S_star) - 1.0
    ws = W_star * sqrt_term
    return float(ws)


def ws_from_table13_method(method: str, dn_m: float, s: float, g: float, nu_m2s: float, Sp: float) -> float:
    if method == "Wu & Wang (2006)":
        pqm = pqm_wu_wang_2006(Sp)
    else:
        pqm = TABLE_13[method]
    return ws_eq_140(dn_m, s, g, nu_m2s, pqm.P, pqm.Q, pqm.m)


def compute_ws_suite(
    dn_m: float,
    s: float,
    g: float,
    nu_m2s: float,
    Sp: float,
    methods_table13: List[str],
    methods_table14: List[str],
    include_dietrich: bool = True,
    include_jm: bool = True,
) -> pd.DataFrame:
    """
    Returns a DataFrame with ws for selected methods (like Example 1.4).
    Example 1.4 uses Eq (1.40)+Table 1.3 and Table 1.4 and Eqs (1.41)-(1.43).
    
    """
    rows = []
    for m in methods_table13:
        ws = ws_from_table13_method(m, dn_m, s, g, nu_m2s, Sp)
        rows.append({"family": "Table 1.3 (Eq. 1.40)", "method": m, "ws_m_s": ws})

    for m in methods_table14:
        if m == "Hallermeier (1981)":
            ws = ws_hallermeier_1981(dn_m, s, g, nu_m2s)
        elif m == "Chang & Liou (2001)":
            ws = ws_chang_liou_2001(dn_m, s, g, nu_m2s)
        elif m == "Guo (2002)":
            ws = ws_guo_2002(dn_m, s, g, nu_m2s)
        else:
            continue
        rows.append({"family": "Table 1.4", "method": m, "ws_m_s": ws})

    if include_dietrich:
        rows.append({"family": "Eq. (1.41)", "method": "Dietrich (1982) fit", "ws_m_s": ws_dietrich_1982_fit(dn_m, s, g, nu_m2s)})
    if include_jm:
        rows.append({"family": "Eq. (1.43)", "method": "Jiménez & Madsen (2003)", "ws_m_s": ws_jimenez_madsen_2003(dn_m, s, g, nu_m2s)})

    df = pd.DataFrame(rows)
    return df.sort_values(["family", "method"]).reset_index(drop=True)


# -----------------------------
# Hindered settling (Eqs 1.44–1.46) + Soulsby adjustment (Eq. 1.47)
# -----------------------------
def reynolds_particle(ws: float, dn_m: float, nu_m2s: float) -> float:
    # Re = ws dn / nu  
    return float(ws * dn_m / nu_m2s)


def n_richardson_zaki(Re: float) -> float:
    """
    n varies from 4.9 to 2.3 as Re increases 0.1 -> 1e3 (as stated in text).
    We implement a log-linear interpolation between these endpoints.
    
    """
    Re1, n1 = 0.1, 4.9
    Re2, n2 = 1e3, 2.3
    Re_clamped = min(max(Re, Re1), Re2)
    x = (np.log10(Re_clamped) - np.log10(Re1)) / (np.log10(Re2) - np.log10(Re1))
    return float(n1 + x * (n2 - n1))


def hindered_factor_richardson_zaki(C: float, n: float) -> float:
    # wsc = ws (1-C)^n  
    return float((1.0 - C) ** n)


def hindered_factor_oliver(C: float) -> float:
    # wsc = ws (1 - 2.15 C) (1 - 0.75 C^0.33)  
    return float((1.0 - 2.15 * C) * (1.0 - 0.75 * (C ** 0.33)))


def hindered_factor_sha(C: float, d50_mm: float) -> float:
    # wsc = ws [1 - C/(2 d50^0.5)]^3  (as reported) 
    # d50 in mm per statement "d50 <= 0.01 mm" in text.
    denom = 2.0 * (d50_mm ** 0.5)
    return float((1.0 - (C / denom)) ** 3)


def soulsby_adjusted_PQ(C: float) -> Tuple[float, float]:
    # P = 26/(1-C)^4.7 ; Q = 1.3/(1-C)^4.7  
    fac = (1.0 - C) ** 4.7
    return float(26.0 / fac), float(1.3 / fac)


def compute_hindered_suite(
    ws_clear: float,
    dn_m: float,
    nu_m2s: float,
    C: float,
    d50_mm_for_sha: Optional[float] = None,
    include_soulsby_adjustment: bool = True,
) -> pd.DataFrame:
    """
    Returns wsc computed via Eq. 1.44, 1.45, 1.46 (and optionally Eq. 1.47).
    """
    Re = reynolds_particle(ws_clear, dn_m, nu_m2s)
    n = n_richardson_zaki(Re)
    rows = []

    f_rz = hindered_factor_richardson_zaki(C, n)
    rows.append({"method": "Richardson & Zaki (1954) Eq. 1.44", "factor": f_rz, "wsc_m_s": ws_clear * f_rz})

    f_ol = hindered_factor_oliver(C)
    rows.append({"method": "Oliver (1961) Eq. 1.45", "factor": f_ol, "wsc_m_s": ws_clear * f_ol})

    if d50_mm_for_sha is not None:
        f_sha = hindered_factor_sha(C, d50_mm_for_sha)
        rows.append({"method": "Sha (1965) Eq. 1.46", "factor": f_sha, "wsc_m_s": ws_clear * f_sha})

    if include_soulsby_adjustment:
        P, Q = soulsby_adjusted_PQ(C)
        # Soulsby suggests adjusting P,Q for dense suspensions (keeps m=1 in Table 1.3 for Soulsby row)
        # then use Eq. (1.40) structure (implemented at UI level where s,g,dn,nu known).
        rows.append({"method": "Soulsby (1997) Eq. 1.47 (adjust P,Q)", "factor": np.nan, "wsc_m_s": np.nan, "P": P, "Q": Q})

    return pd.DataFrame(rows)
