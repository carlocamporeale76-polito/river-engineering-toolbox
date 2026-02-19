from __future__ import annotations
import numpy as np

def qb_from_Phi(Phi: float, f_i: float, g: float, Delta: float, D: float) -> float:
    """q_bi = f_i * sqrt(g Delta D^3) * Phi  (volumetric per unit width, m^2/s)."""
    return float(f_i * np.sqrt(g * Delta * D**3) * Phi)
