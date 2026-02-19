from __future__ import annotations
import numpy as np

def shear_stress(rho: float, g: float, h: float, S: float) -> float:
    """Bed shear stress tau = rho g h S."""
    return float(rho * g * h * S)

def shields_stress(tau: float, rho: float, rho_s: float, g: float, D: float) -> float:
    """tau* = tau / ((rho_s - rho) g D)."""
    return float(tau / ((rho_s - rho) * g * D))

def depth_from_manning(Q: float, n: float, b: float, S: float) -> float:
    """
    Wide-rectangular approximation:
      Q = (1/n) A R^(2/3) S^(1/2), with A=b h, R≈h.
      => Q = (1/n) b h^(5/3) S^(1/2)
      => h = [Q n / (b S^(1/2))]^(3/5)
    """
    if Q < 0 or n <= 0 or b <= 0 or S <= 0:
        raise ValueError("Require Q>=0, n>0, b>0, S>0.")
    if Q == 0:
        return 0.0
    return float((Q * n / (b * np.sqrt(S))) ** (3.0 / 5.0))
