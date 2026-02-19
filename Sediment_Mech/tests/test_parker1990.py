import pandas as pd
from Sediment_Mech.core.parker1990 import compute_parker1990

def test_parker_runs():
    df = pd.DataFrame({
        "D_i_m": [0.032, 0.016, 0.008, 0.004],
        "f_i":   [0.15, 0.35, 0.30, 0.20]
    })
    out = compute_parker1990(
        df_psd=df,
        tau_star_sg=0.06,
        tau_star_i=None,
        g=9.81,
        Delta=(2650-1000)/1000,
        remove_sand=True
    )
    assert out["q_b_tot"] >= 0.0
    assert out["D_sg"] > 0.0
