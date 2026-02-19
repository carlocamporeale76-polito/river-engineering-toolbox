import pandas as pd
from core.ashida_michiue import compute_ashida_michiue

def test_basic_run():
    df = pd.DataFrame({
        "D_i_m": [0.02, 0.01],
        "f_i": [0.6, 0.4]
    })

    tau_star = {0: 0.06, 1: 0.06}
    g = 9.81
    Delta = (2650-1000)/1000

    res, qtot = compute_ashida_michiue(df, tau_star, g, Delta)

    assert qtot >= 0
