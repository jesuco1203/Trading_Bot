import pandas as pd

def add_fvg_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hi_prev = df["high"].shift(1)
    lo_prev = df["low"].shift(1)
    hi_next = df["high"].shift(-1)
    lo_next = df["low"].shift(-1)

    df["fvg_bull"] = (lo_next > hi_prev).fillna(False)
    df["fvg_bear"] = (hi_next < lo_prev).fillna(False)

    df["fvg_bull_top"]    = hi_prev.where(df["fvg_bull"])
    df["fvg_bull_bottom"] = lo_next.where(df["fvg_bull"])
    df["fvg_bear_top"]    = hi_next.where(df["fvg_bear"])
    df["fvg_bear_bottom"] = lo_prev.where(df["fvg_bear"])
    return df