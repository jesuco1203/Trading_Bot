import pandas as pd
import numpy as np

def add_basic_candles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    o,h,low_price,c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng  = (h - low_price).replace(0, np.nan)
    df["doji"] = (body / rng) < 0.1

    prev_o, prev_c = o.shift(1), c.shift(1)
    bull = (c > o) & (prev_c < prev_o) & (c >= prev_o) & (o <= prev_c)
    bear = (c < o) & (prev_c > prev_o) & (c <= prev_o) & (o >= prev_c)
    df["bull_engulf"] = bull.fillna(False)
    df["bear_engulf"] = bear.fillna(False)
    return df