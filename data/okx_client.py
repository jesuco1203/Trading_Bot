from __future__ import annotations
import os, pandas as pd

def _path(root: str, symbol: str, timeframe: str) -> str:
    p = symbol.replace("/", "_").replace(":", "_")
    return os.path.join(root, p, timeframe, "ohlcv.parquet")

def get_ohlcv(symbol: str, timeframe: str, limit: int = 3000,
              root: str = "data/okx") -> pd.DataFrame:
    fp = _path(root, symbol, timeframe)
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Parquet not found: {fp}")
    df = pd.read_parquet(fp)
    # normaliza columnas esperadas por build_features/main
    df = df.rename(columns={"ts":"ts","open":"open","high":"high","low":"low","close":"close","volume":"volume"})
    df = df.sort_values("ts").reset_index(drop=True)
    if limit and limit > 0:
        df = df.tail(limit).reset_index(drop=True)
    return df