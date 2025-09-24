import pandas as pd
import numpy as np

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr1 = h - l
    tr2 = (h - prev_c).abs()
    tr3 = (l - prev_c).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    diff = s.diff()
    gain = diff.where(diff > 0, 0)
    loss = -diff.where(diff < 0, 0)
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return (100 - (100 / (1 + rs))).fillna(50)

def _zscore(s: pd.Series, n: int = 50) -> pd.Series:
    mean = s.rolling(n).mean()
    std = s.rolling(n).std().replace(0, 1e-9)
    return ((s - mean) / std).bfill()

def add_adx(df, n=14):
    h = df["high"].astype(float); l = df["low"].astype(float); c = df["close"].astype(float)
    up = h.diff(); dn = -l.diff()
    plus_dm  = np.where((up > dn) & (up > 0), up, 0.0).astype(float)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0).astype(float)

    tr1 = (h - l).abs()
    tr2 = (h - c.shift()).abs()
    tr3 = (l - c.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).astype(float)

    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    pdm = pd.Series(plus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean()
    mdm = pd.Series(minus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean()

    plus_di  = (100.0 * (pdm / atr.replace(0, np.nan))).fillna(0)
    minus_di = (100.0 * (mdm / atr.replace(0, np.nan))).fillna(0)
    dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) ) * 100.0
    adx = dx.ewm(alpha=1/n, adjust=False).mean().fillna(0.0)
    return adx.clip(0, 100), plus_di, minus_di

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["close"]

    out["ret"] = c.pct_change().fillna(0.0)
    out["vol"] = out["ret"].rolling(30).std().bfill()
    out["rng_pct"] = ((out["high"] - out["low"]) / c).rolling(14).mean().bfill()

    out['score'] = out['ret'].rolling(14).mean() * 100

    atr_abs = _atr(out, 14).bfill()
    out["atr"] = atr_abs
    out["atr_pct"] = (atr_abs / c).bfill()

    out["rsi14"] = _rsi(c, 14)
    out["z_score_50"] = _zscore(c, 50)

    out["sma20"] = c.rolling(20).mean().bfill()
    out["std20"] = c.rolling(20).std().bfill()

    from .ict import add_fvg_columns
    from .candles import add_basic_candles
    out = add_fvg_columns(out)
    out = add_basic_candles(out)

    out["adx"], out["di_plus"], out["di_minus"] = add_adx(out)

    out["ret"] = out["ret"].clip(-0.1, 0.1)
    out["vol"] = out["vol"].clip(0, out["vol"].quantile(0.99))
    out["rng_pct"] = out["rng_pct"].clip(0, out["rng_pct"].quantile(0.99))
    return out