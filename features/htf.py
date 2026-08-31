"""
Sesgo de marco temporal superior (HTF) para estrategias de 4h.

Diseño y por qué:

1. El HTF se DERIVA de las propias barras de 4h por resampleo, no se descarga
   aparte. 6x4h = 1 día exacto y 42x4h = 1 semana exacta, así que la alineación
   es perfecta por construcción. Descargar velas 1D de OKX introduciría un
   desfase silencioso: OKX cierra el día a las 16:00 UTC (horario Hong Kong).

2. ANTI-LOOKAHEAD. Una barra diaria etiquetada D sólo se conoce cuando D
   termina. Por eso la serie HTF se desplaza un periodo (shift(1)) antes de
   propagarse a las barras de 4h: la barra de 4h de las 04:00 del día D ve el
   cierre del día D-1, nunca el de D. Verificado en tests/test_htf_lookahead.py.

3. El sesgo semanal NO usa EMA200. 200 semanas son ~4 años; con el histórico
   disponible (~340 semanas) quedaría medio dataset consumido en warmup. Se usa
   EMA20 semanal (~5 meses), que es lo que un swing de 4h puede aprovechar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_OHLC = {"open": "first", "high": "max", "low": "min", "close": "last"}


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {k: v for k, v in _OHLC.items() if k in df.columns}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    out = df.resample(rule, closed="left", label="left").agg(agg)
    return out.dropna(subset=["close"])


def _bias(close: pd.Series, ref: pd.Series) -> pd.Series:
    """+1 si el cierre está por encima de la referencia, -1 si por debajo."""
    return pd.Series(np.where(close > ref, 1, -1), index=close.index, dtype=float)


def add_htf_bias(
    df: pd.DataFrame,
    ema_len_4h: int = 200,
    ema_len_d: int = 200,
    ema_len_w: int = 20,
) -> pd.DataFrame:
    """
    Añade a un DataFrame de 4h (índice datetime UTC, columnas OHLC):

      htf_bias_4h : +1/-1  cierre 4h vs EMA200 de 4h
      htf_bias_d  : +1/-1  cierre diario vs EMA200 diaria   (con shift(1))
      htf_bias_w  : +1/-1  cierre semanal vs EMA20 semanal  (con shift(1))

    Las columnas diaria y semanal valen NaN hasta que hay HTF cerrado suficiente;
    el llamante decide si eso bloquea la operativa o no.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("add_htf_bias requiere índice datetime (barras de 4h)")

    out = df.copy()
    close = out["close"].astype(float)

    # --- 4h: mismo marco, sin shift (la EMA en la barra i sólo usa cierres <= i)
    ema200_4h = close.ewm(span=ema_len_4h, adjust=False).mean()
    out["htf_ema200_4h"] = ema200_4h
    out["htf_bias_4h"] = _bias(close, ema200_4h)
    # Sin 'ema_len_4h' cierres reales la EMA es sólo el arranque: no es sesgo.
    out.loc[out.index[: ema_len_4h], "htf_bias_4h"] = np.nan

    # --- Diario y semanal: resampleo + shift(1) + propagación hacia adelante
    for tag, rule, span in (("d", "1D", ema_len_d), ("w", "1W", ema_len_w)):
        htf = _resample(out[["open", "high", "low", "close"]], rule)
        ema = htf["close"].ewm(span=span, adjust=False).mean()
        bias = _bias(htf["close"], ema)
        bias.iloc[:span] = np.nan  # EMA aún no formada

        # shift(1): en la etiqueta D queda el sesgo del periodo D-1, que es el
        # último HTF efectivamente CERRADO cuando empieza D. Sin esto, la barra
        # de 4h de las 04:00 vería el cierre de su propio día.
        bias_lagged = bias.shift(1)

        out[f"htf_bias_{tag}"] = bias_lagged.reindex(out.index, method="ffill")

    return out


def htf_gate(row: pd.Series, side: str, mode: str, require_htf: bool = True) -> bool:
    """
    ¿Permite el sesgo HTF operar en 'side' ("long"/"short") en esta barra?

    mode:
      "off"          sin filtro (baseline)
      "ema200"       sólo EMA200 de 4h
      "daily"        EMA200 4h + sesgo diario
      "daily_weekly" EMA200 4h + sesgo diario + semanal

    require_htf=True bloquea cuando el sesgo aún es NaN (HTF sin formar). Es la
    opción honesta: operar ahí sería operar sin el filtro que dices tener.
    """
    if mode == "off":
        return True

    want = 1.0 if side == "long" else -1.0
    cols = {"ema200": ["htf_bias_4h"],
            "daily": ["htf_bias_4h", "htf_bias_d"],
            "daily_weekly": ["htf_bias_4h", "htf_bias_d", "htf_bias_w"]}.get(mode)
    if cols is None:
        raise ValueError(f"modo HTF desconocido: {mode}")

    for c in cols:
        v = row.get(c, np.nan)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            if require_htf:
                return False
            continue
        if float(v) != want:
            return False
    return True
