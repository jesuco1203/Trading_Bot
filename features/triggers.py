"""
Gatillos de entrada por acción del precio, para la Fase 3.

Las tres variantes que comparamos, todas evaluadas en la barra i y ejecutadas al
cierre de i (que es lo que hace main.py), así que no hay lookahead:

  engulf_classic : envolvente de 2 velas del libro de texto.
  engulf_multi   : el cierre de i supera el extremo de las N barras previas
                   ("varias velas que envuelven la anterior").
  thrust         : versión cuantitativa — cuerpo > k*ATR y cierre en el cuarto
                   superior (o inferior) del rango de la barra.

Sobre la envolvente clásica en cripto: el patrón debe su significado al hueco de
apertura entre sesiones. En un mercado 24/7 no hay huecos, así que se degrada a
"vela con cuerpo grande". Por eso incluimos 'thrust', que mide directamente lo
que la envolvente intenta señalar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def engulf_classic(df: pd.DataFrame) -> pd.DataFrame:
    o, c = df["open"], df["close"]
    po, pc = o.shift(1), c.shift(1)
    return pd.DataFrame({
        "long": ((c > o) & (pc < po) & (c >= po) & (o <= pc)).fillna(False),
        "short": ((c < o) & (pc > po) & (c <= po) & (o >= pc)).fillna(False),
    }, index=df.index)


def engulf_multi(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """El cierre de i supera el máximo (o mínimo) de las n barras anteriores."""
    c = df["close"]
    prev_hi = df["high"].shift(1).rolling(n).max()
    prev_lo = df["low"].shift(1).rolling(n).min()
    return pd.DataFrame({
        "long": (c > prev_hi).fillna(False),
        "short": (c < prev_lo).fillna(False),
    }, index=df.index)


def thrust(df: pd.DataFrame, atr: pd.Series, k: float = 0.8,
           close_pos: float = 0.75) -> pd.DataFrame:
    """Cuerpo > k*ATR y cierre en el extremo del rango de la propia barra."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    pos = ((c - l) / rng)  # 1.0 = cierra en máximos, 0.0 = en mínimos
    big = body > (k * atr)
    return pd.DataFrame({
        "long": (big & (c > o) & (pos >= close_pos)).fillna(False),
        "short": (big & (c < o) & (pos <= 1.0 - close_pos)).fillna(False),
    }, index=df.index)


def build_all(df: pd.DataFrame, atr: pd.Series, multi_n: int = 3,
              thrust_k: float = 0.8, thrust_pos: float = 0.75) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for name, t in (("engulf", engulf_classic(df)),
                    ("multi", engulf_multi(df, multi_n)),
                    ("thrust", thrust(df, atr, thrust_k, thrust_pos))):
        out[f"trig_{name}_long"] = t["long"]
        out[f"trig_{name}_short"] = t["short"]
    return out


def trigger_gate(row: pd.Series, side: str, mode: str) -> bool:
    """¿La barra de entrada satisface el gatillo pedido? mode='off' no filtra."""
    if mode == "off":
        return True
    if mode not in ("engulf", "multi", "thrust"):
        raise ValueError(f"gatillo desconocido: {mode}")
    key = f"trig_{mode}_{'long' if side == 'long' else 'short'}"
    return bool(row.get(key, False))
