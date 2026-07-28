"""Tests de features/triggers.py: definición correcta y ausencia de lookahead."""
import numpy as np
import pandas as pd
import pytest

from features.triggers import (build_all, engulf_classic, engulf_multi,
                               thrust, trigger_gate)


def _bars(rows):
    idx = pd.date_range("2021-01-01", periods=len(rows), freq="4h", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def test_engulfente_alcista_reconocida():
    # barra 0 bajista (10->9), barra 1 alcista que la envuelve (8.9->10.5)
    df = _bars([[10, 10.1, 8.9, 9.0], [8.9, 10.6, 8.8, 10.5]])
    t = engulf_classic(df)
    assert bool(t["long"].iloc[1]) is True
    assert bool(t["short"].iloc[1]) is False


def test_engulfente_no_dispara_si_no_envuelve():
    # segunda vela alcista pero no cubre el cuerpo previo
    df = _bars([[10, 10.1, 8.9, 9.0], [9.5, 9.9, 9.4, 9.8]])
    assert bool(engulf_classic(df)["long"].iloc[1]) is False


def test_multi_rompe_extremo_de_n_barras():
    df = _bars([[10, 11, 9, 10], [10, 12, 9, 10], [10, 10.5, 9, 10], [10, 13, 9, 12.5]])
    t = engulf_multi(df, n=3)
    assert bool(t["long"].iloc[3]) is True   # 12.5 > max(11,12,10.5)
    assert bool(t["long"].iloc[2]) is False


def test_thrust_exige_cuerpo_y_cierre_en_extremo():
    atr = pd.Series(1.0, index=_bars([[0, 0, 0, 0]] * 2).index)
    grande_arriba = _bars([[10, 10, 10, 10], [10, 11.1, 9.9, 11.0]])
    assert bool(thrust(grande_arriba, atr, k=0.8, close_pos=0.75)["long"].iloc[1]) is True

    # mismo cuerpo pero cierra en mitad del rango -> no es thrust
    medio = _bars([[10, 10, 10, 10], [10, 12.5, 9.9, 11.0]])
    assert bool(thrust(medio, atr, k=0.8, close_pos=0.75)["long"].iloc[1]) is False

    # cierra arriba pero el cuerpo es pequeño frente al ATR
    pequeno = _bars([[10, 10, 10, 10], [10, 10.3, 9.99, 10.2]])
    assert bool(thrust(pequeno, atr, k=0.8, close_pos=0.75)["long"].iloc[1]) is False


@pytest.mark.parametrize("cut", [40, 63, 80])
def test_sin_lookahead(cut):
    """Los gatillos miran la barra i y las previas; truncar el futuro no cambia nada."""
    rng = np.random.default_rng(3)
    n = 120
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = _bars(np.c_[close, close * 1.01, close * 0.99, close])
    atr = pd.Series(np.abs(rng.normal(1, .1, n)), index=df.index)

    full = build_all(df, atr)
    trunc = build_all(df.iloc[:cut], atr.iloc[:cut])
    for c in full.columns:
        pd.testing.assert_series_equal(full[c].iloc[:cut], trunc[c],
                                       check_names=False, obj=c)


def test_gate():
    fila = pd.Series({"trig_engulf_long": True, "trig_engulf_short": False})
    assert trigger_gate(fila, "long", "off") is True
    assert trigger_gate(fila, "long", "engulf") is True
    assert trigger_gate(fila, "short", "engulf") is False
    with pytest.raises(ValueError):
        trigger_gate(fila, "long", "martillo")
