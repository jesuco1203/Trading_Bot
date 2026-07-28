"""
Tests de no-lookahead para features/htf.py.

El fallo que buscamos: que la barra de 4h de las 04:00 del día D vea el cierre
del día D (que sólo se conoce a las 24:00). Un backtest con ese sesgo produce
resultados brillantes e irreproducibles en real.

El test central es de CAUSALIDAD: si el valor de una barra depende sólo del
pasado, truncar los datos posteriores no puede cambiarlo.
"""
import numpy as np
import pandas as pd
import pytest

from features.htf import add_htf_bias, htf_gate


def _synthetic(n=4000, seed=7):
    idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(seed)
    close = 10_000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {"open": close, "high": close * 1.004, "low": close * 0.996,
         "close": close, "volume": 1.0},
        index=idx,
    )


COLS = ["htf_bias_4h", "htf_bias_d", "htf_bias_w"]


@pytest.mark.parametrize("cut", [1500, 2001, 2617, 3000])
def test_truncar_el_futuro_no_altera_el_pasado(cut):
    """Causalidad: recalcular sin los datos futuros da los mismos valores."""
    df = _synthetic()
    full = add_htf_bias(df)
    trunc = add_htf_bias(df.iloc[:cut])

    for c in COLS:
        a = full[c].iloc[:cut]
        b = trunc[c]
        pd.testing.assert_series_equal(a, b, check_names=False,
                                       obj=f"{c} cambió al conocer el futuro")


def test_sesgo_diario_usa_el_dia_anterior():
    """
    Comprobación directa: el sesgo diario vigente durante el día D debe salir
    de datos que terminan, como muy tarde, al cerrar D-1.
    """
    df = _synthetic()
    out = add_htf_bias(df)

    day = pd.Timestamp("2020-10-15", tz="UTC")
    en_el_dia = out.loc[day: day + pd.Timedelta(hours=20), "htf_bias_d"]
    assert en_el_dia.notna().all()
    # constante dentro del día: no se actualiza intradía
    assert en_el_dia.nunique() == 1

    # y coincide con recalcularlo viendo SÓLO hasta el cierre de D-1
    hasta_ayer = add_htf_bias(df.loc[: day - pd.Timedelta(seconds=1)])
    assert hasta_ayer["htf_bias_d"].iloc[-1] == en_el_dia.iloc[0]


def test_sesgo_no_formado_es_nan_no_cero():
    """Un sesgo desconocido debe ser NaN; 0.0 se confundiría con 'neutral'."""
    out = add_htf_bias(_synthetic(n=300))
    assert out["htf_bias_4h"].iloc[:200].isna().all()
    assert out["htf_bias_d"].iloc[0] != 0.0 or np.isnan(out["htf_bias_d"].iloc[0])


def test_gate_bloquea_cuando_el_htf_no_esta_formado():
    fila = pd.Series({"htf_bias_4h": np.nan, "htf_bias_d": 1.0, "htf_bias_w": 1.0})
    assert htf_gate(fila, "long", "off") is True          # baseline nunca bloquea
    assert htf_gate(fila, "long", "daily") is False       # NaN => no operar
    assert htf_gate(fila, "long", "daily", require_htf=False) is True


def test_gate_respeta_la_direccion():
    alcista = pd.Series({"htf_bias_4h": 1.0, "htf_bias_d": 1.0, "htf_bias_w": 1.0})
    mixta = pd.Series({"htf_bias_4h": 1.0, "htf_bias_d": -1.0, "htf_bias_w": 1.0})

    assert htf_gate(alcista, "long", "daily_weekly") is True
    assert htf_gate(alcista, "short", "daily_weekly") is False
    assert htf_gate(mixta, "long", "ema200") is True      # sólo mira 4h
    assert htf_gate(mixta, "long", "daily") is False      # el diario discrepa


def test_modo_desconocido_falla_fuerte():
    fila = pd.Series({"htf_bias_4h": 1.0})
    with pytest.raises(ValueError):
        htf_gate(fila, "long", "diario")
