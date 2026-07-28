import math
import numpy as np
import pandas as pd

def track_mfe_mae(ohlcv, atr_series, side, entry_idx, exit_idx, entry_px, atr_at_entry):
    """
    ohlcv: DataFrame con columnas high, low (y opcional close)
    atr_series: Serie ATR alineada
    side: +1 long, -1 short
    entry_idx, exit_idx: índices absolutos
    entry_px: precio de entrada
    atr_at_entry: ATR en la barra de entrada
    """
    window = slice(entry_idx, exit_idx + 1)
    highs = ohlcv["high"].iloc[window]
    lows  = ohlcv["low"].iloc[window]

    if side == 1:  # long
        mfe_abs = highs.max() - entry_px
        mae_abs = entry_px - lows.min()
    else:          # short
        mfe_abs = entry_px - lows.min()
        mae_abs = highs.max() - entry_px

    if not atr_at_entry or atr_at_entry == 0:
        mfe_atr = 0.0
        mae_atr = 0.0
    else:
        mfe_atr = float(mfe_abs / atr_at_entry)
        mae_atr = float(mae_abs / atr_at_entry)

    return float(mfe_abs), float(mae_abs), mfe_atr, mae_atr


def compute_pf_expectancy(trades_df: pd.DataFrame, group_col: str = "trade_id"):
    """
    Espera un DataFrame con trades cerrados, con al menos:
      - 'pnl' (R o dinero)
      - 'closed' (bool) opcional para filtrar
      - group_col (opcional): identificador de operación. Si está presente, las
        filas de una misma operación (tomas parciales + cierre final) se suman
        antes de calcular PF/expectancy.

    La unidad económica es la OPERACIÓN, no la fila. Un trade con parcial genera
    dos filas ('tp_partial' y el cierre del resto); tratarlas por separado infla
    el número de operaciones y —si además se descarta la fila del parcial—
    elimina la parte ganadora y sesga PF y expectancy a la baja.
    """
    if trades_df is None or len(trades_df) == 0:
        return 0.0, 0.0

    df = trades_df.copy()
    if "closed" in df.columns:
        df = df[df["closed"] == True]

    if len(df) == 0:
        return 0.0, 0.0

    if "pnl" not in df.columns:
        for candidate in ["pnl_r", "net_pnl", "profit"]:
            if candidate in df.columns:
                df["pnl"] = df[candidate]
                break
    if "pnl" not in df.columns:
        return 0.0, 0.0

    # Agrupar DESPUÉS de resolver el nombre de la columna de PnL, para que un
    # DataFrame con 'pnl_r'/'net_pnl' también se agregue por operación.
    if group_col and group_col in df.columns and df[group_col].notna().any():
        df = df.groupby(group_col, dropna=False, as_index=False)["pnl"].sum()

    pnl = (
        df["pnl"]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if pnl.empty:
        return 0.0, 0.0

    gains = pnl[pnl > 0.0]
    losses = pnl[pnl < 0.0]

    gross_profit = gains.sum() if not gains.empty else 0.0
    gross_loss = losses.sum() if not losses.empty else 0.0  # negativo

    if gross_loss < 0.0:
        pf = gross_profit / abs(gross_loss)
    elif gross_profit > 0.0 and gross_loss == 0.0:
        pf = float("inf")
    else:
        pf = 0.0

    expectancy = pnl.mean()

    if not math.isfinite(pf):
        pf = 0.0
    if not math.isfinite(expectancy):
        expectancy = 0.0

    return float(pf), float(expectancy)
