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
