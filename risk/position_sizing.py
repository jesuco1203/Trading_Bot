def atr_position_size(equity: float, sl_pts: float, risk_pct: float, tick_value: float) -> float:
    risk_usd = equity * risk_pct
    if sl_pts <= 0: return 0.0
    qty = risk_usd / max(sl_pts * tick_value, 1e-9)
    return max(qty, 0.0)