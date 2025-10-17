# risk/position_sizing.py

from dataclasses import dataclass

@dataclass
class InstrumentSpec:
    symbol: str
    linear: bool = True          # True: contrato lineal (USDT margined); False: inverso (coin margined)
    contract_size: float = 1.0   # p.ej., 1 para swaps lineales; 0.001 para algunos inversos
    lot_step: float = 0.0001     # paso mínimo de qty
    min_qty: float = 0.001

@dataclass
class RiskConfig:
    risk_usd_per_trade: float = 25.0   # riesgo fijo por trade
    max_notional_usd: float = 1_000.0  # tope de nocional
    max_leverage: float = 1.0          # no aplicar dos veces
    slippage_usd: float = 0.0

def floor_to_step(x: float, step: float) -> float:
    if step <= 0: 
        return x
    return (int(x / step)) * step

def clamp_qty(qty: float, min_qty: float, step: float) -> float:
    q = floor_to_step(max(qty, min_qty), step)
    return q

def compute_qty_for_stop(
    entry_price: float,
    stop_price: float,
    inst: InstrumentSpec,
    risk: RiskConfig,
    side: int,  # +1 long, -1 short
) -> float:
    """
    qty = risk_usd / (USD loss per unit at SL)
    Maneja contratos lineales e inversos y aplica topes de nocional y leverage *una sola vez*.
    """

    # 1) Distancia al stop en USD por unidad de qty
    dist_px = abs(entry_price - stop_price)
    if dist_px <= 0:
        return 0.0

    if inst.linear:
        # Pérdida por 1 unidad de qty ≈ dist_px USD (si contract_size=1 equivale a 1 nocional)
        loss_per_qty_usd = dist_px * inst.contract_size
        notional_per_qty_usd = entry_price * inst.contract_size
    else:
        # Inverso: PnL ≈ (1/entry - 1/exit) * contract_size * USD_per_coin
        # Aproximación local: d(1/p) ≈ dist_px / (entry^2)
        loss_per_qty_usd = (dist_px / max(entry_price**2, 1e-12)) * inst.contract_size * entry_price
        notional_per_qty_usd = (inst.contract_size * entry_price)  # prox.

    if loss_per_qty_usd <= 0:
        return 0.0

    # 2) Qty inicial por riesgo fijo
    qty = (risk.risk_usd_per_trade + risk.slippage_usd) / loss_per_qty_usd

    # 3) Tope por nocional y leverage (no duplicar leverage en otra capa)
    #    nocional = qty * notional_per_qty_usd  => limitar por max_notional_usd * max_leverage
    max_notional = risk.max_notional_usd * max(risk.max_leverage, 1.0)
    if max_notional > 0:
        qty_cap = max_notional / max(notional_per_qty_usd, 1e-12)
        qty = min(qty, qty_cap)

    # 4) Clamp a paso y mínimo
    qty = clamp_qty(qty, inst.min_qty, inst.lot_step)

    return qty
