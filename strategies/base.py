from __future__ import annotations
from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import logging

@dataclass
class Signal:
    side: str
    strength: float
    sl_pts: Optional[float]
    tp_pts: Optional[float]
    partial_tp_pts: Optional[float] = None
    partial_sl_offset_atr_mult: Optional[float] = None
    reason: Optional[str] = None
    rr: Optional[float] = None # New

class BaseStrategy:
    def __init__(self, name: str, risk_mult: float = 1.0, time_stop_bars: int = 0, time_stop_mfe_atr: float = 0.0):
        self.name = name
        self.risk_mult = risk_mult
        self.time_stop_bars = time_stop_bars
        self.time_stop_mfe_atr = time_stop_mfe_atr

    def signal(self, ctx: Dict[str, Any]) -> Signal | None:
        raise NotImplementedError

    def warmup_bars(self) -> int:
        return 0

    def on_stop(self):
        pass

    def print_summary(self, trades: list):
        strat_trades = [t for t in trades if t.get("strategy") == self.name]
        if not strat_trades:
            return 0.0, 0.0, 0, 0, 0.0

        # agrupar por operación (entrada)
        ops = {}
        for t in strat_trades:
            ets = t.get("entry_ts")
            if ets is None:  # cordón de seguridad
                ets = t.get("ts")
            ops.setdefault(ets, []).append(t)

        entries = len(ops); wins = 0; total_pnl = 0.0; rrs = []
        for ets, evs in ops.items():
            evs.sort(key=lambda x: x.get("ts", 0))
            # rr desde el evento de entrada
            entry_ev = next((e for e in evs if e.get("event") == "entry"), None)
            if entry_ev and entry_ev.get("rr") is not None:
                rrs.append(float(entry_ev["rr"]))

            # pnl total de la operación
            pnl_op = sum(float(e.get("pnl", 0.0)) for e in evs)
            total_pnl += pnl_op

            # win: cierre final por TP (estricto) o pnl_op>0 (simple). Elige 1 y deja comentada la otra.
            close_ev = next((e for e in reversed(evs) if e.get("event") == "close"), None)
            is_win = (close_ev and close_ev.get("exit_reason") == "tp")
            is_win = is_win or (pnl_op > 0)   # <-- opción alternativa
            if is_win:
                wins += 1

        rr_avg = (sum(rrs) / len(rrs)) if rrs else 0.0
        hit_rate = (wins / entries) * 100 if entries else 0.0
        return total_pnl, rr_avg, entries, wins, hit_rate
