from __future__ import annotations
from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd

@dataclass
class Signal:
    side: str
    strength: float
    sl_pts: Optional[float]
    tp_pts: Optional[float]
    partial_tp_pts: Optional[float] = None
    partial_sl_offset_atr_mult: Optional[float] = None
    reason: Optional[str] = None

class BaseStrategy:
    def __init__(self, name: str, risk_mult: float = 1.0):
        self.name = name
        self.risk_mult = risk_mult

    def signal(self, ctx: Dict[str, Any]) -> Signal | None:
        raise NotImplementedError

    def warmup_bars(self) -> int:
        return 0

    def on_stop(self):
        pass

    def print_summary(self, trades: list):
        strat_trades = [t for t in trades if t.get("strategy") == self.name]
        
        grouped_trades = {}
        for trade in strat_trades:
            ts = trade['ts']
            if ts not in grouped_trades:
                grouped_trades[ts] = []
            grouped_trades[ts].append(trade)

        entries = len(grouped_trades)
        wins = 0
        total_strat_pnl = 0.0
        for ts, trade_group in grouped_trades.items():
            total_pnl_for_entry = sum(ev["pnl"] for ev in trade_group)
            if total_pnl_for_entry > 0:
                wins += 1
            total_strat_pnl += total_pnl_for_entry

        pnl = total_strat_pnl
        hit_rate = (wins / entries) * 100 if entries > 0 else 0.0
        
        rr_ratios = []
        for trade in strat_trades:
            # Ensure sl_pts and tp_pts exist and sl_pts is not zero to avoid division by zero
            if trade.get('sl_pts') is not None and trade.get('tp_pts') is not None and trade['sl_pts'] > 0:
                rr_ratios.append(trade['tp_pts'] / trade['sl_pts'])
        rr_avg = sum(rr_ratios) / len(rr_ratios) if rr_ratios else 0.0

        print(f"--- {self.name} Summary ---")
        print(f"Entries: {entries} | Wins: {wins} | Hit Rate: {hit_rate:.2f}% | PnL: {pnl:.2f} | RR Avg: {rr_avg:.2f}")
