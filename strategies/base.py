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
        # An entry is a sequence of trades (partial + final) starting at the same timestamp.
        # For simplicity, we count unique timestamps of non-partial trades.
        entry_timestamps = {t['ts'] for t in strat_trades if not t.get("partial")}
        entries = len(entry_timestamps)
        
        # A win is when the sum of pnl for a given entry is positive.
        wins = 0
        for ts in entry_timestamps:
            trade_pnl = sum(t["pnl"] for t in strat_trades if t['ts'] == ts)
            if trade_pnl > 0:
                wins += 1

        pnl = sum(t["pnl"] for t in strat_trades)
        hit_rate = (wins / entries) * 100 if entries > 0 else 0
        print(f"--- {self.name} Summary ---")
        print(f"Entries: {entries} | Wins: {wins} | Hit Rate: {hit_rate:.2f}% | PnL: {pnl:.2f}")