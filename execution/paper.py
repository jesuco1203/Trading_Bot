from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import logging

@dataclass
class Position:
    side: int = 0
    qty: float = 0.0
    entry: float = 0.0
    sl: Optional[float] = None
    tp: Optional[float] = None
    partial_tp: Optional[float] = None
    strategy_name: Optional[str] = None
    partial_done: bool = False
    be_set: bool = False
    trail_set: bool = False
    entry_atr: float = 0.0
    partial_sl_offset_atr_mult: Optional[float] = None

class PaperBroker:
    def __init__(self, initial_capital=10000, comm_rate=0.0005, slippage_min=0.5, spread_bias=0.0, all_strategies: dict = {}):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.comm = comm_rate
        self.slip = slippage_min
        self.bias = spread_bias
        self.pos = Position()
        self.trades = []
        self.entries_count = 0
        self.exits_count = 0
        self.flips_count = 0
        self.partials_count = 0
        self._strategies = all_strategies

    def get_equity(self):
        return self.capital

    def exposure(self) -> int: return self.pos.side

    def enter_or_flip(self, side, qty, price, sl_pts=None, tp_pts=None, partial_tp_pts=None, ts=None, strategy_name: str = None, atr: float = 0.0, partial_sl_offset_atr_mult: Optional[float] = None):
        if self.exposure() != 0:
            return

        fee = (abs(qty) * price) * self.comm
        self.capital -= fee
        
        self.pos.side = side
        self.pos.qty = qty
        self.pos.entry = price
        self.pos.entry_atr = atr

        if sl_pts is not None:
            self.pos.sl = self.pos.entry - self.pos.side * sl_pts
        if tp_pts is not None:
            self.pos.tp = self.pos.entry + self.pos.side * tp_pts
        if partial_tp_pts is not None:
            self.pos.partial_tp = self.pos.entry + self.pos.side * partial_tp_pts
        
        self.pos.strategy_name = strategy_name
        self.pos.partial_done = False
        self.pos.be_set = False
        self.pos.trail_set = False
        self.pos.partial_sl_offset_atr_mult = partial_sl_offset_atr_mult

    def mark_to_market(self, price, ts=None, high=None, low=None):
        if self.pos.side == 0:
            return 0.0

        if self.pos.partial_tp is not None and not self.pos.partial_done:
            hit_partial = (self.pos.side > 0 and high >= self.pos.partial_tp) or (self.pos.side < 0 and low <= self.pos.partial_tp)
            if hit_partial:
                fill_price = self.pos.partial_tp
                half = self.pos.qty * 0.5
                pnl = (fill_price - self.pos.entry) * self.pos.side * half
                fee = (abs(half) * fill_price) * self.comm
                self.capital += pnl - fee
                self.trades.append({"ts": ts, "pnl": pnl - fee, "partial": True, "side": self.pos.side, "strategy": self.pos.strategy_name, "exit_reason": "Partial TP", "exit_price": fill_price})
                self.pos.qty -= half
                self.pos.partial_done = True
                # Move SL to BE or BE + offset
                if self.pos.partial_sl_offset_atr_mult is not None:
                    self.pos.sl = self.pos.entry + self.pos.side * (self.pos.partial_sl_offset_atr_mult * self.pos.entry_atr)
                else:
                    self.pos.sl = self.pos.entry
                self.pos.be_set = True
                self.partials_count += 1
                logging.info(f"PARTIAL ts={ts} px={fill_price:.2f} rem_qty={self.pos.qty:.4f} moved_SL=BE")

        if self.pos.partial_done and not self.pos.trail_set:
            target_trail = self.pos.entry + self.pos.side * (1.5 * self.pos.entry_atr)
            hit_trail = (self.pos.side > 0 and high >= target_trail) or (self.pos.side < 0 and low <= target_trail)
            if hit_trail:
                self.pos.sl = self.pos.entry + self.pos.side * (0.3 * self.pos.entry_atr)
                self.pos.trail_set = True
                logging.info(f"TRAIL ts={ts} new_sl={self.pos.sl:.2f}")

        hit_sl = (self.pos.sl is not None) and ((self.pos.side > 0 and low <= self.pos.sl) or (self.pos.side < 0 and high >= self.pos.sl))
        hit_tp = (self.pos.tp is not None) and ((self.pos.side > 0 and high >= self.pos.tp) or (self.pos.side < 0 and low <= self.pos.tp))

        if hit_sl or hit_tp:
            exit_px = self.pos.sl if hit_sl and not hit_tp else self.pos.tp
            pnl = (exit_px - self.pos.entry) * self.pos.side * self.pos.qty
            fee = (abs(self.pos.qty) * exit_px) * self.comm
            self.capital += pnl - fee
            exit_reason = "sl" if hit_sl and not hit_tp else "tp"
            self.trades.append({"ts": ts, "pnl": pnl - fee, "partial": False, "side": self.pos.side, "strategy": self.pos.strategy_name, "exit_reason": exit_reason})
            
            if hit_sl and self.pos.strategy_name:
                strat = self._strategies.get(self.pos.strategy_name)
                if strat and hasattr(strat, "on_stop"): strat.on_stop()
            
            self.pos = Position()
            self.exits_count += 1
            return pnl

        return 0.0

    def summary(self):
        n = len(self.trades)
        pnl = sum(t["pnl"] for t in self.trades)
        wins = sum(1 for t in self.trades if t["pnl"] > 0)
        return {
            "n_trades": n, "net_pnl": pnl, "hit_rate": (wins/n if n else 0.0),
            "entries_count": self.entries_count, "exits_count": self.exits_count,
            "flips_count": self.flips_count, "partials_count": self.partials_count
        }

    def print_summary(self):
        s = self.summary()
        print("--- Backtest Summary ---")
        print(f"Entries: {s['entries_count']} | Exits: {s['exits_count']} | Partials: {s['partials_count']} | Flips: {s['flips_count']}")
        print(f"Total Trades Recorded: {s['n_trades']}")
        print(f"Net PnL: {s['net_pnl']:.2f}")
        print(f"Hit Rate: {s['hit_rate']:.2%}")
        print(f"Final Capital: {self.capital:.2f}")
