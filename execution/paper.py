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
    trail_trigger_atr_mult: Optional[float] = None # New
    trail_sl_offset_atr_mult: Optional[float] = None # New
    rr: Optional[float] = None # New
    bars_open: int = 0 # New
    mfe_atr: float = 0.0 # New

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

    def enter_or_flip(self, side, qty, price, sl_pts=None, tp_pts=None, partial_tp_pts=None, ts=None, strategy_name: str = None, atr: float = 0.0, partial_sl_offset_atr_mult: Optional[float] = None, trail_trigger_atr_mult: Optional[float] = None, trail_sl_offset_atr_mult: Optional[float] = None, rr: Optional[float] = None, time_stop_bars: int = 0, time_stop_mfe_atr: float = 0.0):
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
        self.pos.trail_trigger_atr_mult = trail_trigger_atr_mult
        self.pos.trail_sl_offset_atr_mult = trail_sl_offset_atr_mult
        self.pos.rr = rr
        self.pos.time_stop_bars = time_stop_bars
        self.pos.time_stop_mfe_atr = time_stop_mfe_atr

    def mark_to_market(self, price, ts=None, high=None, low=None):
        if self.pos.side == 0:
            return 0.0

        # Update bars_open and mfe_atr
        self.pos.bars_open += 1
        atr_est = self.pos.entry_atr # Assuming entry_atr is a good estimate for current ATR
        if atr_est:
            adv = (high - self.pos.entry) if self.pos.side > 0 else (self.pos.entry - low)
            atr_units = (adv / atr_est) if atr_est else 0.0
            self.pos.mfe_atr = max(self.pos.mfe_atr, atr_units)

        # Time-stop logic
        if self.pos.time_stop_bars > 0 and self.pos.bars_open >= self.pos.time_stop_bars and self.pos.mfe_atr < self.pos.time_stop_mfe_atr:
            exit_px = price # or close
            pnl = (exit_px - self.pos.entry) * self.pos.side * self.pos.qty
            fee = (abs(self.pos.qty) * exit_px) * self.comm # Recalculate fee for time-stop exit
            self.capital += pnl - fee
            self.trades.append({"ts": ts, "pnl": pnl - fee, "partial": False, "side": self.pos.side, "strategy": self.pos.strategy_name, "exit_reason": "time_stop", "sl_pts": self.pos.sl, "tp_pts": self.pos.tp, "rr": self.pos.rr})
            self.pos = Position()
            self.exits_count += 1
            return pnl

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

        if self.pos.partial_done and not self.pos.trail_set and self.pos.trail_trigger_atr_mult is not None and self.pos.trail_sl_offset_atr_mult is not None:
            target_trail = self.pos.entry + self.pos.side * (self.pos.trail_trigger_atr_mult * self.pos.entry_atr)
            hit_trail = (self.pos.side > 0 and high >= target_trail) or (self.pos.side < 0 and low <= target_trail)
            if hit_trail:
                self.pos.sl = self.pos.entry + self.pos.side * (self.pos.trail_sl_offset_atr_mult * self.pos.entry_atr)
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
            old_sl_pts = self.pos.sl
            old_tp_pts = self.pos.tp
            old_rr = self.pos.rr # Capture rr

            self.trades.append({"ts": ts, "pnl": pnl - fee, "partial": False, "side": self.pos.side, "strategy": self.pos.strategy_name, "exit_reason": exit_reason, "sl_pts": old_sl_pts, "tp_pts": old_tp_pts, "rr": old_rr})
            
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
