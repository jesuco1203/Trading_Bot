from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import logging
from uuid import uuid4

@dataclass
class Position:
    side: int = 0
    qty: float = 0.0
    entry: float = 0.0
    entry_ts: Any = None # New: Timestamp of entry
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
    mae_atr: float = 0.0 # New
    symbol: Optional[str] = None # New
    tf: Optional[str] = None # New
    trade_id: Optional[str] = None # New: Unique ID for each trade

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

    def enter_or_flip(self, side, qty, price, sl_pts=None, tp_pts=None, partial_tp_pts=None, ts=None, strategy_name: str = None, atr: float = 0.0, partial_sl_offset_atr_mult: Optional[float] = None, trail_trigger_atr_mult: Optional[float] = None, trail_sl_offset_atr_mult: Optional[float] = None, rr: Optional[float] = None, time_stop_bars: int = 0, time_stop_mfe_atr: float = 0.0, mae_atr: float = 0.0, symbol: Optional[str] = None, tf: Optional[str] = None):
        if self.exposure() != 0:
            return

        fee = (abs(qty) * price) * self.comm
        self.capital -= fee
        
        self.pos.trade_id = str(uuid4())
        self.pos.side = side
        self.pos.qty = qty
        self.pos.entry = price
        self.pos.entry_ts = ts # Set entry timestamp
        self.pos.entry_atr = atr
        self.pos.rr = rr if rr is not None else (tp_pts / max(sl_pts or 0.0, 1e-9))

        # Add entry event to trades list
        self.trades.append({
            "trade_id": self.pos.trade_id,
            "ts": ts,
            "entry_ts": ts,
            "strategy": strategy_name,
            "partial": False,
            "pnl": 0.0,
            "rr": self.pos.rr,
            "event": "entry"
        })

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
        self.pos.mae_atr = mae_atr
        self.pos.symbol = symbol
        self.pos.tf = tf

    def mark_to_market(self, price, ts=None, high=None, low=None):
        if self.pos.side == 0:
            return 0.0

        # Update bars_open, mfe_atr, and mae_atr
        self.pos.bars_open += 1
        atr_est = self.pos.entry_atr # Assuming entry_atr is a good estimate for current ATR
        if atr_est:
            adv = (high - self.pos.entry) if self.pos.side > 0 else (self.pos.entry - low)
            atr_units = (adv / atr_est) if atr_est else 0.0
            self.pos.mfe_atr = max(self.pos.mfe_atr, atr_units)

            # Calculate MAE in ATR units
            adv_mae = (self.pos.entry - low) if self.pos.side > 0 else (high - self.pos.entry)
            mae_atr_units = (adv_mae / atr_est) if atr_est else 0.0
            self.pos.mae_atr = max(self.pos.mae_atr, mae_atr_units)

        # Time-stop logic
        if self.pos.bars_open >= 6 and self.pos.mfe_atr < 0.6:  # 6 velas sin alza ≥0.6*ATR
            exit_px = price
            # Calculate fee for time-stop exit
            fee = (abs(self.pos.qty) * exit_px) * self.comm
            pnl = (exit_px - self.pos.entry) * self.pos.side * self.pos.qty - fee
            self.capital += pnl
            self.trades.append({"ts": ts, "trade_id": self.pos.trade_id, "entry_ts": self.pos.entry_ts,
                                "strategy": self.pos.strategy_name, "partial": False,
                                "pnl": pnl, "exit_reason": "time_stop", "event": "close"})
            logging.info(f"EXIT reason=time_stop bars_open={self.pos.bars_open} mfe_atr={self.pos.mfe_atr:.2f}")
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
                fill = self.pos.entry + self.pos.side * self.pos.partial_tp
                self.trades.append({
                    "trade_id": self.pos.trade_id,
                    "ts": ts,
                    "entry_ts": self.pos.entry_ts,
                    "strategy": self.pos.strategy_name,
                    "partial": True,
                    "pnl": pnl - fee,
                    "event": "partial",
                    "fill_px": fill
                })
                self.pos.qty -= half
                self.pos.partial_done = True
                # Move SL to BE or BE + offset
                if self.pos.partial_sl_offset_atr_mult is not None:
                    self.pos.sl = self.pos.entry + self.pos.side * (self.pos.partial_sl_offset_atr_mult * self.pos.entry_atr)
                else:
                    self.pos.sl = self.pos.entry
                self.pos.be_set = True
                self.partials_count += 1
                logging.info(f"PARTIAL px={fill_price:.2f} rem_qty={self.pos.qty:.4f} -> SL={self.pos.sl:.2f}")

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

            self.trades.append({
                "trade_id": self.pos.trade_id,
                "ts": ts,
                "entry_ts": self.pos.entry_ts,
                "strategy": self.pos.strategy_name,
                "partial": False,
                "pnl": pnl - fee,
                "exit_reason": exit_reason,
                "event": "close"
            })
            
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

    def export_trades_to_csv(self, filename="trades.csv"):
        if not self.trades:
            logging.info("No trades to export.")
            return

        df_trades = pd.DataFrame(self.trades)
        # Ensure all specified columns exist, fill missing with None or NaN
        required_columns = ["ts", "symbol", "tf", "strategy", "side", "entry", "exit", "exit_reason", "partial", "pnl", "rr", "bars_open", "mfe_atr", "mae_atr"]
        for col in required_columns:
            if col not in df_trades.columns:
                df_trades[col] = None # Or np.nan

        # Rename columns to match user's request (entry_price -> entry, exit_price -> exit)
        df_trades = df_trades.rename(columns={"entry_price": "entry", "exit_price": "exit"})

        # Select and reorder columns
        df_trades = df_trades[required_columns]

        df_trades.to_csv(filename, index=False)
        logging.info(f"Trades exported to {filename}")
