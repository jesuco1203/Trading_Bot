from dataclasses import dataclass
from typing import Optional, Any
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
    max_loss_usd: float = 0.0
    entry_fee: float = 0.0

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

    def enter_or_flip(self, side, qty, price, sl_pts=None, tp_pts=None, partial_tp_pts=None, ts=None, strategy_name: str = None, atr: float = 0.0, partial_sl_offset_atr_mult: Optional[float] = None, trail_trigger_atr_mult: Optional[float] = None, trail_sl_offset_atr_mult: Optional[float] = None, rr: Optional[float] = None, time_stop_bars: int = 0, time_stop_mfe_atr: float = 0.0, mae_atr: float = 0.0, symbol: Optional[str] = None, tf: Optional[str] = None, max_loss_usd: float = 0.0):
        if self.exposure() != 0:
            return

        fee = (abs(qty) * price) * self.comm
        self.capital -= fee
        self.pos.entry_fee = fee
        
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
        
        if strategy_name == "Trend" and tf == "30m":
            self.pos.partial_tp = self.pos.entry + self.pos.side * (1.0 * atr)
            self.pos.partial_sl_offset_atr_mult = 0.3
        elif partial_tp_pts is not None:
            self.pos.partial_tp = self.pos.entry + self.pos.side * partial_tp_pts
        
        self.pos.strategy_name = strategy_name
        self.pos.partial_done = False
        self.pos.be_set = False
        self.pos.trail_set = False
        self.pos.partial_sl_offset_atr_mult = partial_sl_offset_atr_mult
        self.pos.trail_trigger_atr_mult = trail_trigger_atr_mult
        self.pos.trail_sl_offset_atr_mult = trail_sl_offset_atr_mult
        self.pos.time_stop_bars = time_stop_bars
        self.pos.time_stop_mfe_atr = time_stop_mfe_atr
        self.pos.mae_atr = mae_atr
        self.pos.symbol = symbol
        self.pos.tf = tf
        self.pos.max_loss_usd = max_loss_usd

    def _close_position(self, exit_px, reason, ts):
        pnl_gross = (exit_px - self.pos.entry) * self.pos.side * self.pos.qty
        close_fee = (abs(self.pos.qty) * exit_px) * self.comm
        total_fees = self.pos.entry_fee + close_fee
        pnl_net = pnl_gross - total_fees

        self.capital += pnl_gross - close_fee

        if self.pos.strategy_name == "MeanRevert":
            logging.info(f"MR CLOSE DEBUG:\n" +
                         f"  exit={exit_px:.2f}, pnl_gross={pnl_gross:.2f}, fees_total={total_fees:.2f}, pnl_net={pnl_net:.2f}, bars_open={self.pos.bars_open}, exit_reason={reason}")
            
            allowed_loss = self.pos.max_loss_usd + 5.0 * total_fees
            if self.pos.bars_open <= 1 and -pnl_net > allowed_loss + 1e-6:
                 raise AssertionError(f"MR loss exceeded (1-bar sanity): pnl_net={pnl_net:.2f} > allowed_loss={allowed_loss:.2f}")

        self.trades.append({
            "trade_id": self.pos.trade_id,
            "ts": ts,
            "entry_ts": self.pos.entry_ts,
            "strategy": self.pos.strategy_name,
            "partial": False,
            "pnl": pnl_net,
            "exit_reason": reason,
            "event": "close",
            "bars_open": self.pos.bars_open,
            "mfe_atr": round(self.pos.mfe_atr, 4),
            "mae_atr": round(self.pos.mae_atr, 4)
        })
        
        if reason == 'sl' and self.pos.strategy_name:
            strat = self._strategies.get(self.pos.strategy_name)
            if strat and hasattr(strat, "on_stop"):
                strat.on_stop()
        
        self.pos = Position()
        self.exits_count += 1
        return pnl_net

    def close_open_position(self, price, ts, reason="session_end"):
        if self.exposure() == 0:
            return 0.0
        return self._close_position(price, reason, ts)

    def mark_to_market(self, price, ts=None, high=None, low=None, current_atr: float = 0.0):
        if self.pos.side == 0:
            return 0.0

        # Update bars_open, mfe_atr, and mae_atr
        atr_est = current_atr # Use current_atr
        if atr_est:
            rng_high = high if self.pos.side>0 else self.pos.entry
            rng_low  = low  if self.pos.side>0 else self.pos.entry
            adv = (rng_high - self.pos.entry) if self.pos.side>0 else (self.pos.entry - rng_low)
            ret = (self.pos.entry - rng_low) if self.pos.side>0 else (rng_high - self.pos.entry)

            self.pos.mfe_atr = max(self.pos.mfe_atr, adv / max(atr_est,1e-9))
            self.pos.mae_atr = max(self.pos.mae_atr, ret / max(atr_est,1e-9))
        self.pos.bars_open += 1

        # Time-stop logic for Trend (ADAPTIVE)
        if self.pos.strategy_name == "Trend":
            extend_duration = self.pos.bars_open >= 8 and self.pos.mfe_atr >= 0.8 and self.pos.mfe_atr < 1.2
            close_for_stagnation = self.pos.bars_open >= 6 and self.pos.mfe_atr < 0.4
            if close_for_stagnation and not extend_duration:
                return self._close_position(price, "time_stop_no_progress", ts)

        # Time-stop logic for MeanRevert
        if self.pos.strategy_name == "MeanRevert" and self.pos.time_stop_bars > 0 and self.pos.bars_open >= self.pos.time_stop_bars and self.pos.mfe_atr < self.pos.time_stop_mfe_atr:
            return self._close_position(price, "time_stop", ts)

        if self.pos.partial_tp is not None and not self.pos.partial_done:
            hit_partial = (self.pos.side > 0 and high >= self.pos.partial_tp) or (self.pos.side < 0 and low <= self.pos.partial_tp)
            if hit_partial:
                fill_price = self.pos.partial_tp
                qty_closed = self.pos.qty * 0.5
                
                pnl_gross = (fill_price - self.pos.entry) * self.pos.side * qty_closed
                partial_fee = (qty_closed * fill_price) * self.comm
                pnl_net = pnl_gross - partial_fee

                if self.pos.strategy_name == "MeanRevert":
                    logging.info(
                        f"MR PARTIAL DEBUG: entry={self.pos.entry:.2f} exit={fill_price:.2f} "
                        f"qty_closed={qty_closed:.6f} side={self.pos.side} "
                        f"pnl_gross={pnl_gross:.2f} fees={partial_fee:.2f} pnl_net={pnl_net:.2f} "
                        f"pos.qty_after={self.pos.qty - qty_closed:.6f}"
                    )
                    assert abs(pnl_net) < 1e5, "PnL event out of range"

                self.capital += pnl_net
                self.pos.entry_fee += partial_fee

                self.trades.append({
                    "trade_id": self.pos.trade_id,
                    "ts": ts,
                    "entry_ts": self.pos.entry_ts,
                    "strategy": self.pos.strategy_name,
                    "partial": True,
                    "pnl": pnl_net,
                    "event": "partial",
                })
                self.pos.qty -= qty_closed
                self.pos.partial_done = True
                # Move SL to BE or BE + offset (soft BE)
                move_to = self.pos.entry + self.pos.side * 0.30 * atr_est
                if self.pos.side > 0:
                    self.pos.sl = max(self.pos.sl, move_to)
                else:
                    self.pos.sl = min(self.pos.sl, move_to)
                self.pos.be_set = True
                self.partials_count += 1
                logging.info(f"PARTIAL px={fill_price:.2f} rem_qty={self.pos.qty:.4f} -> SL={self.pos.sl:.2f}")

        if self.pos.partial_done:
            gain_atr = (price - self.pos.entry) * self.pos.side / max(atr_est, 1e-9)
            trail_mult = 0.3 if gain_atr < 2.0 else 0.5
            trail_level = price - self.pos.side * trail_mult * atr_est
            if self.pos.side > 0:
                self.pos.sl = max(self.pos.sl, trail_level)
            else:
                self.pos.sl = min(self.pos.sl, trail_level)
            logging.info(f"TRAIL ts={ts} new_sl={self.pos.sl:.2f}")

        hit_sl = (self.pos.sl is not None) and ((self.pos.side > 0 and low <= self.pos.sl) or (self.pos.side < 0 and high >= self.pos.sl))
        hit_tp = (self.pos.tp is not None) and ((self.pos.side > 0 and high >= self.pos.tp) or (self.pos.side < 0 and low <= self.pos.tp))

        if hit_sl or hit_tp:
            exit_px = self.pos.sl if hit_sl and not hit_tp else self.pos.tp
            exit_reason = "sl" if hit_sl and not hit_tp else "tp"
            return self._close_position(exit_px, exit_reason, ts)

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
