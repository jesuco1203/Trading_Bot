from dataclasses import dataclass, field
from typing import Optional, Any
import pandas as pd
import logging
from uuid import uuid4
from utils.perf import track_mfe_mae
from exits import manage_exit

@dataclass
class Position:
    side: int = 0
    qty: float = 0.0
    qty0: float = 0.0  # cantidad al abrir; qty se reduce en los parciales
    entry: float = 0.0
    entry_ts: Any = None # New: Timestamp of entry
    sl: Optional[float] = None
    initial_sl: Optional[float] = None
    tp: Optional[float] = None
    partial_target: Optional[float] = None
    final_target: Optional[float] = None
    partial_tp: Optional[float] = None
    strategy_name: Optional[str] = None
    partial_done: bool = False
    partial_take_r: float = 0.0
    partial_take_frac: float = 0.0
    trail_activate_r: float = 1.0
    partial_be_eps_atr: float = 0.0
    partial_last_px: Optional[float] = None
    be_set: bool = False
    trail_set: bool = False
    entry_atr: float = 0.0
    force_exit: bool = False
    raw_triggers_long: int = 0
    raw_triggers_short: int = 0
    filters_passed_long: int = 0
    filters_passed_short: int = 0
    trigger_raw: int = 0
    filters_passed: int = 0
    bars_open: int = 0
    mfe_atr: float = 0.0
    mae_atr: float = 0.0
    audit_bias: Optional[int] = None
    audit_adx: Optional[float] = None
    audit_slope: Optional[float] = None
    audit_vol_ratio: Optional[float] = None
    metadata: dict = field(default_factory=dict)
    max_rr: float = 0.0
    tag: Optional[str] = None

def _pnl_usd(entry_px: float, exit_px: float, qty_closed: float, side: int, fee_bps: float = 0.0) -> float:
    """
    side: +1 long, -1 short
    fee_bps: comisiones totales (ida+vuelta) en basis points, e.g. 8 = 0.08%
    """
    gross = (exit_px - entry_px) * qty_closed if side > 0 else (entry_px - exit_px) * qty_closed
    notional = (abs(entry_px) + abs(exit_px)) * qty_closed
    fees = notional * (fee_bps / 10000.0) if fee_bps else 0.0
    return float(gross - fees)

def normalize_exit_reason(reason: str, side: int, entry_px: float, exit_px: float) -> str:
    if reason == "sl":
        if (side == 1 and exit_px > entry_px) or (side == -1 and exit_px < entry_px):
            return "tsl_win"      # stop movido, cierre con ganancia
        if abs(exit_px - entry_px) < 1e-9:
            return "be"           # break-even exacto
    return reason

class PaperBroker:
    def __init__(self, initial_capital=10000, taker_fee=0.0005, maker_fee=0.0002, slippage_bps=1.0, all_strategies: dict = {}, wf_start_index: int = 0,
                 sl_atr: float = 0.0, tp_r_primary: float = 0.0, tp_primary_ratio: float = 0.0, tp_final_r: float = 0.0,
                 be_trigger_atr: float = 0.0, trail_atr_mult: float = 0.0, time_stop_bars: int = 0,
                 trail_activate_r: float = 1.0, partial_take_r: float = 0.0, partial_take_frac: float = 0.0,
                 partial_be_eps_atr: float = 0.05,
                 risk_manager: Any = None):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.slippage_pct = slippage_bps / 10000.0  # Convert basis points to percentage
        self.fee_bps = (self.taker_fee + self.slippage_pct) * 10000 # Calculate total fee in basis points
        self.pos = Position()
        self.trades = []
        self.entries_count = 0
        self.entries_long = 0
        self.entries_short = 0
        self.exits_count = 0
        self.flips_count = 0
        self.partials_count = 0
        self._strategies = all_strategies
        self.wf_start_index = wf_start_index
        self.sl_atr = sl_atr
        self.tp_r_primary = tp_r_primary
        self.tp_primary_ratio = tp_primary_ratio
        self.tp_final_r = tp_final_r
        self.be_trigger_atr = be_trigger_atr
        self.trail_atr_mult = trail_atr_mult
        self.time_stop_bars = time_stop_bars
        self.trail_activate_r = trail_activate_r
        self.partial_take_r = partial_take_r
        self.partial_take_frac = partial_take_frac
        self.partial_be_eps_atr = partial_be_eps_atr
        self._risk_manager = risk_manager

    def get_equity(self):
        return self.capital

    def exposure(self) -> int: return self.pos.side

    def enter_or_flip(self, side, qty, price, sl_pts=None, tp_pts=None, partial_tp_pts=None, ts=None, strategy_name: str = None, atr: float = 0.0, partial_sl_offset_atr_mult: Optional[float] = None, trail_trigger_atr_mult: Optional[float] = None, trail_sl_offset_atr_mult: Optional[float] = None, rr: Optional[float] = None, time_stop_bars: int = 0, time_stop_mfe_atr: float = 0.0, mae_atr: float = 0.0, symbol: Optional[str] = None, tf: Optional[str] = None, max_loss_usd: float = 0.0, origin: str = "normal",
                        raw_triggers_long: int = 0, raw_triggers_short: int = 0, filters_passed_long: int = 0, filters_passed_short: int = 0, trigger_raw: int = 0, filters_passed: int = 0,
                        audit_bias: Optional[int] = None, audit_adx: Optional[float] = None, audit_slope: Optional[float] = None, audit_vol_ratio: Optional[float] = None,
                        tag: Optional[str] = None):
        if self.exposure() != 0:
            return

        strat = self._strategies.get(strategy_name)
        if strat:
            strat.reentered = False

        # Store raw entry price
        self.pos.raw_entry_price = price
        # Apply slippage to entry price for fee calculation and PnL tracking
        entry_price_with_slippage = price * (1 + self.slippage_pct * side)

        fee = (abs(qty) * entry_price_with_slippage) * self.taker_fee
        self.capital -= fee
        self.pos.entry_fee = fee
        
        self.pos.trade_id = str(uuid4())
        self.pos.side = side
        self.pos.qty = qty
        self.pos.qty0 = qty
        self.pos.entry = entry_price_with_slippage # This is the slippage-adjusted entry price
        self.pos.entry_ts = ts # Set entry timestamp
        self.pos.entry_atr = atr
        self.pos.rr = rr if rr is not None else (tp_pts / max(sl_pts or 0.0, 1e-9))
        self.pos.raw_triggers_long = raw_triggers_long
        self.pos.raw_triggers_short = raw_triggers_short
        self.pos.filters_passed_long = filters_passed_long
        self.pos.filters_passed_short = filters_passed_short
        self.pos.trigger_raw = trigger_raw
        self.pos.filters_passed = filters_passed
        self.pos.audit_bias = audit_bias
        self.pos.audit_adx = audit_adx
        self.pos.audit_slope = audit_slope
        self.pos.audit_vol_ratio = audit_vol_ratio


        if sl_pts is not None:
            self.pos.sl = self.pos.raw_entry_price - self.pos.side * sl_pts
            self.pos.initial_sl = self.pos.sl
        
        # al abrir
        R = abs(self.pos.raw_entry_price - self.pos.sl)
        self.pos.tp = self.pos.raw_entry_price + side * (self.tp_final_r * R)
        
        # Partial + BE + Time-stop logic
        self.pos.partial_take_r = self.partial_take_r
        self.pos.partial_take_frac = self.partial_take_frac
        self.pos.trail_activate_r = self.trail_activate_r
        self.pos.partial_be_eps_atr = self.partial_be_eps_atr
        self.pos.partial_done = False
        self.pos.max_rr = 0.0
        initial_risk = abs(self.pos.raw_entry_price - self.pos.sl)
        if self.pos.partial_take_r > 0 and initial_risk > 0:
            self.pos.partial_target = self.pos.raw_entry_price + self.pos.side * (self.pos.partial_take_r * initial_risk)
        else:
            self.pos.partial_target = None
        self.pos.partial_last_px = None
        self.pos.final_target   = self.pos.entry + self.pos.side * (1.8 * 2.2 * atr) # consistent with TP=1.8*SL and SL=2.2*ATR
        self.pos.time_stop_bars = 24
        self.pos.partial_filled = False # Ensure this is reset on new entry

        self.pos.strategy_name = strategy_name
        self.pos.be_set = False
        self.pos.trail_set = False
        self.pos.partial_sl_offset_atr_mult = partial_sl_offset_atr_mult
        self.pos.trail_trigger_atr_mult = trail_trigger_atr_mult
        self.pos.trail_sl_offset_atr_mult = trail_sl_offset_atr_mult
        self.pos.time_stop_mfe_atr = time_stop_mfe_atr
        self.pos.mae_atr = mae_atr
        self.pos.symbol = symbol
        self.pos.tf = tf
        self.pos.max_loss_usd = max_loss_usd
        self.pos.metadata['origin'] = origin
        self.pos.metadata['tag'] = tag
        self.pos.tag = tag

    def _close_position(self, exit_px, reason, ts, i: int = 0, shared_context: dict = {}, strategy_stats: dict = {}, pnl_net: float = None):
        logging.info(f"[CLOSE_POS DEBUG] i={i}, reason={reason}, bars_open={self.pos.bars_open}") # Debug print
        qty_closed = self.pos.qty # Assuming no partial closes for now
        if pnl_net is None: # If pnl_net is not provided, calculate it
            pnl_net = _pnl_usd(self.pos.entry, exit_px, qty_closed, self.pos.side, fee_bps=self.fee_bps)

        # Adjust capital based on net PnL
        self.capital += pnl_net

        logging.info(f"EXIT {reason} side={self.pos.side} entry={self.pos.entry:.2f} sl={self.pos.sl:.2f} tp={self.pos.tp:.2f} exit={exit_px:.2f} pnl={pnl_net:.2f}")

        if self.pos.metadata.get('origin') == 'momentum_override':
            shared_context['mo_open'] = False
            if pnl_net < 0:
                shared_context['mo_cooldown_until'] = i + 50 # COOLDOWN_BARS
            i_abs = self.wf_start_index + i
            logging.info(f"TREND MO CLOSE: i={i} i_abs={i_abs}, exit={exit_px:.2f}, pnl_net={pnl_net:.2f}, bars_open={self.pos.bars_open}, exit_reason={reason}, next_cooldown_until={shared_context.get('mo_cooldown_until')}")



        if self.pos.strategy_name in ["Trend", "TrendV2"]:
            # Recalculate total_fees for the sanity check
            entry_fee = (abs(self.pos.qty) * self.pos.entry) * self.taker_fee
            close_fee = (abs(self.pos.qty) * exit_px) * self.taker_fee # Use raw exit_px for close_fee calculation
            total_fees_for_check = entry_fee + close_fee
            allowed_loss = self.pos.max_loss_usd + 5.0 * total_fees_for_check
            if -pnl_net > allowed_loss + 1e-6:
                raise AssertionError(f"Trend loss exceeded: {pnl_net:.2f} > {allowed_loss:.2f}")

        entry_idx = shared_context['df_base'].index.get_loc(self.pos.entry_ts)
        exit_idx = shared_context['df_base'].index.get_loc(ts)
        mfe_abs, mae_abs, mfe_atr, mae_atr = track_mfe_mae(shared_context['df_base'], shared_context['df_base']['atr'], self.pos.side, entry_idx, exit_idx, self.pos.entry, self.pos.entry_atr)

        logging.info("CLOSE %s @ %.2f -> MFE_ATR=%.2f MAE_ATR=%.2f", "LONG" if self.pos.side==1 else "SHORT", exit_px, mfe_atr, mae_atr)
        
        strat = self._strategies.get(self.pos.strategy_name)
        if strat and hasattr(strat, "on_exit"):
            logging.info(
                f"[EXIT DEBUG] reason={reason} side={self.pos.side} bars_open={self.pos.bars_open} i_local={i}"
            )
            strat.on_exit(
                reason,
                self.pos.bars_open,
                self.pos.side,
                i_local=i,
                exit_price=exit_px,
                entry_price=self.pos.entry,
            )

        if self.pos.strategy_name:
            if strat:
                strat.stats_mfe_atr.append(mfe_atr)
                strat.stats_mae_atr.append(mae_atr)

        norm_reason = normalize_exit_reason(reason, self.pos.side, self.pos.entry, exit_px)

        pnl_price = (exit_px - self.pos.entry) if self.pos.side == 1 else (self.pos.entry - exit_px)
        initial_risk_R = abs(self.pos.entry - self.pos.initial_sl) if self.pos.initial_sl is not None else 0
        rr_achieved = pnl_price / initial_risk_R if initial_risk_R > 0 else 0

        trade_entry = {
            "ts": ts,
            "entry_ts": self.pos.entry_ts,
            "qty0": getattr(self.pos, "qty0", self.pos.qty),
            "symbol": self.pos.symbol,
            "tf": self.pos.tf,
            "strategy": self.pos.strategy_name or "TrendV2",
            "side": self.pos.side,
            "entry": self.pos.entry,
            "exit": exit_px,
            "exit_reason": norm_reason,
            "partial": self.pos.partial_done,
            "pnl": round(pnl_net, 4),
            "rr": round(rr_achieved, 4),
            "bars_open": int(self.pos.bars_open),
            "max_rr": round(self.pos.max_rr, 4),
            "origin": self.pos.metadata.get('origin'),
            "trade_id": getattr(self.pos, "trade_id", None),
            "tag": self.pos.metadata.get('tag') or "",
        }
        self.trades.append(trade_entry)

        if self._risk_manager is not None:
            try:
                self._risk_manager.on_trade_close(
                    idx=i,
                    pnl_r_multiple=rr_achieved,
                    pnl=pnl_net,
                    equity=self.capital,
                    ts=ts,
                )
            except Exception:
                logging.exception("RiskManager.on_trade_close failed")
        
        self.pos = Position()
        self.exits_count += 1
        return pnl_net

    def close_open_position(self, price, ts, reason="session_end", shared_context: dict = {}, i: int = 0):
        if self.exposure() == 0:
            return 0.0
        return self._close_position(price, reason, ts, i, shared_context, strategy_stats=self._strategies.get(self.pos.strategy_name).stats)

    def mark_to_market(self, price, ts=None, high: pd.Series = None, low: pd.Series = None, current_atr: float = 0.0, i: int = 0, shared_context: dict = {}):
        if self.pos.side == 0:
            return 0.0

        self.pos.bars_open += 1

        hi_val = float(high.iloc[-1])
        lo_val = float(low.iloc[-1])
        cl = float(shared_context['df_base']['close'].iloc[i])

        # MFE/MAE calculation
        if self.pos.side == 1:
            mfe = hi_val - self.pos.entry
            mae = self.pos.entry - lo_val
        else:
            mfe = self.pos.entry - lo_val
            mae = hi_val - self.pos.entry
        self.pos.mfe_atr = max(self.pos.mfe_atr, mfe / self.pos.entry_atr if self.pos.entry_atr > 0 else 0)
        self.pos.mae_atr = max(self.pos.mae_atr, mae / self.pos.entry_atr if self.pos.entry_atr > 0 else 0)

        initial_risk = abs(self.pos.raw_entry_price - self.pos.initial_sl) if self.pos.initial_sl is not None else 0.0
        if initial_risk > 0:
            if self.pos.side == 1:
                bar_rr = (hi_val - self.pos.raw_entry_price) / initial_risk
            else:
                bar_rr = (self.pos.raw_entry_price - lo_val) / initial_risk
        else:
            bar_rr = 0.0

        if bar_rr > self.pos.max_rr:
            self.pos.max_rr = bar_rr
            logging.info(f"[RR DEBUG] i={i} rr={bar_rr:.2f} side={self.pos.side} entry={self.pos.raw_entry_price:.2f}")

        if self.pos.partial_target is not None and not self.pos.partial_done:
            try:
                logging.info(f"[PARTIAL DEBUG] i={i} R={bar_rr:.2f} thr={self.pos.partial_take_r:.2f} taken={self.pos.partial_done}")
            except Exception:
                pass

        if (
            not self.pos.partial_done
            and self.pos.partial_take_frac > 0
            and self.pos.partial_take_r > 0
            and initial_risk > 0
        ):
            partial_px = self.pos.raw_entry_price + self.pos.side * (self.pos.partial_take_r * initial_risk)
            hit_partial = (self.pos.side == 1 and hi_val >= partial_px) or (self.pos.side == -1 and lo_val <= partial_px)
            if hit_partial:
                self._execute_partial(partial_px, ts)

        new_sl, _, _ = manage_exit(
            self.pos,
            self.pos.side,
            self.pos.raw_entry_price,
            self.pos.sl,
            current_atr,
            self.pos.bars_open,
            high,
            low,
            self.sl_atr,
            self.tp_r_primary,
            self.tp_primary_ratio,
            self.tp_final_r,
            self.be_trigger_atr,
            self.trail_atr_mult,
            self.time_stop_bars,
            self.trail_activate_r,
            bar_rr,
        )
        self.pos.sl = new_sl

        assert not (self.pos.be_set and not (self.pos.partial_done or bar_rr >= self.pos.trail_activate_r))

        if self.pos.force_exit:
            return self._close_position(price, "time_stop", ts, i, shared_context, self._strategies.get(self.pos.strategy_name).stats)

        tp = self.pos.tp
        sl = self.pos.sl
        side = self.pos.side  # +1 long, -1 short

        hit_tp = (hi_val >= tp) if side==1 else (lo_val <= tp)
        hit_sl = (lo_val <= sl) if side==1 else (hi_val >= sl)

        prefer_tp = getattr(self, "prefer_tp_if_both", True)
        if hit_tp and hit_sl:
            exit_reason = "tp" if prefer_tp else "sl"
            exit_px = tp if prefer_tp else sl
        elif hit_tp:
            exit_reason = "tp"; exit_px = tp
        elif hit_sl:
            exit_reason = "sl"; exit_px = sl
        elif self.pos.force_exit:
            exit_reason = "time"; exit_px = cl
        else:
            exit_reason = None; exit_px = None

        if exit_reason:
            qty_closed = self.pos.qty  # o parcial si corresponde
            pnl_usd = _pnl_usd(entry_px=self.pos.entry, exit_px=exit_px, qty_closed=qty_closed, side=side, fee_bps=self.fee_bps)
            # self._export_trade(exit_px, exit_reason, pnl_usd) # This function does not exist yet
            return self._close_position(exit_px, exit_reason, ts, i, shared_context, pnl_net=pnl_usd)

        return 0.0

    def _execute_partial(self, exit_px, ts):
        qty_partial = self.pos.qty * self.pos.partial_take_frac
        qty_partial = min(self.pos.qty, qty_partial)
        if qty_partial <= 0:
            return

        pnl_usd = _pnl_usd(self.pos.entry, exit_px, qty_partial, self.pos.side, fee_bps=self.fee_bps)
        self.capital += pnl_usd
        self.partials_count += 1
        self.pos.partial_done = True
        self.pos.partial_filled = True
        self.pos.partial_last_px = exit_px
        self.pos.qty = max(self.pos.qty - qty_partial, 0.0)

        eps = self.pos.partial_be_eps_atr * self.pos.entry_atr if self.pos.entry_atr else 0.0
        break_even_px = self.pos.raw_entry_price + self.pos.side * eps if eps > 0 else self.pos.raw_entry_price
        if self.pos.side == 1:
            self.pos.sl = max(self.pos.sl, break_even_px)
        else:
            self.pos.sl = min(self.pos.sl, break_even_px)
        self.pos.be_set = True

        trade_entry = {
            "ts": ts,
            "entry_ts": self.pos.entry_ts,
            "qty0": getattr(self.pos, "qty0", self.pos.qty),
            "symbol": self.pos.symbol,
            "tf": self.pos.tf,
            "strategy": self.pos.strategy_name or "TrendV2",
            "side": self.pos.side,
            "entry": self.pos.entry,
            "exit": exit_px,
            "exit_reason": "tp_partial",
            "partial": True,
            "pnl": round(pnl_usd, 4),
            "rr": round(self.pos.partial_take_r, 4),
            "bars_open": int(self.pos.bars_open),
            "MFE_ATR": round(self.pos.mfe_atr, 4),
            "MAE_ATR": round(self.pos.mae_atr, 4),
            "max_rr": round(self.pos.max_rr, 4),
            "origin": self.pos.metadata.get('origin'),
            "trade_id": getattr(self.pos, "trade_id", None),
            "tag": self.pos.metadata.get('tag') or "",
        }
        self.trades.append(trade_entry)

    def summary(self):
        n = len(self.trades)
        pnl = sum(t.get("pnl_net", 0.0) for t in self.trades)
        wins = sum(1 for t in self.trades if t.get("pnl_net", 0.0) > 0)
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
        df_trades.to_csv(filename, index=False)
        logging.info(f"Trades exported to {filename}")
