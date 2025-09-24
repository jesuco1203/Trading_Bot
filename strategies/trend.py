from __future__ import annotations
from strategies.base import BaseStrategy, Signal
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)

class Trend(BaseStrategy):
    name = "Trend"
    ALLOW_SHORTS = False

    def __init__(self, atr_pct_80_percentile: float = 0.06):
        super().__init__("Trend", risk_mult=1.0)
        self.cooldown = 0
        self.atr_pct_80 = atr_pct_80_percentile
        self.armed_level: Optional[tuple[float, str]] = None
        self.armed_bars = 0
        self.retested = False
        self.arms = 0

    def on_stop(self):
        self.cooldown = 5
        self.armed_level = None
        self.retested = False
        logging.info(f"STOP-LOSS hit. Cooldown activated for {self.cooldown} bars.")

    def print_summary(self, trades: list):
        strat_trades = [t for t in trades if t.get("strategy") == self.name]
        entries = len(set(t['ts'] for t in strat_trades if not t.get("partial")))
        wins = 0
        for ts in set(t['ts'] for t in strat_trades):
            trade_pnl = sum(t["pnl"] for t in strat_trades if t['ts'] == ts)
            if trade_pnl > 0:
                wins += 1
        pnl = sum(t["pnl"] for t in strat_trades)
        hit_rate = (wins / entries) * 100 if entries > 0 else 0
        print(f"--- {self.name} Summary ---")
        print(f"Arms: {self.arms} | Entries: {entries} | Wins: {wins} | Hit Rate: {hit_rate:.2f}% | PnL: {pnl:.2f}")

    def warmup_bars(self) -> int: return 60

    def signal(self, ctx: dict) -> Signal:
        lab = ctx.get("regime_label", "")
        pmax = float(ctx.get("pmax", 0.0))
        if lab != "trend" or pmax < 0.40:
            self.armed_level = None
            self.armed_bars = 0
            self.retested = False
            return Signal("flat", 0.0, None, None, reason="not_trend")

        if self.cooldown > 0:
            self.cooldown -= 1
            return Signal("flat", 0.0, None, None, reason="cooldown")

        df = ctx["df"]; feats = ctx["feats"]
        close = float(df["close"].iat[-1])
        atr_abs = float(feats["atr"].iat[-1])
        adx_now = float(feats["adx"].iat[-1])
        adx_prev = float(feats["adx"].iat[-2])
        di_plus = float(feats.get("di_plus", 0.0).iat[-1])
        di_minus = float(feats.get("di_minus", 0.0).iat[-1])

        if atr_abs <= 0:
            return Signal("flat", 0.0, None, None, reason="low_atr")

        ma_fast = df["close"].rolling(20).mean()
        ma_slow = df["close"].rolling(50).mean()
        up = ma_fast.iat[-1] > ma_slow.iat[-1]
        sep = abs(ma_fast.iat[-1] - ma_slow.iat[-1]) / max(close, 1e-9)
        sep_min = max(0.0006, 0.35 * float(feats["vol"].iat[-1]), 0.35 * (atr_abs/close))
        slope_fast = ma_fast.iat[-1] - ma_fast.iat[-4]
        slope_slow = ma_slow.iat[-1] - ma_slow.iat[-4]
        pullback_long = (df["low"].iat[-1] <= ma_fast.iat[-1] * (1.0 + 0.0025)) and (close > ma_fast.iat[-1])

        o, h, l, cl = map(float, (df["open"].iat[-1], df["high"].iat[-1], df["low"].iat[-1], df["close"].iat[-1]))
        rng = max(h - l, 1e-9)
        rbody = max(cl - o, 0.0) / rng if rng > 0 else 0
        prev_high = float(df["high"].iat[-2])

        if self.armed_level is None:
            arm_long = (
                up and (sep >= sep_min) and (slope_fast > 0) and (slope_slow >= 0) and
                pullback_long and (cl > prev_high) and (rbody >= 0.55) and 
                (adx_now >= 20) and ((adx_now - adx_prev) >= 0.5) and (di_plus > di_minus)
            )
            if arm_long:
                self.armed_level = (prev_high, "long")
                self.armed_bars = 0
                self.retested = False
                self.arms += 1
                logging.info(f"ARM i={ctx['ts']} bars={self.armed_bars} lvl={self.armed_level[0]:.2f}")
                return Signal("flat", 0.0, None, None, reason="arm_bpb")

        if self.armed_level is not None:
            level, side = self.armed_level
            self.armed_bars += 1

            if self.armed_bars < 1:
                return Signal("flat", 0.0, None, None, reason="armed_wait_min1")

            if self.armed_bars > 12:
                self.armed_level = None; self.armed_bars = 0; self.retested = False
                return Signal("flat", 0.0, None, None, reason="disarm_long_timeout")

            eps = 0.0025
            touched = (l <= level * (1.0 + eps)) or (l <= ma_fast.iat[-1] * (1.0 + eps))
            if touched and close > level and close > ma_fast.iat[-1]:
                self.retested = True

            reconf = (close > level * (1.0 + 0.0003)) or (h > prev_high * (1.0 + 0.0008))

            lose_struct = not (sep >= 0.8 * sep_min and slope_fast > 0 and slope_slow >= 0)
            if lose_struct:
                self.armed_level = None; self.armed_bars = 0; self.retested = False
                return Signal("flat", 0.0, None, None, reason="disarm_long_losestruct")

            if self.retested and reconf:
                swing_low = float(df["low"].iloc[-7:-1].min())
                sl_pts = max(close - swing_low, 2.2 * atr_abs)
                tp_pts = max(1.8 * sl_pts, 2.6 * atr_abs)
                partial_tp_pts = 1.0 * atr_abs
                self.armed_level = None; self.armed_bars = 0; self.retested = False
                return Signal("long", 0.7, sl_pts, tp_pts, partial_tp_pts, 0.3, reason=f"retest_and_reconf")

            return Signal("flat", 0.0, None, None, reason="armed_wait")

        return Signal("flat", 0.0, None, None, reason="no_setup")