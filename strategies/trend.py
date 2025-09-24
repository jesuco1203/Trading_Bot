from __future__ import annotations
from strategies.base import BaseStrategy, Signal
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)

class Trend(BaseStrategy):
    name = "Trend"

    def __init__(self, atr_pct_80_percentile: float, risk_mult: float = 1.0,
                 arm_rbody_th: float = 0.55, arm_adx_th: float = 20.0, arm_adx_delta_th: float = 0.5,
                 arm_timeout: int = 10, retest_eps_pct: float = 0.25, reconfirm_mult_1: float = 0.0003,
                 reconfirm_mult_2: float = 0.0008, sl_mult_atr: float = 2.4, tp_mult_sl: float = 1.8,
                 tp_mult_atr: float = 3.0, partial_tp_atr_mult: float = 1.2, partial_sl_offset_atr_mult: float = 0.3,
                 sl_cooldown_duration: int = 3, allow_shorts: bool = True):
        super().__init__(name="Trend", risk_mult=risk_mult)
        self.atr_pct_80_percentile = atr_pct_80_percentile
        self.position_open = False
        self.entry_price = 0.0
        self.sl_price = 0.0
        self.tp_price = 0.0
        self.cooldown = 0
        self.cooldown_duration = sl_cooldown_duration # Use configurable cooldown
        self.last_sl_bar = -100 # To track last SL event

        self.armed_level = None
        self.armed_bars = 0
        self.retested = False
        self.arms = 0 # Initialize arms counter

        # New parameters for BTC 4H Trend
        self.arm_rbody_th = arm_rbody_th
        self.arm_adx_th = arm_adx_th
        self.arm_adx_delta_th = arm_adx_delta_th
        self.arm_timeout = arm_timeout
        self.retest_eps_pct = retest_eps_pct
        self.reconfirm_mult_1 = reconfirm_mult_1
        self.reconfirm_mult_2 = reconfirm_mult_2
        self.sl_mult_atr = sl_mult_atr
        self.tp_mult_sl = tp_mult_sl
        self.tp_mult_atr = tp_mult_atr
        self.partial_tp_atr_mult = partial_tp_atr_mult
        self.partial_sl_offset_atr_mult = partial_sl_offset_atr_mult
        self.allow_shorts = allow_shorts

    def on_stop(self):
        self.cooldown = self.cooldown_duration
        self.armed_level = None
        self.retested = False
        logging.info(f"STOP-LOSS hit. Cooldown activated for {self.cooldown} bars.")



    def warmup_bars(self) -> int:
        return max(60, self.arm_timeout)

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

        # Swing low for SL calculation
        swing_low_9 = float(df["low"].iloc[-9:-1].min())

        if self.armed_level is None:
            arm_long = (
                up and (sep >= sep_min) and (slope_fast > 0) and (slope_slow >= 0) and
                pullback_long and (cl > prev_high) and 
                (rbody >= self.arm_rbody_th) and 
                (adx_now >= self.arm_adx_th) and 
                ((adx_now - adx_prev) >= self.arm_adx_delta_th) and 
                (di_plus > di_minus)
            )
            if arm_long:
                self.armed_level = (prev_high, "long")
                self.armed_bars = 0
                self.retested = False
                logging.info(f"ARM i={ctx['ts']} bars={self.armed_bars} lvl={self.armed_level[0]:.2f}")
                return Signal("flat", 0.0, None, None, reason="arm_bpb")

        if self.armed_level is not None:
            level, side = self.armed_level
            self.armed_bars += 1

            if self.armed_bars < 1:
                return Signal("flat", 0.0, None, None, reason="armed_wait_min1")

            if self.armed_bars > self.arm_timeout: # Use configurable timeout
                self.armed_level = None; self.armed_bars = 0; self.retested = False
                return Signal("flat", 0.0, None, None, reason="disarm_long_timeout")

            # Retest condition
            eps_retest = self.retest_eps_pct / 100
            touched = (l <= level * (1.0 + eps_retest)) or (l <= ma_fast.iat[-1] * (1.0 + eps_retest))
            if touched and close > level and close > ma_fast.iat[-1]:
                self.retested = True

            # Reconfirm condition
            reconf = (close > level * (1.0 + self.reconfirm_mult_1)) or (h > prev_high * (1.0 + self.reconfirm_mult_2))

            lose_struct = not (sep >= 0.8 * sep_min and slope_fast > 0 and slope_slow >= 0)
            if lose_struct:
                self.armed_level = None; self.armed_bars = 0; self.retested = False
                return Signal("flat", 0.0, None, None, reason="disarm_long_losestruct")

            if self.retested and reconf:
                # SHORTS OFF check
                if not self.allow_shorts and side == "short":
                    return Signal("flat", 0.0, None, None, reason="shorts_off")

                # SL calculation
                sl_pts = max(close - swing_low_9, self.sl_mult_atr * atr_abs)
                
                # TP calculation
                tp_pts = max(self.tp_mult_sl * sl_pts, self.tp_mult_atr * atr_abs)
                
                # Partial TP and SL offset
                partial_tp_pts = self.partial_tp_atr_mult * atr_abs
                partial_sl_offset_atr_mult = self.partial_sl_offset_atr_mult

                self.armed_level = None; self.armed_bars = 0; self.retested = False
                return Signal("long", 0.7, sl_pts, tp_pts, partial_tp_pts, partial_sl_offset_atr_mult, reason=f"retest_and_reconf")

            return Signal("flat", 0.0, None, None, reason="armed_wait")

        return Signal("flat", 0.0, None, None, reason="no_setup")
