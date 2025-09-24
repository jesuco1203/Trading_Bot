from __future__ import annotations
from strategies.base import BaseStrategy, Signal
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
import math
logger = logging.getLogger(__name__)

class Trend(BaseStrategy):
    name = "Trend"

    def __init__(self, atr_pct_80_percentile: float, risk_mult: float = 1.0,
                 arm_rbody_th: float = 0.55, arm_adx_th: float = 20.0, arm_adx_delta_th: float = 0.5,
                 arm_timeout: int = 10, retest_eps_pct: float = 0.25, reconfirm_mult_1: float = 0.0003,
                 reconfirm_mult_2: float = 0.0008, sl_mult_atr: float = 2.4, tp_mult_sl: float = 1.8,
                 tp_mult_atr: float = 3.0, partial_tp_atr_mult: float = 1.2, partial_sl_offset_atr_mult: float = 0.3,
                 sl_cooldown_duration: int = 8, allow_shorts: bool = True,
                 min_pmax: float = 0.40, max_dist_ma20_atr: float = 1.6, time_stop_bars: int = 0, time_stop_mfe_atr: float = 0.0,
                 ma_fast_len: int = 20, ma_slow_len: int = 50, sep_min: float = 0.002):
        super().__init__(name="Trend", risk_mult=risk_mult, time_stop_bars=time_stop_bars, time_stop_mfe_atr=time_stop_mfe_atr)
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
        self.reconf_bar = -1            # ← default
        self.waiting_pullback = False    # ← default

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

        self.min_pmax = min_pmax
        self.max_dist_ma20_atr = max_dist_ma20_atr
        self.time_stop_bars = time_stop_bars
        self.time_stop_mfe_atr = time_stop_mfe_atr
        self.ma_fast_len = ma_fast_len
        self.ma_slow_len = ma_slow_len
        self.sep_min = getattr(self, "sep_min", 0.0010)

    def on_stop(self):
        self.cooldown = self.cooldown_duration
        self._disarm()
        logging.info(f"STOP-LOSS hit. Cooldown activated for {self.cooldown} bars.")

    def _disarm(self):
        self.armed_level = None
        self.armed_bars  = 0
        self.retested    = False
        self.waiting_pullback = False
        self.reconf_bar  = -1



    def warmup_bars(self) -> int:
        return max(60, self.arm_timeout)

    def signal(self, ctx: dict) -> Signal:
        lab = ctx.get("regime_label", "")
        pmax = float(ctx.get("pmax", 0.0))

        df = ctx["df"]; feats = ctx["feats"]

        # === en strategies/trend.py, dentro de signal() ===
        i   = int(ctx.get("i", 0))
        o   = float(df["open"].iat[-1])
        h   = float(df["high"].iat[-1])
        l   = float(df["low"].iat[-1])
        cl  = float(df["close"].iat[-1])

        atr_abs = float(feats["atr"].iat[-1])                 # ATR en precio
        adx_now = float(feats["adx"].iat[-1])
        adx_prev= float(feats["adx"].iat[-2]) if len(feats["adx"])>=2 else adx_now
        ma_fast = feats.get("ma_fast") or df["close"].rolling(self.ma_fast_len, min_periods=self.ma_fast_len).mean()
        ma_slow = feats.get("ma_slow") or df["close"].rolling(self.ma_slow_len, min_periods=self.ma_slow_len).mean()

        if atr_abs <= 0:
            return Signal("flat", 0.0, None, None, reason="low_atr")

        atr_pct = (atr_abs / cl) if cl > 0 else 0.0
        atr_p90 = ctx["atr_pct_p90"]
        if atr_pct > atr_p90:
            return Signal("flat", 0.0, None, None, reason="atr_vol_too_high")



        # overextension guard
        ma20 = ma_fast.iat[-1]
        dist_ma20_atr = abs(cl - ma20) / atr_abs if atr_abs > 0 else 0
        if dist_ma20_atr > self.max_dist_ma20_atr:
            return Signal("flat", 0.0, None, None, reason="overextended_vs_ma20")
        
        diff = float(ma_fast.iat[-1] - ma_slow.iat[-1])
        den  = max(abs(float(ma_slow.iat[-1])), 1e-9)
        sep  = abs(diff) / den
        # if sep ~0 por redondeo, imprime y aborta sólo esa barra
        if sep < 1e-6 or math.isnan(sep):
            logging.warning(f"TREND-MA-ANOM i={ctx['i']} fast={fast} slow={slow} sep={sep}")
        logging.debug(f"MA_CALC i={ctx['i']} fast={ma_fast.iat[-1]:.2f} slow={ma_slow.iat[-1]:.2f} diff={diff:.2f} den={den:.2f} sep={sep:.4f}")

        up  = (ma_fast.iat[-1] > ma_fast.iat[-2]) and (ma_slow.iat[-1] >= ma_slow.iat[-2])
        cross_up = (ma_fast.iat[-2] <= ma_slow.iat[-2]) and (ma_fast.iat[-1] > ma_slow.iat[-1])

        slope_fast = ma_fast.iat[-1] - ma_fast.iat[-4]
        slope_slow = ma_slow.iat[-1] - ma_slow.iat[-4]
        pullback_long = (df["low"].iat[-1] <= ma_fast.iat[-1] * (1.0 + 0.0025)) and (cl > ma_fast.iat[-1])


        rng = max(h - l, 1e-9)
        rbody = max(cl - o, 0.0) / rng if rng > 0 else 0
        long_wick = (o - l)/rng >= 0.45
        prev_high = float(df["high"].iat[-2])

        arm_long = ((up and sep >= self.sep_min) or cross_up) \
                   and (adx_now >= 18) and ((adx_now - adx_prev) >= 0.4) \
                   and (rbody >= 0.40 or (o - l)/rng >= 0.45)

        # TREND-GATE log
        if ctx['i'] % 200 == 0 and lab == "trend":
            logging.info(f"MA fast={ma_fast.iat[-1]:.2f} slow={ma_slow.iat[-1]:.2f} diff={ma_fast.iat[-1]-ma_slow.iat[-1]:.2f}")
        logging.debug(f"TREND-GATE i={ctx['i']} up={up} sep={sep:.4f} adx={adx_now:.1f} rbody={rbody:.2f} arm_ok={arm_long}")

        # Swing low for SL calculation
        swing_low_9 = float(df["low"].iloc[-9:-1].min())

        if self.armed_level is None:
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
            RETEST_EPS = 0.0025    # 0.25%
            touched = (l <= level*(1 + (-RETEST_EPS)) <= h) or (abs(l - level)/level <= RETEST_EPS)
            logging.debug(f"RETEST_DEBUG i={ctx['i']} touched={touched} l={l:.2f} level={level:.2f} eps={RETEST_EPS:.4f} ma_fast={ma_fast.iat[-1]:.2f} close={cl:.2f}")
            if touched and cl > level and cl > ma_fast.iat[-1]:
                self.retested = True
            
            TIMEOUT_BARS = 16
            if self.armed_bars > TIMEOUT_BARS and not touched:
                self.armed_level = None; self.armed_bars = 0; self.retested = False
                return Signal("flat", 0.0, None, None, reason="retest_timeout")

            # Reconfirm condition
            RECONF_CLOSE_BPS = 30   # 0.30% por cierre
            RECONF_HIGH_BPS  = 8    # 0.08% por ruptura de máximo previo

            reconf_close = cl > level * (1 + RECONF_CLOSE_BPS/10000.0)
            reconf_high  = h  > prev_high * (1 + RECONF_HIGH_BPS/10000.0)
            reconf = reconf_close or reconf_high
            logging.debug(f"RECONF_DEBUG i={ctx['i']} reconf={reconf} close={cl:.2f} level={level:.2f} reconf_close={reconf_close} h={h:.2f} prev_high={prev_high:.2f} reconf_high={reconf_high}")

            lose_struct = not (sep >= 0.8 * self.sep_min and slope_fast > 0 and slope_slow >= 0)
            if lose_struct:
                self._disarm()
                return Signal("flat", 0.0, None, None, reason="disarm_long_losestruct")

            # Guardia de vela extendida
            tr = h - l
            if (tr >= 1.3 * atr_abs) or (rbody >= 0.85):
                return Signal("flat", 0.0, None, None, reason="reconf_extended_bar")

            if self.retested and (reconf_close or reconf_high):
                # SHORTS OFF check
                if not self.allow_shorts and side == "short":
                    self._disarm()
                    return Signal("flat", 0.0, None, None, reason="shorts_off")
                self.waiting_pullback = True
                self.reconf_bar = ctx['i']
                return Signal("flat", 0.0, None, None, reason="waiting_pullback")

            if self.waiting_pullback:
                PULLBACK_ATR = 0.20   # retest del nivel tras el breakout
                PULLBACK_TIMEOUT = 6      # barras para esperar el retest tras reconfirm

                pull = level + PULLBACK_ATR * atr_abs # level + 0.2*ATR
                touched_pull = (l <= pull <= h)
                bullish_reject = (cl > o) and ((o - l)/(h - l + 1e-9) >= 0.45)

                if touched_pull and bullish_reject:
                    # SL calculation
                    swing_low = float(df["low"].iloc[-9:-1].min())
                    sl_pts = max(close - (swing_low - 0.6 * atr_abs), 3.0 * atr_abs)     # ↑ SL para 4H
                    
                    # TP calculation
                    tp_pts = max(1.8 * sl_pts, 3.0 * atr_abs)
                    
                    # Partial TP and SL offset
                    partial_tp_pts = 1.2 * atr_abs
                    partial_sl_offset_atr_mult = 0.3

                    self._disarm()
                    rr = tp_pts / sl_pts if sl_pts > 0 else 0.0
                    return Signal("long", 0.7, sl_pts, tp_pts, partial_tp_pts, partial_sl_offset_atr_mult, reason="bo_retest_enter", rr=rr)

                # timeout del pullback
                if ctx['i'] - self.reconf_bar >= PULLBACK_TIMEOUT:
                    self._disarm()
                    return Signal("flat", 0.0, None, None, reason="pullback_timeout")

            return Signal("flat", 0.0, None, None, reason="waiting_retest_or_reconf")

        return Signal("flat", 0.0, None, None, reason="no_setup")