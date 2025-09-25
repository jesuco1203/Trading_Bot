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
        self.blocks = {}

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
        tf = ctx.get("tf", "")
        is_4h = tf == "4h"
        lab = ctx.get("regime_label", "")
        pmax = float(ctx.get("pmax", 0.0))

        ctx['shared_context']['trend_armed'] = self.armed_level is not None

        df = ctx["df"]; feats = ctx["feats"]

        # === PRE-CÁLCULOS SEGUROS ===
        i   = int(ctx.get("i", 0))
        o   = float(df["open"].iat[-1]); h = float(df["high"].iat[-1])
        l   = float(df["low"].iat[-1]);  cl = float(df["close"].iat[-1])

        atr_abs  = float(feats["atr"].iat[-1])
        adx_now  = float(feats["adx"].iat[-1])
        adx_prev = float(feats["adx"].iat[-2]) if len(feats["adx"]) >= 2 else adx_now

        ma_fast = feats.get("ma_fast")
        ma_slow = feats.get("ma_slow")
        if ma_fast is None:
            ma_fast = df["close"].rolling(window=20, min_periods=20).mean()
        if ma_slow is None:
            ma_slow = df["close"].rolling(window=50, min_periods=50).mean()

        # === DIRECCIÓN Y SEPARACIÓN ===
        fast = float(ma_fast.iat[-1]); slow = float(ma_slow.iat[-1])
        sep  = abs(fast - slow) / max(abs(slow), 1e-9)

        # up: fast sube y slow no cae (>= permite mesetas)
        up       = (ma_fast.iat[-1] > ma_fast.iat[-2]) and (ma_slow.iat[-1] >= ma_slow.iat[-2])
        # cross_up: cruce alcista entre fast y slow (segunda vía)
        cross_up = (ma_fast.iat[-2] <= ma_slow.iat[-2]) and (ma_fast.iat[-1] >  ma_slow.iat[-1])

        # === IMPULSO DE BARRA ===
        rng   = max(h - l, 1e-9)
        rbody = max(cl - o, 0.0) / rng              # proporción de cuerpo alcista
        long_wick  = (o - l)/rng >= 0.45            # mecha inferior larga (rechazo)
        bar_expand = (h - l) >= 1.10 * atr_abs      # rango ≥ 1.1×ATR

        # === UMBRALES ADAPTATIVOS ===
        sep_min = 0.004 if adx_now >= 22.0 else 0.006   # antes 0.010/0.006
        adx_abs_ok  = (adx_now >= 22.0)
        adx_slope_ok= ((adx_now - adx_prev) >= 0.4 and adx_now >= 18.0)

        # evidencia de impulso: cualquiera de los 3
        impulse_ok   = (rbody >= 0.35) or long_wick or bar_expand
        # dirección de tendencia: (sube y separa) O cruce
        trend_dir_ok = ((up and sep >= sep_min) or cross_up)
        # ADX: valor absoluto o pendiente
        adx_ok       = (adx_abs_ok or adx_slope_ok)

        arm_ok = trend_dir_ok and adx_ok and impulse_ok

        if not arm_ok and adx_now >= 30 and bar_expand and sep >= 0.004:
            arm_ok = True
            logging.info(f"TREND-GATE i={i} momentum_override=True")

        logging.info(f"TREND-GATE i={i} adx={adx_now:.1f} up={up} sep={sep:.4f} "
             f"trend_dir_ok={trend_dir_ok} adx_ok={adx_ok} impulse_ok={impulse_ok} arm_ok={arm_ok}")

        # contadores de motivo (diagnóstico)
        self.blocks = getattr(self, 'blocks', {})
        if not arm_ok:
            if not trend_dir_ok: self.blocks['gate_trend_dir_fail'] = self.blocks.get('gate_trend_dir_fail',0)+1
            if not adx_ok:       self.blocks['gate_adx_fail']       = self.blocks.get('gate_adx_fail',0)+1
            if not impulse_ok:   self.blocks['gate_impulse_fail']   = self.blocks.get('gate_impulse_fail',0)+1
            return Signal("flat", 0.0, None, None, reason="trend_gate_fail")

        if self.armed_level is None:
            if arm_ok:
                prev_high = float(df["high"].iat[-2])
                h_now     = float(df["high"].iat[-1])
                cl_now    = float(df["close"].iat[-1])

                level_raw = max(prev_high, cl_now)
                self.armed_level = min(level_raw, h_now)
                self.armed_bars  = 0
                self.retested    = False
                self.waiting_pullback = False

                if self.armed_level > h_now:
                    return Signal("flat", 0.0, None, None, reason="invalid_level_rearm")

                logging.info(f"ARM i={i} lvl={self.armed_level:.2f}")
                return Signal("flat", 0.0, None, None, reason="armed")

        if self.armed_level is not None:
            RETEST_EPS      = 0.0015
            MICRO_PULL_ATR  = 0.80
            OVERSHOOT_ATR   = 2.50
            RETURN_EPS      = 0.0050
            WICK_MIN        = 0.20
            TIMEOUT_BARS    = 14

            ma20 = feats.get("ma20", df["close"].rolling(20).mean().iat[-1])

            touched_level = (l <= self.armed_level * (1 - RETEST_EPS) <= h)
            touched_ma20  = (abs(ma20 - self.armed_level)/max(self.armed_level,1e-9) <= 0.003) and (l <= ma20 <= h)

            dist_low_atr = abs(l - self.armed_level) / max(atr_abs, 1e-9)
            lower_wick   = (o - l) / max(h - l, 1e-9)
            micro_pull   = (dist_low_atr <= MICRO_PULL_ATR) and (lower_wick >= WICK_MIN)

            overshoot     = (self.armed_level - l) / max(atr_abs, 1e-9) >= OVERSHOOT_ATR
            return_close  = cl >= self.armed_level * (1 - RETURN_EPS)
            overshoot_rej = overshoot and return_close and (lower_wick >= WICK_MIN)

            self.retested = bool(touched_level or touched_ma20 or micro_pull or overshoot_rej)
            if not self.retested:
                self.armed_bars += 1

                CONT_ADV_ATR  = 0.50
                CONT_MAX_BARS = 6

                adv_from_lvl = (h - self.armed_level) / max(atr_abs, 1e-9)
                if self.armed_level <= h and self.armed_bars <= CONT_MAX_BARS and adv_from_lvl >= CONT_ADV_ATR:
                    swing_low = float(df["low"].iloc[-7:-1].min())
                    sl_pts = max(cl - swing_low, 2.2 * atr_abs)
                    tp_pts = max(1.8 * sl_pts, 2.6 * atr_abs)
                    partial_tp_pts = 1.0 * atr_abs
                    partial_sl_offset_atr_mult = 0.3
                    rr = tp_pts / max(sl_pts, 1e-9)
                    self._disarm()
                    return Signal("long", 0.7, sl_pts, tp_pts, partial_tp_pts, partial_sl_offset_atr_mult,
                                  reason="continuation_enter", rr=rr)

                if self.armed_bars > TIMEOUT_BARS:
                    self._disarm()
                    return Signal("flat", 0.0, None, None, reason="retest_timeout")

                logging.info(f"RETEST_DEBUG i={i} lvl={self.armed_level:.2f} ma20={ma20:.2f} "
                             f"dist_low_atr={dist_low_atr:.2f} wick={lower_wick:.2f} "
                             f"overshoot={overshoot} return_close={return_close} retested={self.retested}")
                return Signal("flat", 0.0, None, None, reason="waiting_retest")
            
            if self.retested:
                swing_low = float(df["low"].iloc[-7:-1].min())
                sl_pts = max(cl - swing_low, 2.0 * atr_abs)
                tp_pts = max(1.8 * sl_pts, 2.6 * atr_abs)
                partial_tp_pts = 1.0 * atr_abs
                partial_sl_offset_atr_mult = 0.3
                rr = tp_pts / max(sl_pts, 1e-9)
                self._disarm()
                return Signal("long", 0.7, sl_pts, tp_pts, partial_tp_pts, partial_sl_offset_atr_mult,
                              reason="pullback_reject_enter", rr=rr)

        return Signal("flat", 0.0, None, None, reason="no_setup")