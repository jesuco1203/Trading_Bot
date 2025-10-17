from __future__ import annotations
from strategies.base import BaseStrategy, Signal
import logging
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
        self.armed_side = 0
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

    def _check_side(self, ctx: dict, side: int):
        df = ctx["df"]
        feats = ctx["feats"]
        i = int(ctx.get("i", 0))

        # Common calcs
        o, h, low_price, cl = float(df["open"].iat[-1]), float(df["high"].iat[-1]), float(df["low"].iat[-1]), float(df["close"].iat[-1])
        atr_abs = float(feats["atr"].iat[-1])
        adx_now = float(feats["adx"].iat[-1])
        adx_prev = float(feats["adx"].iat[-2]) if len(feats["adx"]) >= 2 else adx_now
        ma_fast = float(feats.get("ma_fast", df["close"].rolling(20).mean()).iat[-1])
        ma_slow = float(feats.get("ma_slow", df["close"].rolling(50).mean()).iat[-1])
        sep = abs(ma_fast - ma_slow) / max(abs(ma_slow), 1e-9)

        # DYNAMIC GATES (HMM-driven)
        regime  = ctx.get("regime_label")
        pmax    = float(ctx.get("pmax", 0.0))

        if regime == "trend" and pmax >= 0.70:
            sep_min_dyn  = 0.0025
        else:
            sep_min_dyn  = 0.0040

        adx_abs_ok = (adx_now >= 21.0) # Keep this for normal arming
        adx_slope_ok = ((adx_now - adx_prev) >= 0.4 and adx_now >= 18.0)
        adx_ok = adx_abs_ok or adx_slope_ok

        # Side-dependent direction
        if side > 0:
            trend_dir_ok = (ma_fast > ma_slow) and (sep >= sep_min_dyn)
        else:
            trend_dir_ok = (ma_fast < ma_slow) and (sep >= sep_min_dyn)

        # Side-dependent impulse
        rng = max(h - low_price, 1e-9)
        rbody = max(cl - o, 0.0) / rng if side > 0 else max(o - cl, 0.0) / rng
        wick = (o - low_price) / rng if side > 0 else (h - o) / rng
        bar_expand = (h - low_price) >= 1.20 * atr_abs # Stricter bar expand
        impulse_ok = (rbody >= 0.35) or (wick >= 0.45) or bar_expand

        # Momentum Override Logic
        sc = ctx['shared_context']
        now_i = ctx.get('i', 0)
        sc.setdefault('mo_last_i', -10000)
        sc.setdefault('mo_cooldown_until', -1)
        sc.setdefault('mo_open', False)

        sep_ok = sep >= 0.0040
        adx_abs_ok_mo = adx_now >= 32.0
        adx_slope_ok_mo = (adx_now - adx_prev) >= 0.5
        momentum_override_conditions = bar_expand and sep_ok and (adx_abs_ok_mo or adx_slope_ok_mo)

        mo_allowed = (
            momentum_override_conditions
            and not sc['mo_open']
            and now_i - sc['mo_last_i'] >= 80
            and now_i >= sc['mo_cooldown_until']
        )

        dbg = {
            "bar_expand": bar_expand, "sep": float(sep), "sep_ok": sep_ok,
            "adx_now": float(adx_now), "adx_prev": float(adx_prev),
            "adx_abs_ok_mo": adx_abs_ok_mo, "adx_slope_ok_mo": adx_slope_ok_mo,
            "mo_conditions": momentum_override_conditions, "mo_open": sc.get("mo_open"),
            "mo_last_i": sc.get("mo_last_i"), "now_i": now_i, "mo_cooldown_until": sc.get("mo_cooldown_until"),
        }
        if not mo_allowed:
            if momentum_override_conditions: # Log only when conditions are met but budget fails
                logging.info(f"TREND MO BLOCK: {dbg}")
        else:
            logging.info(f"TREND MO ALLOW: {dbg}")

        arm_ok = trend_dir_ok and adx_ok and impulse_ok
        origin = "normal"

        if mo_allowed:
            arm_ok = True
            origin = "momentum_override"
            sc['mo_open'] = True
            sc['mo_last_i'] = now_i
            # Force entry
            sl_pts = 2.2 * atr_abs
            tp_pts = 1.8 * sl_pts
            self._disarm()
            logging.info(f"TREND MO ENTER (FORCED): i={i} side={side}")
            return Signal("long" if side == 1 else "short", 0.8, sl_pts, tp_pts, reason="momentum_override_forced", rr=(tp_pts/sl_pts), origin='momentum_override')

        # contadores de motivo (diagnóstico)
        self.blocks.setdefault(side, {}).setdefault('gate_trend_dir_fail', 0)
        self.blocks.setdefault(side, {}).setdefault('gate_adx_fail', 0)
        self.blocks.setdefault(side, {}).setdefault('gate_impulse_fail', 0)

        if not arm_ok:
            if not trend_dir_ok:
                self.blocks[side]['gate_trend_dir_fail'] += 1
            if not adx_ok:
                self.blocks[side]['gate_adx_fail'] += 1
            if not impulse_ok:
                self.blocks[side]['gate_impulse_fail'] += 1
            return None

        # Arming Logic
        if self.armed_level is None or self.armed_side != side:
            self.armed_level = cl if side > 0 else cl
            self.armed_side = side
            self.armed_bars = 0
            logging.info(f"ARM i={i} side={side} lvl={self.armed_level:.2f}")
            return None

        # Entry Logic
        if self.armed_level is not None and self.armed_side == side:
            self.armed_bars += 1

            # Retest Logic (now fully symmetrical)
            RETEST_EPS = 0.0015
            OVERSHOOT_ATR = 2.50
            RETURN_EPS = 0.0011  # A2 TUNING
            WICK_MIN = 0.50
            TIMEOUT_BARS = 14

            dist_to_level = (cl - self.armed_level) * side
            
            # Overshoot and rejection
            overshoot_pts = (self.armed_level - low_price) if side > 0 else (h - self.armed_level)
            overshoot = overshoot_pts / max(atr_abs, 1e-9) >= OVERSHOOT_ATR
            return_close = (cl >= self.armed_level * (1 - RETURN_EPS)) if side > 0 else (cl <= self.armed_level * (1 + RETURN_EPS))
            overshoot_rej = overshoot and return_close and (wick >= WICK_MIN)

            retested = overshoot_rej or (abs(dist_to_level) / self.armed_level) < RETEST_EPS

            if retested:
                sl_pts = 2.0 * atr_abs
                tp_pts = 1.8 * sl_pts
                self._disarm()
                return Signal("long" if side == 1 else "short", 0.7, sl_pts, tp_pts, reason="pullback_reject_enter", rr=(tp_pts/sl_pts), origin=origin)

            # Continuation Logic
            CONT_ADV_ATR = 0.80
            CONT_MAX_BARS = 6      # acorta ventana de continuación
            rbody_th = 0.50 if adx_now >= 30 else 0.55
            adv_from_lvl = (cl - self.armed_level) * side / max(atr_abs, 1e-9)

            if self.armed_bars <= CONT_MAX_BARS and adv_from_lvl >= CONT_ADV_ATR and adx_now >= 28.0 and (bar_expand or rbody >= rbody_th):
                sl_pts = 2.2 * atr_abs
                tp_pts = 1.8 * sl_pts
                self._disarm()
                return Signal("long" if side == 1 else "short", 0.7, sl_pts, tp_pts, reason="continuation_enter", rr=(tp_pts/sl_pts), origin=origin)

            if self.armed_bars > TIMEOUT_BARS:
                self._disarm()

        return None

    def signal(self, ctx: dict) -> Signal:
        if self.cooldown > 0:
            self.cooldown -= 1
            return Signal("flat", 0.0, None, None, reason="cooldown")

        # Check for long signal
        long_signal = self._check_side(ctx, side=1)
        if long_signal:
            return long_signal

        # Check for short signal
        if self.allow_shorts:
            short_signal = self._check_side(ctx, side=-1)
            if short_signal:
                return short_signal

        return Signal("flat", 0.0, None, None, reason="no_setup")