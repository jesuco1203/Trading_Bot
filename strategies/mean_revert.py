from __future__ import annotations
from strategies.base import BaseStrategy, Signal
from typing import Dict, Any

class MeanRevert(BaseStrategy):
    def __init__(self, risk_mult: float = 1.0,
                 gate_pmax_th: float = 0.40, gate_adx_th: float = 16.0, gate_atr_pct_th: float = 0.85,
                 dist_sma_mult: float = 1.1, signal_z_th: float = -1.0, signal_rsi_th: float = 45.0,
                 signal_rbody_th: float = 0.45, signal_lower_wick_th: float = 0.55,
                 sl_mult_atr: float = 1.3, sl_swing_low_bars: int = 3,
                 tp_mult_sl: float = 1.4, tp_mult_atr: float = 1.8,
                 partial_tp_atr_mult: float = 0.8, partial_sl_offset_atr_mult: float = 0.1,
                 local_cooldown_duration: int = 14, gate_adx_max: float = 20.0, time_stop_bars: int = 0, time_stop_mfe_atr: float = 0.0,
                 min_dist_sma_atr: float = 1.1, rr_min: float = 1.4, partial_atr: float = 0.8, partial_sl_offset_atr: float = 0.1):
        super().__init__(name="MeanRevert", risk_mult=risk_mult, time_stop_bars=time_stop_bars, time_stop_mfe_atr=time_stop_mfe_atr)
        self.gate_pmax_th = gate_pmax_th
        self.gate_adx_th = gate_adx_th
        self.gate_atr_pct_th = gate_atr_pct_th
        self.dist_sma_mult = min_dist_sma_atr # Use new parameter
        self.signal_z_th = signal_z_th
        self.signal_rsi_th = signal_rsi_th
        self.signal_rbody_th = signal_rbody_th
        self.signal_lower_wick_th = signal_lower_wick_th
        self.sl_mult_atr = sl_mult_atr
        self.sl_swing_low_bars = sl_swing_low_bars
        self.tp_mult_sl = tp_mult_sl
        self.tp_mult_atr = tp_mult_atr
        self.partial_tp_atr_mult = partial_atr # Use new parameter
        self.partial_sl_offset_atr_mult = partial_sl_offset_atr # Use new parameter
        self.local_cooldown_duration = local_cooldown_duration
        self.gate_adx_max = gate_adx_max
        self.rr_min = rr_min
        self.partial_atr = partial_atr
        self.partial_sl_offset_atr = partial_sl_offset_atr
        self.blocks = {} # Initialize blocks dictionary
        self.mr_regime_bars = 0 # Initialize mr_regime_bars
        self.cooldown = 0 # Initialize cooldown
        self.cooldown_duration = local_cooldown_duration # Initialize cooldown_duration

    def print_summary(self, trades: list):
        print("--- MeanRevert Block Summary ---")
        total_mr_bars = self.mr_regime_bars
        if total_mr_bars > 0:
            for reason, count in self.blocks.items():
                print(f"{reason}: {count} ({count/total_mr_bars:.2%})")
        else:
            for reason, count in self.blocks.items():
                print(f"{reason}: {count}")
        super().print_summary(trades)
        return super().print_summary(trades)

    def signal(self, ctx: Dict[str, Any]) -> Signal | None:
        if ctx.get("shared_context", {}).get("trend_armed", False):
            return Signal("flat", 0.0, None, None, reason="trend_armed_mutex")

        lab = ctx.get("regime_label", "")
        pmax = float(ctx.get("pmax", 0.0))
        
        if lab == "mr":
            self.mr_regime_bars += 1
        
        feats = ctx["feats"]
        df = ctx["df"]
        close = float(df["close"].iat[-1])
        atr_abs = float(feats["atr"].iat[-1])
        sma20 = float(feats["sma20"].iat[-1])
        adx = float(feats["adx"].iat[-1])
        atr_pct = float(feats["atr_pct"].iat[-1])
        atr_p85 = float(ctx["atr_pct_p85"])
        z_score = float(feats["z_score_50"].iat[-1])
        rsi = float(feats["rsi14"].iat[-1])

        # Gate conditions
        gate_ok = True
        if lab != "mr":
            self.blocks["gate_not_mr"] = self.blocks.get("gate_not_mr",0)+1
            gate_ok = False
        if pmax < self.gate_pmax_th:
            self.blocks["gate_pmax_low"] = self.blocks.get("gate_pmax_low",0)+1
            gate_ok = False
        if adx >= self.gate_adx_max:
            self.blocks["gate_adx_high"] = self.blocks.get("gate_adx_high",0)+1
            gate_ok = False
        if atr_pct > atr_p85:
            self.blocks["gate_atr_high"] = self.blocks.get("gate_atr_high",0)+1
            gate_ok = False

        if not gate_ok:
            return Signal("flat",0.0,None,None,reason="mr_gate_fail")

        if self.cooldown > 0:
            self.cooldown -= 1
            self.blocks["cooldown"] = self.blocks.get("cooldown", 0) + 1
            return Signal("flat", 0.0, None, None, reason="cooldown")

        # Estiramiento (Stretch) condition
        dist_sma = abs(close - sma20) / atr_abs if atr_abs > 0 else 0
        if dist_sma < self.dist_sma_mult:
            self.blocks["too_close_to_sma"] = self.blocks.get("too_close_to_sma", 0) + 1
            return Signal("flat", 0.0, None, None, reason="too_close_to_sma")

        o, h, low_price, c = map(float, (df["open"].iat[-1], df["high"].iat[-1], df["low"].iat[-1], df["close"].iat[-1]))
        rng = max(h - low_price, 1e-9)
        rbody_long = (c - o) / rng if rng > 0 and c > o else 0
        lower_wick_pct = (o - low_price) / rng if rng > 0 and o > low_price else 0

        # Signal A: Z-score, RSI, and candle body/wick
        long_a = (z_score <= self.signal_z_th) and (rsi < self.signal_rsi_th) and \
                 (rbody_long >= self.signal_rbody_th or lower_wick_pct >= self.signal_lower_wick_th)

        # Signal B: BB re-entry
        mid = feats["sma20"].iat[-1]
        std = feats["std20"].iat[-1]
        bb_low = float(mid - 2.0 * std)
        prev_close = float(df["close"].iat[-2])
        long_b = (prev_close < bb_low) and (close > bb_low) and (close < sma20)

        if long_a or long_b:
            swing_low_bars = float(df["low"].iloc[-self.sl_swing_low_bars:-1].min())
            
            # SL calculation
            sl_pts = max(close - swing_low_bars, self.sl_mult_atr * atr_abs)
            
            # TP calculation
            tp_target_media = abs(close - sma20)
            tp_raw = min(self.tp_mult_atr * atr_abs, tp_target_media)
            tp_pts = max(self.tp_mult_sl * sl_pts, tp_raw)
            
            # Partial TP and SL offset
            partial_tp_pts = self.partial_tp_atr_mult * atr_abs
            partial_sl_offset_atr_mult = self.partial_sl_offset_atr_mult

            self.cooldown = self.cooldown_duration # Use configurable cooldown
            # self.last_entry_i = ctx["i"] # This is handled by broker
            rr = tp_pts / sl_pts if sl_pts > 0 else 0.0
            if rr < self.rr_min:
                self.blocks["rr_below_min"] = self.blocks.get("rr_below_min", 0) + 1
                return Signal("flat", 0.0, None, None, reason="rr_below_min")
            return Signal("long", 0.7, sl_pts, tp_pts, partial_tp_pts, partial_sl_offset_atr_mult, reason=("z_rsi_candle" if long_a else "bb_reentry"), rr=rr)

        self.blocks["no_setup"] = self.blocks.get("no_setup", 0) + 1
        return Signal("flat", 0.0, None, None, reason="no_setup")

    def warmup_bars(self) -> int:
        return 50