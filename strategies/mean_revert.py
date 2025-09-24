from __future__ import annotations
from strategies.base import BaseStrategy, Signal
from typing import Dict, Any
import pandas as pd
import numpy as np

class MeanRevert(BaseStrategy):
    def __init__(self):
        super().__init__("MeanRevert", risk_mult=0.6)
        self.cooldown = 0
        self.blocks = {}
        self.last_entry_i = -10**9
        self.local_cd = 12

    def print_summary(self, trades: list):
        print("--- MeanRevert Block Summary ---")
        for reason, count in self.blocks.items():
            print(f"{reason}: {count}")
        super().print_summary(trades)

    def signal(self, ctx: Dict[str, Any]) -> Signal | None:
        lab = ctx.get("regime_label", "")
        pmax = float(ctx.get("pmax", 0.0))
        if lab != "mr" or pmax < 0.40:
            self.blocks["not_mr"] = self.blocks.get("not_mr", 0) + 1
            return Signal("flat", 0.0, None, None, reason="not_mr")

        if (ctx["i"] - self.last_entry_i) < self.local_cd:
            self.blocks["mr_local_cooldown"] = self.blocks.get("mr_local_cooldown", 0) + 1
            return Signal("flat", 0.0, None, None, reason="mr_local_cooldown")

        feats = ctx["feats"]
        df = ctx["df"]
        close = float(df["close"].iat[-1])
        atr_abs = float(feats["atr"].iat[-1])
        sma20 = float(feats["sma20"].iat[-1])

        dist_sma = abs(close - sma20) / atr_abs if atr_abs > 0 else 0
        if dist_sma < 1.0:
            self.blocks["too_close_to_sma"] = self.blocks.get("too_close_to_sma", 0) + 1
            return Signal("flat", 0.0, None, None, reason="too_close_to_sma")

        atr_pct = float(feats["atr_pct"].iat[-1])
        atr_p85 = float(ctx["atr_pct_p85"])
        adx = float(feats["adx"].iat[-1])
        z = float(feats["z_score_50"].iat[-1])
        rsi = float(feats["rsi14"].iat[-1])

        if adx >= 16 or atr_pct > atr_p85:
            self.blocks["filtered"] = self.blocks.get("filtered", 0) + 1
            return Signal("flat", 0.0, None, None, reason="filtered")

        o, h, l, c = map(float, (df["open"].iat[-1], df["high"].iat[-1], df["low"].iat[-1], df["close"].iat[-1]))
        rng = max(h - l, 1e-9)
        bull_body = c > o and (c - o) / rng >= 0.45
        long_wick = (o - l) / rng >= 0.55
        long_a = (z <= -1.0) and (rsi < 45) and (bull_body or long_wick)

        mid = feats["sma20"].iat[-1]
        std = feats["std20"].iat[-1]
        bb_low = float(mid - 2.0 * std)
        prev_close = float(df["close"].iat[-2])
        long_b = (prev_close < bb_low) and (close > bb_low) and (close < sma20)

        if long_a or long_b:
            swing_low = float(df["low"].iloc[-3:-1].min())
            sl_pts = max(close - swing_low, 1.3 * atr_abs)
            tp_target_media = abs(close - sma20)
            tp_raw = min(1.8 * atr_abs, tp_target_media)
            tp_pts = max(1.4 * sl_pts, tp_raw)
            partial_tp_pts = 0.8 * atr_abs
            partial_sl_offset_atr_mult = 0.1 # BE + 0.1*ATR
            self.cooldown = 2
            self.last_entry_i = ctx["i"]
            return Signal("long", 0.7, sl_pts, tp_pts, partial_tp_pts, partial_sl_offset_atr_mult, reason=("z_rsi" if long_a else "bb_reentry"))

        self.blocks["no_setup"] = self.blocks.get("no_setup", 0) + 1
        return Signal("flat", 0.0, None, None, reason="no_setup")

    def warmup_bars(self) -> int:
        return 50