from __future__ import annotations
from strategies.base import BaseStrategy, Signal
from typing import Dict, Any

import numpy as np

def rolling_percentile(series, window, percentile):
    return series.rolling(window).apply(lambda x: np.percentile(x, percentile), raw=True)

class MeanRevert(BaseStrategy):
    def __init__(self, risk_mult: float = 1.0,
                 gate_pmax_th: float = 0.40, gate_adx_th: float = 16.0, gate_atr_pct_th: float = 0.85,
                 dist_sma_mult: float = 1.1, signal_z_th: float = -1.2, signal_rsi_th: float = 35.0,
                 signal_rbody_th: float = 0.45, signal_lower_wick_th: float = 0.40,
                 sl_mult_atr: float = 1.3, sl_swing_low_bars: int = 3,
                 tp_mult_sl: float = 1.4, tp_mult_atr: float = 1.8,
                 partial_tp_atr_mult: float = 0.8, partial_sl_offset_atr_mult: float = 0.1,
                 local_cooldown_duration: int = 14, gate_adx_max: float = 24.0, time_stop_bars: int = 0, time_stop_mfe_atr: float = 0.0,
                 min_dist_sma_atr: float = 0.25, rr_min: float = 1.4, partial_atr: float = 0.8, partial_sl_offset_atr: float = 0.1):
        super().__init__(name="MeanRevert", risk_mult=risk_mult, time_stop_bars=time_stop_bars, time_stop_mfe_atr=time_stop_mfe_atr)
        self.gate_pmax_th = gate_pmax_th
        self.gate_adx_th = gate_adx_th
        self.gate_atr_pct_th = gate_atr_pct_th
        self.dist_sma_mult = min_dist_sma_atr
        self.signal_z_th = signal_z_th
        self.signal_rsi_th = signal_rsi_th
        self.signal_rbody_th = signal_rbody_th
        self.signal_lower_wick_th = signal_lower_wick_th
        self.sl_mult_atr = sl_mult_atr
        self.sl_swing_low_bars = sl_swing_low_bars
        self.tp_mult_sl = tp_mult_sl
        self.tp_mult_atr = tp_mult_atr
        self.partial_tp_atr_mult = partial_atr
        self.partial_sl_offset_atr_mult = partial_sl_offset_atr
        self.local_cooldown_duration = local_cooldown_duration
        self.gate_adx_max = gate_adx_max
        self.rr_min = rr_min
        self.partial_atr = partial_atr
        self.partial_sl_offset_atr = partial_sl_offset_atr
        self.blocks = {}
        self.mr_regime_bars = 0
        self.cooldown = 0
        self.cooldown_duration = local_cooldown_duration

        self.stats_mfe_atr = []
        self.stats_mae_atr = []

        self.trig_counts = {"don_L":0,"don_S":0,"ema_L":0,"ema_S":0,"atr_L":0,"atr_S":0} # Initialize trig_counts

        # 3) Stats: set seguro + helpers
        self.stats = {
            "trigger_raw_long": 0,
            "trigger_raw_short": 0,
            "passed_filters_long": 0,
            "passed_filters_short": 0,
            "trigger_raw": 0,
            "filters_passed": 0
        }

        # helpers para no volver a fallar por claves faltantes
        def _stat_inc(k, delta=1):
            self.stats[k] = self.stats.get(k, 0) + delta
        def _stat_ensure_keys():
            req = [
                "trigger_raw_long","trigger_raw_short",
                "passed_filters_long","passed_filters_short",
                "trigger_raw","filters_passed"
            ]
            for kk in req:
                if kk not in self.stats:
                    self.stats[kk] = 0
        self._stat_inc = _stat_inc
        self._stat_ensure_keys = _stat_ensure_keys

        # fingerprint + eco de claves al arrancar (debug visible en consola)
        try:
            import hashlib, time
            md5 = hashlib.md5(open(__file__,"rb").read()).hexdigest()
            print(f"[MeanRevert INIT] file={__file__} md5={md5} ts={int(int(time.time()))}")
            print("[MeanRevert INIT] stats keys:", sorted(self.stats.keys()))
        except Exception:
            pass

    def signal(self, ctx: Dict[str, Any]) -> Signal | None:
        # --- Blindeo de stats: tipo dict + claves mínimas
        if not isinstance(self.stats, dict):
            try:
                self.logger.error(f"[STATS] self.stats dejó de ser {type(self.stats)}; rehidratando dict")
            except Exception:
                pass
            self.stats = {}
        self._stat_ensure_keys()

        feats = ctx["feats"]
        df = ctx["df"]
        
        # New strict gates
        adx_now = feats["adx"].iat[-1]
        atr_series = feats["atr"]
        atr_pctl = rolling_percentile(atr_series, window=200, percentile=70).iat[-1]
        trend_hmm = (ctx.get("regime_label") == "trend" and float(ctx.get("pmax",0.0)) >= 0.70)

        block_trend = (adx_now >= 24.0) or trend_hmm
        block_vol = (atr_pctl >= 0.70) # This requires atr_pctl to be a single value

        if block_trend or block_vol:
            self.blocks["mr_gated_by_trend_vol"] = self.blocks.get("mr_gated_by_trend_vol", 0) + 1
            return Signal("flat", 0.0, None, None, reason="mr_gated_by_trend_vol")

        # ... (rest of the signal logic) ...
        close = float(df["close"].iat[-1])
        atr_abs = float(feats["atr"].iat[-1])
        sma20 = float(feats["sma20"].iat[-1])
        z_score = float(feats["z_score_50"].iat[-1])
        rsi = float(feats["rsi14"].iat[-1])

        dist_sma = abs(close - sma20) / atr_abs if atr_abs > 0 else 0
        if dist_sma < self.dist_sma_mult:
            self.blocks["too_close_to_sma"] = self.blocks.get("too_close_to_sma", 0) + 1
            return Signal("flat", 0.0, None, None, reason="too_close_to_sma")

        o, h, low_price, c = map(float, (df["open"].iat[-1], df["high"].iat[-1], df["low"].iat[-1], df["close"].iat[-1]))
        rng = max(h - low_price, 1e-9)
        lower_wick_pct = (min(o, c) - low_price) / rng if rng > 0 else 0

        ok_z = (z_score <= self.signal_z_th)
        ok_rsi = (rsi <= self.signal_rsi_th)
        ok_dist = dist_sma >= self.dist_sma_mult
        ok_wick = lower_wick_pct >= self.signal_lower_wick_th

        # --- Contabiliza triggers RAW SIEMPRE con helpers ---
        # For MeanRevert, a raw trigger is when all confluence conditions are met
        if ok_z and ok_rsi and ok_dist and ok_wick:
            self._stat_inc("trigger_raw_long", 1) # MeanRevert only generates long signals
            self._stat_inc("trigger_raw", 1)

        if not (ok_z and ok_rsi and ok_dist and ok_wick):
            self.blocks["mr_filter_confluence_fail"] = self.blocks.get("mr_filter_confluence_fail", 0) + 1
            return Signal("flat", 0.0, None, None, reason="mr_filter_confluence_fail")
        
        # If all checks pass, generate signal
        sl_pts = self.sl_mult_atr * atr_abs
        tp_pts = self.tp_mult_sl * sl_pts
        rr = tp_pts / sl_pts if sl_pts > 0 else 0.0
        if rr < self.rr_min:
            return Signal("flat", 0.0, None, None, reason="rr_below_min")

        # If a signal is generated, increment passed filters
        self._stat_inc("passed_filters_long", 1)
        self._stat_inc("filters_passed", 1)
        return Signal("long", 0.7, sl_pts, tp_pts, reason="mr_confluence_ok", rr=rr)

    def warmup_bars(self) -> int:
        return 50