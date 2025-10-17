import numpy as np
import pandas as pd
from typing import Optional, Union
import logging
from types import SimpleNamespace # Added for signal method
from collections import defaultdict

from strategies.base import Signal, BaseStrategy
from regime.bias import regime_bias

def _to_series(x: Union[pd.Series, pd.DataFrame, float, int], col_hint: Optional[str] = None) -> pd.Series:
    """Normaliza x a pandas.Series (soporta escalar, Serie o DataFrame)."""
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        if col_hint and col_hint in x.columns:
            return x[col_hint]
        # Si es DataFrame, toma la primera columna como serie
        return x.iloc[:, 0]
    if np.isscalar(x):
        # Construye Serie dummy con un único valor (evitar en producción; mejor no llegar aquí)
        return pd.Series([x])
    return pd.Series([np.nan]) # Fallback

def _safe_val(series_like: pd.Series, i: int, default: float = np.nan) -> float:
    """Devuelve float(series_like.iloc[i]) con guardas contra NaN/bordes/escalar."""
    try:
        v = float(series_like.iloc[i])
        return v if np.isfinite(v) else default
    except (IndexError, TypeError, KeyError):
        return default


# strategies/trend_v2.py
# --- Fingerprint opcional (útil para verificar que este archivo es el que carga Python) ---
try:
    import hashlib, time
    print(f"[TRENDV2] USING FILE={__file__} MD5={hashlib.md5(open(__file__,'rb').read()).hexdigest()} TS={int(time.time())}")
except Exception:
    pass


class TrendV2(BaseStrategy):
    def __init__(self, cfg, logger):
        super().__init__("TrendV2", 100)
        self.cfg = cfg
        self.logger = logger
        t = self.cfg.get("trend_v2", {})
        self.diag = bool(t.get("diag_triggers", False))

        # === Estado de re-entry ===
        self._reentry_armed = False
        self._reentry_until = -1
        self._reentry_side  = 0
        
        self.reentry_armed_count = 0
        self.reentry_exec_count = 0
        self.re_dbg = defaultdict(int)
        self.re_dists = []
        self.re_lateness_bars = []
        self.re_rr_snapshot = []
        self.re_window_hits = []
        self._reentry_exit_i = None
        self._reentry_exit_price = None
        self._reentry_entry_price = None
        self._pending_reentry = None

        # === Contadores/estadísticas seguras ===
        self.trig_counts = {"don_L":0,"don_S":0,"ema_L":0,"ema_S":0,"atr_L":0,"atr_S":0}
        self.blocks = {"no_bias_long": 0, "no_bias_short": 0, "no_volume_ok": 0}
        self.audit_rows = []
        self.stats = {
            "trigger_raw_long": 0,
            "trigger_raw_short": 0,
            "passed_filters_long": 0,
            "passed_filters_short": 0,
            "trigger_raw": 0,
            "filters_passed": 0,
        }
        self.debug_counts = {
            "ema_sep_rejects_L": 0,
            "ema_sep_rejects_S": 0,
            "don_sep_rejects_L": 0,
            "don_sep_rejects_S": 0,
        }
        self._sep_samples = []
        self._don_sep_samples = []

        # Helpers seguros para stats (evitan KeyError)
        def _stat_inc(k, delta=1):
            self.stats[k] = self.stats.get(k, 0) + delta

        def _stat_ensure_keys():
            for kk in (
                "trigger_raw_long",
                "trigger_raw_short",
                "passed_filters_long",
                "passed_filters_short",
                "trigger_raw",
                "filters_passed",
            ):
                if kk not in self.stats:
                    self.stats[kk] = 0

        self._stat_inc = _stat_inc
        self._stat_ensure_keys = _stat_ensure_keys

        self.stats_mfe_atr = []
        self.stats_mae_atr = []
        # Pre-seed keys to keep telemetry stable
        for key in (
            "reentry_exec",
            "relax_gate_used",
            "reject_pending_dup",
            "reject_exec_fail",
            "relax_candidates",
            "relax_rejected",
        ):
            _ = self.re_dbg[key]
        self.logger.info(f"[TRENDV2 INIT DEBUG] self.diag = {self.diag}")

    @staticmethod
    def _percentiles(samples, points=(50, 75, 90, 95, 99)):
        if not samples:
            return {}
        xs = sorted(samples)
        n = len(xs)

        def pct(p):
            k = (p / 100.0) * (n - 1)
            f = int(k)
            c = min(f + 1, n - 1)
            w = k - f
            return xs[f] * (1 - w) + xs[c] * w

        stats = {f"p{p}": pct(p) for p in points}
        stats["min"] = xs[0]
        stats["max"] = xs[-1]
        return stats

    # ---------------------------------------------------------------------
    # on_exit: DEBE quedar al nivel de método de clase (4 espacios)
    # ---------------------------------------------------------------------
    def on_exit(self, reason, bars_open, side, i_local, exit_price=None, entry_price=None):
        """Arma re-entry sólo si SL temprano. NO desarmar por duplicados."""
        try:
            self.logger.info(
                f"[REENTRY on_exit] reason={reason} bars_open={bars_open} side={side} i_local={i_local} "
                f"armed={self._reentry_armed} until={self._reentry_until}"
            )
        except Exception:
            pass
        # reset any pending snapshot when trade closes/cancels
        self._clear_pending_reentry()

        # Índice válido
        if (i_local is None) or (not isinstance(i_local, int)) or (i_local < 0):
            try: self.logger.error(f"[REENTRY] índice inválido en on_exit: i_local={i_local}")
            except Exception: pass
            return

        if not hasattr(self, "_reentry_last_on_exit_i"):
            self._reentry_last_on_exit_i = None

        i_local_int = int(i_local)

        # Debounce: evita procesar dos veces el MISMO i_local
        if self._reentry_last_on_exit_i == i_local_int:
            return
        self._reentry_last_on_exit_i = i_local_int

        # Si ya estamos ARMADOS y la ventana sigue vigente, NO toques nada
        if self._reentry_armed and i_local_int <= self._reentry_until:
            return

        # Armar sólo para SL temprano; no desarmar en otros casos
        cfg_t = self.cfg.get("trend_v2", {})
        threshold = int(getattr(cfg_t, "reentry_sl_bars_open_max", 2))
        win       = int(getattr(cfg_t, "reentry_window", 16))
        if reason == "sl" and bars_open <= threshold and not self._reentry_armed:
            self._reentry_armed = True
            self._reentry_until = int(i_local) + max(8, win)
            self._reentry_side  = int(side)
            self.reentry_armed_count += 1
            self._reentry_exit_i = i_local_int
            self._reentry_exit_price = float(exit_price) if exit_price is not None else None
            self._reentry_entry_price = float(entry_price) if entry_price is not None else None
            self.re_dbg["armed_events"] += 1
            self.logger.info(f"[REENTRY ARMED] side={side} i_local={i_local} until={self._reentry_until}")
            return
        else:
            key = "reject_not_sl_fast"
            if reason != "sl":
                key = f"reject_reason_{reason}"
            elif bars_open > threshold:
                key = "reject_sl_too_late"
            elif self._reentry_armed:
                key = "reject_already_armed"
            self.re_dbg[key] += 1

    def _disarm_reentry(self):
        self._reentry_armed = False
        self._reentry_until = -1
        self._reentry_side = 0
        self._reentry_exit_i = None
        self._reentry_exit_price = None
        self._reentry_entry_price = None

    def _clear_pending_reentry(self):
        self._pending_reentry = None

    def _reject_sl_too_late(self, bars_since_exit: int, max_allowed: int) -> None:
        lateness = max(0, bars_since_exit - max_allowed)
        self.re_lateness_bars.append(lateness)
        self.re_dbg["reject_sl_too_late"] += 1

    def _trend_allows_reentry(self, side: int, feats: dict, i_local: int) -> bool:
        bias_series = feats.get("bias")
        bias_val = 0.0
        try:
            if isinstance(bias_series, pd.Series) and i_local < len(bias_series):
                bias_val = float(bias_series.iloc[i_local])
            else:
                bias_val = float(bias_series) if np.isscalar(bias_series) else float(feats.get("bias", 0.0))
        except Exception:
            bias_val = float(feats.get("bias", 0.0))
        if side > 0:
            return bias_val >= 0.0
        if side < 0:
            return bias_val <= 0.0
        return False

    def _adx_is_rising(self, i_local: int, feats: dict) -> bool:
        adx_series = feats.get("adx")
        if isinstance(adx_series, pd.Series):
            try:
                if i_local <= 0 or i_local >= len(adx_series):
                    return True
                current = float(adx_series.iloc[i_local])
                prev = float(adx_series.iloc[i_local - 1])
                return current >= prev
            except Exception:
                return True
        return True

    def _estimate_rr(self, price: float, ref_price: float, atr_value: float, side: int) -> Optional[float]:
        atr_value = float(atr_value or 0.0)
        if atr_value <= 0.0:
            return None
        tv2 = self.cfg.get("trend_v2", {})
        exits_cfg = self.cfg.get("exits", {})
        sl_mult = float(tv2.get("sl_atr_mult", exits_cfg.get("sl_atr", 1.0)) or 1.0)
        risk = max(sl_mult * atr_value, 1e-8)
        reward_dist = abs(price - float(ref_price or price))
        return reward_dist / risk if reward_dist > 0 else 0.0

    def _structure_ok_for_reentry(self, side: int, market: dict, i_local: int) -> bool:
        # Placeholder: currently no extra structural filters for reentry
        return True

    def register_reentry_execution(self, success: bool) -> None:
        """Actual execution outcome reported by orchestrator."""
        if success:
            self.reentry_exec_count += 1
            self.re_dbg["reentry_exec"] += 1
        else:
            self.re_dbg["reject_exec_fail"] += 1

    def _diagnose_reentry(self, i_local: int, market: dict, feats: dict, cfg_t: SimpleNamespace):
        if not self._reentry_armed:
            return
        self._pending_reentry = None
        exit_i = getattr(self, "_reentry_exit_i", None)
        if exit_i is None:
            self.re_dbg["reject_missing_exit_i"] += 1
            self._disarm_reentry()
            return

        bars_since = i_local - exit_i
        self.re_window_hits.append(bars_since)

        if bars_since < 1:
            self.re_dbg["reject_too_soon"] += 1
            return

        window = int(getattr(cfg_t, "reentry_window", 16))
        max_window = max(8, window)
        if bars_since > max_window or i_local > self._reentry_until:
            self.re_dbg["reject_window_expired"] += 1
            self._disarm_reentry()
            return

        side = self._reentry_side
        if side == 0:
            self.re_dbg["reject_no_side"] += 1
            self._disarm_reentry()
            return

        if not self._trend_allows_reentry(side, feats, i_local):
            self.re_dbg["reject_trend_mismatch"] += 1
            return

        df_ref = getattr(self, "_df_ref", None)
        if df_ref is None or "close" not in df_ref or "atr" not in df_ref:
            self.re_dbg["reject_missing_df"] += 1
            self._disarm_reentry()
            return
        if i_local <= 0 or i_local >= len(df_ref):
            self.re_dbg["reject_index_bounds"] += 1
            return

        try:
            prev_row = df_ref.iloc[i_local - 1]
            atr_prev = float(prev_row.get("atr", 0.0) or 0.0)
            close_prev = float(prev_row.get("close", 0.0) or 0.0)
        except Exception:
            self.re_dbg["reject_prev_row"] += 1
            return

        ref_px = self._reentry_exit_price
        if ref_px is None:
            try:
                ref_px = float(df_ref["close"].iloc[exit_i])
            except Exception:
                ref_px = close_prev

        atr_prev = max(atr_prev, 1e-12)
        eps_atr = abs(close_prev - ref_px) / atr_prev
        self.re_dists.append(eps_atr)
        if eps_atr < float(getattr(cfg_t, "reentry_eps_atr", 0.0)):
            self.re_dbg["reject_eps_too_small"] += 1
            return

        max_sl_bars = int(getattr(cfg_t, "reentry_sl_bars_open_max", 16) or 16)
        lateness = max(0, bars_since - max_sl_bars)
        self.re_lateness_bars.append(lateness)
        rr_est = self._estimate_rr(close_prev, ref_px, atr_prev, side)
        if rr_est is not None:
            self.re_rr_snapshot.append(rr_est)

        if bars_since > max_sl_bars:
            relax_cap = int(getattr(cfg_t, "reentry_sl_bars_open_relax", 0) or 0)
            can_relax = False
            self.re_dbg["relax_candidates"] += 1
            if relax_cap > 0 and bars_since <= max_sl_bars + relax_cap:
                rr_ok = rr_est is not None and rr_est >= float(getattr(cfg_t, "reentry_rr_min", 0.7) or 0.0)
                adx_ok = True
                if bool(getattr(cfg_t, "reentry_adx_uplift", False)):
                    adx_ok = self._adx_is_rising(i_local, feats)
                can_relax = rr_ok and adx_ok
            if can_relax:
                self.re_dbg["relax_gate_used"] += 1
            else:
                self.re_dbg["relax_rejected"] += 1
                self._reject_sl_too_late(bars_since, max_sl_bars)
                self._disarm_reentry()
                return

        if not self._structure_ok_for_reentry(side, market, i_local):
            self.re_dbg["reject_structure"] += 1
            return

        self.re_dbg["passes_all"] += 1
        exec_enabled = bool(getattr(cfg_t, "reentry_exec_enabled", False))
        if exec_enabled:
            entry_side = "long" if side > 0 else "short"
            if self._pending_reentry is not None:
                self.re_dbg["reject_pending_dup"] += 1
            else:
                self._pending_reentry = {
                    "side": entry_side,
                    "side_sign": side,
                    "ref_price": ref_px,
                    "bars_since": bars_since,
                    "rr_est": rr_est,
                }
        else:
            self.re_dbg["reject_no_execution_logic"] += 1
        self._disarm_reentry()

    def _check_signal(self, i_local: int, data: dict) -> tuple[str, str]:
        """Core logic: evaluate feature data and decide on a signal."""
        cfg_t = self.cfg.get('trend_v2', {})
        
        # --- 1. Extract all data points ---
        bias = data.get('bias', 0.0)
        adx_now = data.get('adx_now', 0.0)
        filters_ok = data.get('filters_ok', True)
        trigger_long = data.get('trigger_long', False)
        trigger_short = data.get('trigger_short', False)
        don_ok_L = data.get('don_long_ok', False)
        don_ok_S = data.get('don_short_ok', False)
        ema_ok_L = data.get('ema_long_ok', False)
        ema_ok_S = data.get('ema_short_ok', False)
        origin = data.get('origin', 'n/a')
        min_adx = float(cfg_t.get("min_adx", 15.0))

        # --- 2. Unified Gate Logic & Logging ---
        def _entry_log(side, reason):
            self.logger.info(
                f"[DECISION ENTRY] side={side} i={i_local} reason={reason} "
                f"bias={bias:.2f} adx={adx_now:.2f} min_adx={min_adx:.2f} "
                f"don_ok_L={don_ok_L} don_ok_S={don_ok_S} ema_ok_L={ema_ok_L} ema_ok_S={ema_ok_S} "
                f"filters={filters_ok}"
            )

        def _no_entry_log(side, reasons):
            self.logger.info(
                f"[DECISION NO-ENTRY] side={side} i={i_local} bias={bias:.2f} adx={adx_now:.2f} "
                f"trigL={trigger_long} trigS={trigger_short} "
                f"don_ok_L={don_ok_L} don_ok_S={don_ok_S} ema_ok_L={ema_ok_L} ema_ok_S={ema_ok_S} "
                f"filters={filters_ok} reasons={','.join(reasons)}"
            )

        if trigger_long:
            reasons = []
            allow_adx = (adx_now >= min_adx)
            allow_bias_L = (bias > 0) or (bias == 0 and don_ok_L and adx_now >= (min_adx + 2.0))
            
            if not filters_ok: reasons.append("filters=False")
            if not allow_adx: reasons.append(f"adx<{min_adx}")
            if not allow_bias_L: reasons.append("bias_rule_fail")
            if not (don_ok_L or ema_ok_L): reasons.append("no_valid_long_trigger")

            if not reasons:
                _entry_log("LONG", "gate_pass")
                return 'long', origin
            else:
                _no_entry_log("LONG", reasons)

        elif trigger_short:
            reasons = []
            allow_adx = (adx_now >= min_adx)
            allow_bias_S = (bias < 0) or (bias == 0 and don_ok_S and adx_now >= (min_adx + 2.0))

            if not filters_ok: reasons.append("filters=False")
            if not allow_adx: reasons.append(f"adx<{min_adx}")
            if not allow_bias_S: reasons.append("bias_rule_fail")
            if not (don_ok_S or ema_ok_S): reasons.append("no_valid_short_trigger")

            if not reasons:
                _entry_log("SHORT", "gate_pass")
                return 'short', origin
            else:
                _no_entry_log("SHORT", reasons)

        return 'flat', origin

    def _compute_triggers(self, i_local, market, features, cfg_t):
        import math
        def _get(name, default=None):
            return market.get(name, default) if isinstance(market, dict) else getattr(market, name, default)

        close = _get("close"); high = _get("high"); low = _get("low"); open_ = _get("open")
        ema_fast = _get("ema_fast"); ema_slow = _get("ema_slow")
        atr = _get("atr")
        don_hi = _get("don_hi"); don_lo = _get("don_lo")
        adx = features.get('adx')

        for arr, nm in [(close,"close"), (high,"high"), (low,"low"), (open_,"open"), (ema_fast,"ema_fast"), (ema_slow,"ema_slow"), (atr,"atr"), (adx,"adx")]:
            if arr is None or i_local <= 1 or i_local >= len(arr):
                return dict(trigger_long=False, trigger_short=False, recent_brk_l=False, recent_brk_s=False, origin=None, don_long_ok=False, don_short_ok=False, ema_long_ok=False, ema_short_ok=False, atr_now=0.0, dist_to_break_long=0.0, dist_to_break_short=0.0)

        if don_hi is None or don_lo is None:
            return dict(trigger_long=False, trigger_short=False, recent_brk_l=False, recent_brk_s=False, origin=None, don_long_ok=False, don_short_ok=False, ema_long_ok=False, ema_short_ok=False, atr_now=0.0, dist_to_break_long=0.0, dist_to_break_short=0.0)
        else:
            don_hi_val = don_hi.iloc[i_local-1]
            don_lo_val = don_lo.iloc[i_local-1]

        atr_now = float(atr.iloc[i_local])
        eps = float(getattr(cfg_t, "break_eps_atr", 0.0024)) * atr_now

        close_now = float(close.iloc[i_local]); close_prev = float(close.iloc[i_local-1])
        open_now  = float(open_.iloc[i_local])
        high_now  = float(high.iloc[i_local]);  low_now   = float(low.iloc[i_local])
        ema_f_now = float(ema_fast.iloc[i_local]); ema_f_prev = float(ema_fast.iloc[i_local-1])
        ema_s_now = float(ema_slow.iloc[i_local]); ema_s_prev = float(ema_slow.iloc[i_local-1])

        body_min_k = float(getattr(cfg_t, "don_body_min_atr", 0.10))
        body_ok = True # abs(close_now - open_now) >= body_min_k * atr_now
        don_long_close  = (close_now >= don_hi_val + eps) and (close_prev <= don_hi_val + eps) and body_ok
        don_short_close = (close_now <= don_lo_val - eps) and (close_prev >= don_lo_val - eps) and body_ok

        adx_n_don   = int(getattr(cfg_t, "don_adx_slope_n", 3))
        need_rise_don = bool(getattr(cfg_t, "don_need_adx_rise", True))
        adx_now_don  = float(adx.iloc[i_local])
        adx_prev_don = float(adx.iloc[i_local - adx_n_don]) if i_local - adx_n_don >= 0 else adx_now_don
        adx_rise_don = (adx_now_don >= adx_prev_don)

        # Filtro de separación de EMA para Donchian
        sep_k_don = float(getattr(cfg_t, "don_ema_sep_atr", 0.0)) # 0.0 = apagado
        don_sep_val = abs(ema_f_now - ema_s_now) / max(atr_now, 1e-12)
        self._don_sep_samples.append(don_sep_val)
        sep_ok_don = don_sep_val >= sep_k_don

        don_long_ok  = don_long_close and (adx_rise_don if need_rise_don else True) and sep_ok_don
        don_short_ok = don_short_close and (adx_rise_don if need_rise_don else True) and sep_ok_don

        if don_long_close and not sep_ok_don:
            self.debug_counts["don_sep_rejects_L"] = self.debug_counts.get("don_sep_rejects_L", 0) + 1
        if don_short_close and not sep_ok_don:
            self.debug_counts["don_sep_rejects_S"] = self.debug_counts.get("don_sep_rejects_S", 0) + 1

        if don_long_close or don_short_close:
            try:
                self.logger.info(
                    f"[DON DEBUG] i={i_local} body_ok={body_ok} adx_rise={adx_rise_don} "
                    f"ema_sep_ok={sep_ok_don} "
                    f"don_ok_L={don_long_ok} don_ok_S={don_short_ok}"
                )
            except Exception:
                pass

        if don_long_close or don_short_close:
            try:
                self.logger.info(
                    f"[DON DEBUG] i={i_local} body_ok={body_ok} adx_rise={adx_rise_don} "
                    f"ema_sep_ok={sep_ok_don} "
                    f"don_ok_L={don_long_ok} don_ok_S={don_short_ok}"
                )
            except Exception:
                pass

        base_ema_long_ok  = (ema_f_prev <= ema_s_prev) and (ema_f_now > ema_s_now)
        base_ema_short_ok = (ema_f_prev >= ema_s_prev) and (ema_f_now < ema_s_now)

        sep_k   = float(getattr(cfg_t, "ema_min_sep_atr", 0.25))
        adx_n   = int(getattr(cfg_t, "adx_slope_n", 3))
        need_any= bool(getattr(cfg_t, "ema_need_any", True))

        if i_local > 0:
            atr_prev = float(atr.iloc[i_local - 1])
            ema_f_prev_sep = float(ema_fast.iloc[i_local - 1])
            close_prev_sep = close_prev
        else:
            atr_prev = atr_now
            ema_f_prev_sep = ema_f_now
            close_prev_sep = close_now

        sep_val = abs(close_prev_sep - ema_f_prev_sep) / max(atr_prev, 1e-12)
        self._sep_samples.append(sep_val)
        sep_ok = sep_val >= sep_k

        if base_ema_long_ok and not sep_ok:
            self.debug_counts["ema_sep_rejects_L"] = self.debug_counts.get("ema_sep_rejects_L", 0) + 1
        if base_ema_short_ok and not sep_ok:
            self.debug_counts["ema_sep_rejects_S"] = self.debug_counts.get("ema_sep_rejects_S", 0) + 1

        adx_now = float(adx.iloc[i_local])
        adx_prev = float(adx.iloc[i_local - adx_n]) if i_local - adx_n >= 0 else adx_now
        adx_rise = (adx_now >= adx_prev)

        extra_filter_ok = True if need_any else adx_rise
        ema_long_ok = base_ema_long_ok and sep_ok and extra_filter_ok
        ema_short_ok = base_ema_short_ok and sep_ok and extra_filter_ok

        self.logger.info(f"[V2_TRIG_DEBUG] i={i_local} don_L={don_long_ok} don_S={don_short_ok} ema_L={ema_long_ok} ema_S={ema_short_ok} adx_rise={adx_rise_don} sep_ok={sep_ok_don} body_ok={body_ok}")

        trigger_long  = bool(don_long_ok or ema_long_ok)
        trigger_short = bool(don_short_ok or ema_short_ok)

        origin = None
        if don_long_ok: origin = "don"
        elif ema_long_ok: origin = "ema"
        elif don_short_ok: origin = "don"
        elif ema_short_ok: origin = "ema"

        don_long_raw  = (high_now >= don_hi_val + eps) and (close_prev <= don_hi_val + eps)
        don_short_raw = (low_now  <= don_lo_val - eps) and (close_prev >= don_lo_val - eps)
        recent_brk_l = bool(don_long_raw)
        recent_brk_s = bool(don_short_raw)

        if getattr(self, "diag", False):
            if don_long_close:   self.trig_counts["don_L"] += 1
            if don_short_close:  self.trig_counts["don_S"] += 1
            if ema_long_ok:self.trig_counts["ema_L"] += 1
            if ema_short_ok:self.trig_counts["ema_S"] += 1

        return dict(
            trigger_long=trigger_long, trigger_short=trigger_short,
            recent_brk_l=recent_brk_l, recent_brk_s=recent_brk_s,
            atr_now=atr_now,
            dist_to_break_long=max(0.0, high_now - don_hi_val),
            dist_to_break_short=max(0.0, don_lo_val - low_now),
            origin=origin,
            don_long_ok=don_long_ok, don_short_ok=don_short_ok,
            ema_long_ok=ema_long_ok, ema_short_ok=ema_short_ok
        )

    def get_audit_rows(self):
        return self.audit_rows

    def signal(self, context):
        i_local = context.get('i', 0)
        market  = context.get('market', {})
        feats   = context.get('feats', {})
        self._df_ref = context.get('df')

        if getattr(self, "diag", False):
            try:
                ln_close = len(market.get('close')) if isinstance(market, dict) else len(getattr(market, 'close', []))
                self.logger.info(f"[SANITY] i_local={i_local} len_close={ln_close}")
            except Exception:
                pass

        cfg_t = SimpleNamespace(**self.cfg.get("trend_v2", {}))
        self._diagnose_reentry(i_local, market, feats, cfg_t)
        pending_reentry = self._pending_reentry
        ck = self._compute_triggers(i_local, market, feats, cfg_t)

        def _feature_scalar(name, default=0.0):
            series = feats.get(name)
            if isinstance(series, pd.Series) and i_local < len(series):
                return float(series.iloc[i_local])
            return float(feats.get(name, default))

        data = {
            'bias': _feature_scalar('bias', 0.0),
            'adx_now': _feature_scalar('adx', 0.0),
            'atr_htf': _feature_scalar('atr_htf', 0.0),
            'ema_fast_htf': _feature_scalar('ema_fast_htf', 0.0),
            'ema_slow_htf': _feature_scalar('ema_slow_htf', 0.0),
            'filters_ok': feats.get('filters_ok', True),
            **ck
        }

        if getattr(self, "diag", False) and (ck['trigger_long'] or ck['trigger_short']):
            try:
                self.logger.info(
                    f"[TRIG] i_local={i_local} L={ck['trigger_long']} S={ck['trigger_short']} "
                    f"adx={data['adx_now']:.2f} bias={data['bias']}"
                )
            except Exception:
                pass

        origin = ck.get('origin')
        is_reentry = False
        if pending_reentry:
            side = pending_reentry["side"]
            reason = "reentry_gate"
            origin = "reentry"
            is_reentry = True
            self._pending_reentry = None
        else:
            side, reason = self._check_signal(i_local, data)

        i_abs = context.get('i_abs', i_local)
        ts = context.get('ts', pd.Timestamp.now())
        self.audit_rows.append({"i": i_abs, "ts": ts, "adx": data['adx_now'], "atr": ck['atr_now']})

        if side in ('long', 'short'):
            sl_atr_cfg = self.cfg.get("exits", {}).get("sl_atr", 0.9)
            tp_final_r_cfg = self.cfg.get("exits", {}).get("tp_final_r", 1.2)
            tp_r_primary_cfg = self.cfg.get("exits", {}).get("tp_r_primary", 0.8)

            k_ATR = float(cfg_t.sl_atr_mult)
            pad_ATR = float(cfg_t.sl_swing_pad_atr)
            lookbk = int(cfg_t.sl_swing_lookback)
            mode = cfg_t.sl_mode
            cap = float(cfg_t.sl_swing_extra_atr_cap)

            atr_now_for_signal = ck.get('atr_now', 0.0)
            close_now = market.get('close').iloc[i_local]
            low_series = market.get('low')
            high_series = market.get('high')

            if side == 'long':
                swing_low = min(low_series.iloc[max(0, i_local - lookbk):i_local])
                atr_stop = close_now - k_ATR * atr_now_for_signal
                swing_stop = swing_low - pad_ATR * atr_now_for_signal
                if mode == "capped":
                    sl_min = atr_stop - cap * atr_now_for_signal
                    sl_price = max(swing_stop, sl_min)
                else:
                    sl_price = min(atr_stop, swing_stop)
            else:
                swing_high = max(high_series.iloc[max(0, i_local - lookbk):i_local])
                atr_stop = close_now + k_ATR * atr_now_for_signal
                swing_stop = swing_high + pad_ATR * atr_now_for_signal
                if mode == "capped":
                    sl_max = atr_stop + cap * atr_now_for_signal
                    sl_price = min(swing_stop, sl_max)
                else:
                    sl_price = max(atr_stop, swing_stop)

            self.logger.info(
                f"[SL DEBUG] i={i_local} side={side} entry={close_now:.2f} atr={atr_now_for_signal:.2f} "
                f"k_ATR={k_ATR:.2f} pad_ATR={pad_ATR:.2f} look={lookbk} mode={mode} cap={cap:.2f} "
                f"atr_stop={atr_stop:.2f} swing_stop={swing_stop:.2f} chosen={sl_price:.2f}"
            )

            sl_pts = abs(close_now - sl_price)
            tp_pts = tp_final_r_cfg * sl_pts
            partial_tp_pts = tp_r_primary_cfg * sl_pts
            rr = tp_final_r_cfg

            return SimpleNamespace(
                side=side, strength=1.0, sl_pts=sl_pts, tp_pts=tp_pts,
                partial_tp_pts=partial_tp_pts, reason=reason, rr=rr,
                partial_sl_offset_atr_mult=getattr(cfg_t, "partial_sl_offset_atr_mult", None),
                origin=origin,
                tag="reentry" if is_reentry else None,
                reentry=True if is_reentry else False
            )

    def reentry_enabled(self):
        return self.cfg.get("trend_v2", {}).get("reentry", True)
