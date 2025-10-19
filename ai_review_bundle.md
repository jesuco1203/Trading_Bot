# AI Review Bundle — Trading_Bot

## main.py
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main runner – Trading_Bot / TrendV2
- Orquesta el backtest por ventanas.
- Mantiene shared_context con df_base para execution/paper.py (evita KeyError: 'df_base').
- Imprime [CFG EFFECTIVE] y [SUMMARY run_ts=...] con window_id y métricas básicas.
- Compatible con broker.mark_to_market(close, ts, high=..., low=..., current_atr=..., i=..., shared_context=...).

Nota: No reimplementa estrategias ni broker; solo respeta su interfaz y agrega telemetría/guardas.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import logging
import os
import sys
import time
from typing import Any, Dict, Tuple
from strategies.trend_v2 import TrendV2
from regime.bias import regime_bias

# -------- Logging básico seguro
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

# -------- TOML loader (py311+: tomllib; fallback: toml)
try:
    import tomllib as _toml_lib   # Python 3.11+, lee bytes
    _USE_TOMLLIB = True
except Exception:
    import toml as _toml_lib      # Paquete externo, lee texto
    _USE_TOMLLIB = False

# -------- Pandas / Numpy
try:
    import pandas as pd
    import numpy as np
except Exception as e:
    logging.error("Falta pandas/numpy en el venv: %s", e)
    raise SystemExit(1)


# =========================
# Utilidades
# =========================
def load_config(path: str):
    import os
    if not isinstance(path, str):
        raise TypeError(f"config_path debe ser str, recibido {type(path)}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo de config: {path}")

    if _USE_TOMLLIB:
        # tomllib.load() espera archivo binario
        with open(path, "rb") as f:
            return _toml_lib.load(f)
    else:
        # toml.load() espera archivo de texto
        with open(path, "r", encoding="utf-8") as f:
            return _toml_lib.load(f)


def _first_existing(*paths: str) -> str | None:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def load_ohlcv(config: Dict[str, Any]) -> Tuple[pd.DataFrame, str, str]:
    """
    Intenta cargar OHLCV desde rutas típicas:
      - Parquet/csv bajo data/{SYMBOL}/{TF}/
      - Ruta explícita config['data']['path'] si existe.
    """
    symbol = config.get("symbol") or config.get("market", {}).get("symbol") or "BTC-USDT-SWAP"
    timeframe = config.get("timeframe") or config.get("market", {}).get("timeframe") or "30m"

    # Rutas candidatas
    explicit_path = (
        config.get("data", {}).get("path")
        or config.get("paths", {}).get("data_path")
        or ""
    )
    base1 = os.path.join("data", symbol, timeframe)
    base2 = os.path.join("data_alt", symbol, timeframe)

    # Archivos candidatos
    cand = [
        explicit_path,
        os.path.join(base1, "bars.parquet"),
        os.path.join(base1, "ohlcv.parquet"),
        os.path.join(base1, "ohlcv.csv"),
        os.path.join(base2, "bars.parquet"),
        os.path.join(base2, "ohlcv.parquet"),
        os.path.join(base2, "ohlcv.csv"),
    ]
    path = _first_existing(*[p for p in cand if p])

    if path is None:
        logging.error("No se encontró dataset OHLCV. Intenté: %s", cand)
        raise SystemExit(1)

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # Normaliza columnas y el índice de tiempo
    cols = {c.lower(): c for c in df.columns}
    for col in ["open", "high", "low", "close"]:
        if col not in map(str.lower, df.columns):
            logging.error("El dataset no tiene columna '%s'. Columnas: %s", col, list(df.columns))
            raise SystemExit(1)

    # Asegura nombres estándar (Open/High/Low/Close -> lower)
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in {"open", "high", "low", "close", "volume"} and c != lc:
            rename_map[c] = lc
    if rename_map:
        df = df.rename(columns=rename_map)

    # Índice temporal (si existe columna time/timestamp)
    time_col = None
    for cand_time in ["time", "timestamp", "ts", "date"]:
        if cand_time in df.columns:
            time_col = cand_time
            break
    if time_col:
        try:
            df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
            df = df.set_index(time_col)
        except Exception:
            pass  # si ya viene indexado

    df = df.sort_index()
    return df, symbol, timeframe


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR clásico (Wilder). Sin dependencias externas.
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder's smoothing
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr


# =========================
# Núcleo de backtest (una ventana)
# =========================
def run_single_backtest(
    config: Dict[str, Any],
    df_base: pd.DataFrame,
    symbol: str,
    timeframe: str,
    window_label: str,
    run_ts: int,
) -> Dict[str, Any]:
    """
    Ejecuta una ventana de backtest iterando y llamando al broker.
    Devuelve métricas agregadas por ventana junto con el nombre del CSV exportado.
    """
    try:
        paper_mod = importlib.import_module("execution.paper")
    except Exception as e:
        logging.error("No pude importar execution.paper: %s", e)
        raise SystemExit(1)

    broker_cls = getattr(paper_mod, "PaperBroker", None) or getattr(paper_mod, "Broker", None)
    if broker_cls is None:
        logging.error("No encontré clase Broker/PaperBroker en execution/paper.py")
        raise SystemExit(1)

    trend_logger = logging.getLogger("TrendV2")
    trend_strategy = TrendV2(config, trend_logger)
    all_strategies = {"TrendV2": trend_strategy}

    trend_cfg = config.get("trend_v2", {})
    exits_cfg = config.get("exits", {})
    risk_cfg = config.get("risk", {})
    costs_cfg = config.get("costs", {})

    def _cfg_get(section: Dict[str, Any], key: str, default: float) -> float:
        try:
            return float(section.get(key, default))
        except Exception:
            return default

    broker = broker_cls(
        initial_capital=_cfg_get(risk_cfg, "starting_equity", 10_000.0),
        taker_fee=_cfg_get(costs_cfg, "taker_fee", 0.0005),
        maker_fee=_cfg_get(costs_cfg, "maker_fee", 0.0002),
        slippage_bps=_cfg_get(costs_cfg, "slippage_bps", 0.0),
        all_strategies=all_strategies,
        sl_atr=_cfg_get(exits_cfg, "sl_atr", 1.0),
        tp_r_primary=_cfg_get(exits_cfg, "tp_r_primary", 0.0),
        tp_primary_ratio=_cfg_get(exits_cfg, "tp_primary_ratio", 0.0),
        tp_final_r=_cfg_get(exits_cfg, "tp_final_r", 0.0),
        be_trigger_atr=_cfg_get(exits_cfg, "be_trigger_atr", 0.0),
        trail_atr_mult=_cfg_get(exits_cfg, "trail_atr_mult", 0.0),
        time_stop_bars=int(exits_cfg.get("time_stop_bars", 0)),
        trail_activate_r=_cfg_get(trend_cfg, "trail_activate_r", 1.0),
        partial_take_r=_cfg_get(trend_cfg, "partial_take_r", 0.0),
        partial_take_frac=_cfg_get(trend_cfg, "partial_take_frac", 0.0),
        partial_be_eps_atr=_cfg_get(trend_cfg, "partial_be_eps_atr", trend_cfg.get("reentry_eps_atr", 0.0)),
    )

    logging.info("--- Running Backtest for %s %s (%s) ---", symbol, timeframe, window_label)

    df_base = df_base.copy()
    for col in ("open", "high", "low", "close"):
        if col in df_base.columns:
            df_base[col] = df_base[col].astype(float)

    features_cfg = config.get("features", {})
    atr_period = int(features_cfg.get("atr_period", 14))
    atr = compute_atr(df_base, period=atr_period)
    df_base["atr"] = atr

    ema_fast_period = int(trend_cfg.get("ema_fast", 9) or 9)
    ema_slow_period = int(trend_cfg.get("ema_slow", 21) or 21)
    adx_period = int(trend_cfg.get("min_adx", 15) or 15)
    donchian_period = int(trend_cfg.get("n_don", 20) or 20)

    df_base["ema_fast"] = df_base["close"].ewm(span=ema_fast_period, adjust=False).mean()
    df_base["ema_slow"] = df_base["close"].ewm(span=ema_slow_period, adjust=False).mean()

    from features.core import add_adx

    df_base["adx"], _, _ = add_adx(df_base, n=adx_period)
    df_base["don_hi"] = df_base["high"].rolling(donchian_period).max()
    df_base["don_lo"] = df_base["low"].rolling(donchian_period).min()
    df_base["bias"] = regime_bias(df_base["ema_fast"], df_base["ema_slow"])

    warmup = max(
        atr_period + 1,
        int(features_cfg.get("warmup_bars", 100) or 100),
        int(getattr(trend_strategy, "warmup_bars", lambda: 0)() or 0),
    )

    shared_context: Dict[str, Any] = {"df_base": df_base}

    idx = list(df_base.index)
    close_values = df_base["close"].to_numpy(dtype=float)

    entries = 0
    partials = 0
    flips = 0
    sl_le_2 = 0
    reentry_armed = 0
    reentry_exec = 0
    max_rr = 0.0

    risk_per_trade_pct = float(risk_cfg.get("risk_per_trade_pct", 0.0) or 0.0)
    starting_equity = float(risk_cfg.get("starting_equity", 0.0) or 0.0)

    for i in range(warmup, len(df_base)):
        current_ts = idx[i] if i < len(idx) else None
        atr_value = float(atr.iloc[i]) if i < len(atr) else float("nan")
        if not np.isfinite(atr_value):
            prev_index = max(i - 1, 0)
            atr_value = float(atr.iloc[prev_index])

        if broker.exposure() == 0:
            market_ctx = {
                "open": df_base["open"],
                "high": df_base["high"],
                "low": df_base["low"],
                "close": df_base["close"],
                "ema_fast": df_base["ema_fast"],
                "ema_slow": df_base["ema_slow"],
                "atr": df_base["atr"],
                "don_hi": df_base["don_hi"],
                "don_lo": df_base["don_lo"],
            }
            feats_ctx = {
                "adx": df_base["adx"],
                "bias": df_base["bias"],
                "atr": df_base["atr"],
            }
            context = {
                "i": i,
                "i_abs": i,
                "market": market_ctx,
                "feats": feats_ctx,
                "ts": current_ts,
                "df": df_base,
            }
            signal = trend_strategy.signal(context)
            side = getattr(signal, "side", "flat") if signal else "flat"
            if signal and side in ("long", "short"):
                sl_pts = float(getattr(signal, "sl_pts", 0.0) or 0.0)
                if sl_pts > 0:
                    origin = getattr(signal, "origin", None)
                    risk_scale = 1.0
                    if origin == "reentry":
                        risk_scale = float(trend_cfg.get("reentry_size_mult", 1.0) or 1.0)
                    risk_usd = risk_per_trade_pct * starting_equity * risk_scale
                    qty = risk_usd / sl_pts if sl_pts > 0 else 0.0
                    if qty > 0:
                        side_int = 1 if side == "long" else -1
                        try:
                            broker.enter_or_flip(
                                side=side_int,
                                qty=qty,
                                price=float(close_values[i]),
                                sl_pts=sl_pts,
                                tp_pts=float(getattr(signal, "tp_pts", 0.0) or 0.0),
                                partial_tp_pts=getattr(signal, "partial_tp_pts", None),
                                ts=current_ts,
                                strategy_name="TrendV2",
                                atr=atr_value,
                                partial_sl_offset_atr_mult=getattr(signal, "partial_sl_offset_atr_mult", None),
                                rr=getattr(signal, "rr", None),
                                symbol=symbol,
                                tf=timeframe,
                                origin=getattr(signal, "origin", None),
                                max_loss_usd=risk_usd,
                                tag=getattr(signal, "tag", None),
                            )
                            if getattr(signal, "reentry", False):
                                try:
                                    trend_strategy.register_reentry_execution(True)
                                    trend_strategy.logger.info(
                                        "[REENTRY EXEC SUCCESS] i=%s qty=%.4f price=%.2f",
                                        i,
                                        qty,
                                        float(close_values[i]),
                                    )
                                except Exception:
                                    pass
                            entries += 1
                        except Exception:
                            logging.exception("Fallo al abrir posición en i=%s", i)
                            if getattr(signal, "reentry", False):
                                try:
                                    trend_strategy.register_reentry_execution(False)
                                except Exception:
                                    pass

        try:
            broker.mark_to_market(
                float(close_values[i]),
                ts=current_ts,
                high=df_base["high"].iloc[: i + 1],
                low=df_base["low"].iloc[: i + 1],
                current_atr=atr_value,
                i=i,
                shared_context=shared_context,
            )
        except Exception as e:
            logging.exception("Error en broker.mark_to_market(i=%s): %s", i, e)
            raise SystemExit(1)

    if broker.exposure() != 0 and len(df_base) > 0:
        try:
            broker.close_open_position(
                float(close_values[-1]),
                idx[-1] if idx else None,
                reason="window_end",
                shared_context=shared_context,
                i=len(df_base) - 1,
            )
        except Exception:
            logging.exception("No se pudo cerrar la posición al final de la ventana")

    partials = int(getattr(broker, "partials_count", partials))
    flips = int(getattr(broker, "flips_count", flips))

    try:
        payload: Dict[str, int] = {}
        tc = getattr(trend_strategy, "trig_counts", None)
        if isinstance(tc, dict):
            payload.update({k: int(tc.get(k, 0) or 0) for k in ("don_L", "don_S", "ema_L", "ema_S")})
        dc = getattr(trend_strategy, "debug_counts", None)
        if isinstance(dc, dict):
            payload.update({k: int(dc.get(k, 0) or 0) for k in (
                "ema_sep_rejects_L",
                "ema_sep_rejects_S",
                "don_sep_rejects_L",
                "don_sep_rejects_S",
            )})
        sep_stats = {}
        don_sep_stats = {}
        reentry_counts = {}
        reentry_stats = {}
        reentry_late_stats = {}
        reentry_rr_stats = {}
        if hasattr(trend_strategy, "_percentiles"):
            try:
                stats_raw = trend_strategy._percentiles(getattr(trend_strategy, "_sep_samples", []))
                sep_stats = {k: round(float(v), 4) for k, v in stats_raw.items()}
            except Exception:
                sep_stats = {}
            try:
                don_stats_raw = trend_strategy._percentiles(getattr(trend_strategy, "_don_sep_samples", []))
                don_sep_stats = {k: round(float(v), 4) for k, v in don_stats_raw.items()}
            except Exception:
                don_sep_stats = {}
            try:
                re_stats_raw = trend_strategy._percentiles(getattr(trend_strategy, "re_dists", []))
                reentry_stats = {k: round(float(v), 4) for k, v in re_stats_raw.items()}
            except Exception:
                reentry_stats = {}
        try:
            dbg = getattr(trend_strategy, "re_dbg", {})
            if isinstance(dbg, dict):
                for key in (
                    "reentry_exec",
                    "relax_gate_used",
                    "reject_pending_dup",
                    "reject_exec_fail",
                    "relax_candidates",
                    "relax_rejected",
                ):
                    dbg.setdefault(key, 0)
                reentry_counts = {str(k): int(v) for k, v in dbg.items()}
        except Exception:
            reentry_counts = {}
        if payload:
            logging.info("TriggerCounts %s", payload)
        if sep_stats:
            logging.info("EMA_SEP_ATR_STATS %s", sep_stats)
        if don_sep_stats:
            logging.info("DON_SEP_ATR_STATS %s", don_sep_stats)
        if reentry_counts:
            logging.info("REENTRY_COUNTS %s", reentry_counts)
        if reentry_stats:
            logging.info("REENTRY_EPS_ATR_STATS %s", reentry_stats)
        reentry_late_stats = {}
        try:
            lateness = getattr(trend_strategy, "re_lateness_bars", [])
            if lateness:
                xs = sorted(float(x) for x in lateness)
                def _pct(arr, p):
                    if not arr:
                        return 0.0
                    k = (p / 100.0) * (len(arr) - 1)
                    f = int(k)
                    c = min(f + 1, len(arr) - 1)
                    w = k - f
                    return arr[f] * (1 - w) + arr[c] * w
                reentry_late_stats = {
                    "min": round(xs[0], 2),
                    "p50": round(_pct(xs, 50), 2),
                    "p75": round(_pct(xs, 75), 2),
                    "p90": round(_pct(xs, 90), 2),
                    "max": round(xs[-1], 2),
                }
        except Exception:
            reentry_late_stats = {}
        reentry_rr_stats = {}
        try:
            rr_samples = getattr(trend_strategy, "re_rr_snapshot", [])
            if rr_samples:
                ys = sorted(float(y) for y in rr_samples)
                def _pct_rr(arr, p):
                    if not arr:
                        return 0.0
                    k = (p / 100.0) * (len(arr) - 1)
                    f = int(k)
                    c = min(f + 1, len(arr) - 1)
                    w = k - f
                    return arr[f] * (1 - w) + arr[c] * w
                reentry_rr_stats = {
                    "p50": round(_pct_rr(ys, 50), 2),
                    "p75": round(_pct_rr(ys, 75), 2),
                    "p90": round(_pct_rr(ys, 90), 2),
                }
        except Exception:
            reentry_rr_stats = {}
        logging.info("REENTRY_LATENESS_STATS %s", reentry_late_stats or {})
        logging.info("REENTRY_RR_STATS %s", reentry_rr_stats or {})
    except Exception:
        logging.debug("No se pudo imprimir TriggerCounts", exc_info=True)

    trades = list(getattr(broker, "trades", []))
    closes = [t for t in trades if not t.get("partial")]
    if closes:
        max_rr = max(float(t.get("max_rr", 0.0) or 0.0) for t in closes)
        sl_le_2 = sum(
            1
            for t in closes
            if t.get("exit_reason") in {"sl", "be"} and int(t.get("bars_open", 0) or 0) <= 2
        )

    reentry_armed = int(getattr(trend_strategy, "reentry_armed_count", reentry_armed))
    reentry_exec = int(getattr(trend_strategy, "reentry_exec_count", reentry_exec))

    window_id = window_label.split()[-1].strip("()").replace(" ", "")
    csv_name = f"trades_{symbol}_{timeframe}_window_{window_id}_{run_ts}.csv"
    shared_context["csv_name"] = csv_name

    try:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(csv_name, index=False)
        logging.info("Trades exportado a %s (%s filas)", csv_name, len(trades_df))
    except Exception:
        logging.exception("No se pudo exportar CSV %s", csv_name)

    print(
        f"[SUMMARY] run_ts={run_ts} window={window_id} trades={entries} "
        f"SL_le_2={sl_le_2} partials={partials} reentry_armed={reentry_armed} "
        f"reentry_exec={reentry_exec} max_rr≈{round(float(max_rr or 0.0), 2)}"
    )

    return {
        "entries": entries,
        "exits": len(closes),
        "partials": partials,
        "flips": flips,
        "sl_le_2": sl_le_2,
        "reentry_armed": reentry_armed,
        "reentry_exec": reentry_exec,
        "max_rr": float(max_rr or 0.0),
        "csv_name": csv_name,
        "trend_strategy": trend_strategy, # Return the strategy object for debug info
    }


# =========================
# Orquestación multi-ventana
# =========================
def print_cfg_effective(cfg: Dict[str, Any]) -> None:
    tv2 = cfg.get("trend_v2", {})
    # Campos críticos que acordamos monitorear
    payload = {
        "min_adx": tv2.get("min_adx"),
        "break_eps_atr": tv2.get("break_eps_atr"),
        "don_body_min_atr": tv2.get("don_body_min_atr"),
        "don_need_adx_rise": tv2.get("don_need_adx_rise"),
        "don_adx_slope_n": tv2.get("don_adx_slope_n"),
        "don_ema_sep_atr": tv2.get("don_ema_sep_atr"),
        "ema_min_sep_atr": tv2.get("ema_min_sep_atr"),
        "adx_slope_n": tv2.get("adx_slope_n"),
        "sl_mode": tv2.get("sl_mode"),
        "sl_swing_extra_atr_cap": tv2.get("sl_swing_extra_atr_cap"),
        "partial_take_r": tv2.get("partial_take_r"),
        "partial_take_frac": tv2.get("partial_take_frac"),
        "trail_activate_r": tv2.get("trail_activate_r"),
        "trail_atr_mult": tv2.get("trail_atr_mult"),
        "reentry_window": tv2.get("reentry_window"),
        "reentry_sl_bars_open_max": tv2.get("reentry_sl_bars_open_max"),
        "reentry_sl_bars_open_relax": tv2.get("reentry_sl_bars_open_relax"),
        "reentry_eps_atr": tv2.get("reentry_eps_atr"),
        "reentry_rr_min": tv2.get("reentry_rr_min"),
        "reentry_adx_uplift": tv2.get("reentry_adx_uplift"),
        "reentry_exec_enabled": tv2.get("reentry_exec_enabled"),
        "relaxer_enabled": tv2.get("relaxer_enabled"),
    }
    print("[CFG EFFECTIVE]", payload)


def _parse_override_value(raw: str):
    sval = raw.strip()
    if sval.lower() in {"true", "false"}:
        return sval.lower() == "true"
    try:
        if "." in sval:
            return float(sval)
        return int(sval)
    except ValueError:
        return sval


def apply_overrides(cfg: Dict[str, Any], overrides: list[str]) -> None:
    if not overrides:
        return
    trend_cfg = cfg.setdefault("trend_v2", {})
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--override espera key=value, recibí: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--override inválido (clave vacía): {item!r}")
        trend_cfg[key] = _parse_override_value(value)


def main(config_path: str, limit_bars: int | None, start_index: int | None, overrides: list[str] | None = None) -> None:
    cfg = load_config(config_path)
    apply_overrides(cfg, overrides or [])
    df, symbol, timeframe = load_ohlcv(cfg)

    # Telemetría de config
    print_cfg_effective(cfg)

    # Si el usuario da start/limit → una sola ventana
    run_ts = int(time.time())
    windows = []  # lista de (start, end, label)

    if start_index is not None and limit_bars is not None:
        start = int(start_index)
        end = int(start_index + limit_bars)
        windows.append((start, end, f"Window custom ({start}-{end})"))
    else:
        # Tres ventanas fijas que veníamos usando
        windows = [
            (0, 2000, "Window 1 (0-2000)"),
            (2000, 4000, "Window 2 (2000-4000)"),
            (4000, 6000, "Window 3 (4000-6000)"),
        ]

    # Define _pct helper function for percentiles
    def _pct(arr, p):
        if not arr: return 0.0
        xs = sorted(arr)
        n = len(xs)
        k = (p / 100.0) * (n - 1)
        f = int(k)
        c = min(f + 1, n - 1)
        w = k - f
        return xs[f] * (1 - w) + xs[c] * w

    # Verifica rango
    n = len(df)
    for start, end, label in windows:
        if end > n:
            logging.warning("La ventana %s excede el dataset (len=%s). Ajustando fin a %s.", label, n, n)
            end = n
        if start >= end:
            logging.warning("Ventana vacía: %s (start=%s, end=%s). Saltando.", label, start, end)
            continue

        # Slice por ventana (copia para evitar SettingWithCopy warns)
        df_win = df.iloc[start:end].copy()
        try:
            result = run_single_backtest(cfg, df_win, symbol, timeframe, label, run_ts)
            trend_strategy = result.get("trend_strategy")
            import strategies.trend_v2 # Import the module to access its global DEBUG_REENTRY flag
            if trend_strategy and hasattr(trend_strategy, "_debug_reentry_data") and strategies.trend_v2.DEBUG_REENTRY:
                dbg_data = trend_strategy._debug_reentry_data
                
                reentry_lateness_debug = {
                    "late_events": dbg_data["lateness_events"],
                    "lateness_p50": round(_pct(dbg_data["lateness_values"], 50), 2),
                    "lateness_p75": round(_pct(dbg_data["lateness_values"], 75), 2),
                    "bars_open_p50": round(_pct(dbg_data["bars_open_at_check"], 50), 2),
                    "bars_open_p75": round(_pct(dbg_data["bars_open_at_check"], 75), 2),
                    "eps_p50": round(_pct(dbg_data["eps_values"], 50), 4),
                    "eps_p75": round(_pct(dbg_data["eps_values"], 75), 4),
                    "eps_thr": dbg_data["eps_threshold"],
                }
                print("REENTRY_LATENESS_DEBUG", reentry_lateness_debug)

        except SystemExit:
            # Propagamos errores críticos (importantes para no dejar estado corrupto)
            raise
        except Exception as e:
            logging.exception("Fallo en %s: %s", label, e)
            # continúa con la siguiente ventana; el error ya quedó en logs

    logging.info("Backtest(s) finalizado(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner – Trading_Bot / TrendV2")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Ruta al archivo TOML de configuración (p. ej., configs/btc_30m.toml)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Índice inicial de la ventana (opcional; si no se especifica, corre 3 ventanas fijas)",
    )
    parser.add_argument(
        "--limit-bars",
        type=int,
        default=None,
        help="Cantidad de barras para la ventana (opcional; si no se especifica, corre 3 ventanas fijas)",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Sobreescribe parámetros de trend_v2 con key=value (puede repetirse)",
    )
    args = parser.parse_args()

    try:
        main(
            config_path=args.config,
            limit_bars=args.limit_bars,
            start_index=args.start_index,
            overrides=args.override,
        )
    except KeyboardInterrupt:
        logging.warning("Interrumpido por el usuario.")
        sys.exit(130)

```

## execution/paper.py
```python
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
                 partial_be_eps_atr: float = 0.05):
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

```

## strategies/trend_v2.py
```python
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

# --- DEBUG FLAGS (no funcionales, solo logs) ---
DEBUG_REENTRY = True


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

        # Contadores y stats para debug de re-entry
        self._debug_reentry_data = {
            "lateness_events": 0,
            "lateness_values": [],      # barras_open - (max - relax)
            "eps_values": [],           # delta/ATR observado al disparar check
            "eps_threshold": None,      # umbral usado en la corrida
            "bars_open_at_check": [],
        }

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
        if DEBUG_REENTRY:
            self._debug_reentry_data["lateness_events"] += 1
            self._debug_reentry_data["lateness_values"].append(lateness)

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

        if DEBUG_REENTRY:
            self._debug_reentry_data["eps_values"].append(eps_atr)
            self._debug_reentry_data["bars_open_at_check"].append(bars_since)
            self._debug_reentry_data["eps_threshold"] = float(getattr(cfg_t, "reentry_eps_atr", 0.0))

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
        body_ok = abs(close_now - open_now) >= body_min_k * atr_now
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

```

## strategies/trend.py
```python
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
```

## strategies/base.py
```python
from __future__ import annotations
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Signal:
    side: str
    strength: float
    sl_pts: Optional[float]
    tp_pts: Optional[float]
    partial_tp_pts: Optional[float] = None
    partial_sl_offset_atr_mult: Optional[float] = None
    reason: Optional[str] = None
    rr: Optional[float] = None # New
    risk_scale: float = 1.0
    origin: str = "normal"

class BaseStrategy:
    def __init__(self, name: str, risk_mult: float = 1.0, time_stop_bars: int = 0, time_stop_mfe_atr: float = 0.0):
        self.name = name
        self.risk_mult = risk_mult
        self.time_stop_bars = time_stop_bars
        self.time_stop_mfe_atr = time_stop_mfe_atr

    def signal(self, ctx: Dict[str, Any]) -> Signal | None:
        raise NotImplementedError

    def warmup_bars(self) -> int:
        return 0

    def on_stop(self):
        pass

    def print_summary(self, trades: list):
        strat_trades = [t for t in trades if t.get("strategy") == self.name]
        if not strat_trades:
            return 0.0, 0.0, 0, 0, 0.0, 0, 0.0, 0.0

        # agrupar por operación (entrada)
        ops = {}
        for t in strat_trades:
            key = t.get("trade_id") or t.get("entry_ts") or t.get("ts")
            ops.setdefault(key, []).append(t)

        entries = 0
        wins = 0
        total_pnl = 0.0
        rrs = []
        bars_open_list = []
        mfe_atr_list = []
        mae_atr_list = []

        for key, evs in ops.items():
            evs.sort(key=lambda x: x.get("ts", 0))
            entries += 1
            entry_ev = next((e for e in evs if e.get("event")=="entry"), None)
            if entry_ev and entry_ev.get("rr") is not None:
                rrs.append(float(entry_ev["rr"]))

            # pnl total de la operación
            pnl_op = sum(float(e.get("pnl", 0.0)) for e in evs)
            total_pnl += pnl_op

            # win: cierre final por TP (estricto) o pnl_op>0 (simple). Elige 1 y deja comentada la otra.
            close_ev = next((e for e in reversed(evs) if e.get("event") == "close"), None)
            is_win = (close_ev and close_ev.get("exit_reason") == "tp")
            is_win = is_win or (pnl_op > 0)   # <-- opción alternativa
            if is_win:
                wins += 1

            if close_ev:
                bars_open_list.append(close_ev.get("bars_open", 0))
                mfe_atr_list.append(close_ev.get("mfe_atr", 0.0))
                mae_atr_list.append(close_ev.get("mae_atr", 0.0))

        rr_avg = (sum(rrs) / len(rrs)) if rrs else 0.0
        hit_rate = (wins / entries) * 100 if entries else 0.0
        avg_bars_open = (sum(bars_open_list) / len(bars_open_list)) if bars_open_list else 0
        mfe_atr_avg = (sum(mfe_atr_list) / len(mfe_atr_list)) if mfe_atr_list else 0.0
        mae_atr_avg = (sum(mae_atr_list) / len(mae_atr_list)) if mae_atr_list else 0.0

        return total_pnl, rr_avg, entries, wins, hit_rate, avg_bars_open, mfe_atr_avg, mae_atr_avg

```

## exits.py
```python
import logging
import math

def manage_exit(state, side, entry_px, sl_px, atr_now, bar_count_since_entry, high, low,
                sl_atr, tp_r_primary, tp_primary_ratio, tp_final_r,
                be_trigger_atr, trail_atr_mult, time_stop_bars, trail_activate_r, bar_rr):

    logger = logging.getLogger()
    old_sl = sl_px
    new_sl = old_sl

    # --- Guards ---
    unrealized_R = bar_rr
    partial_taken = state.partial_done
    cfg_trail_activate_r = state.trail_activate_r

    allow_BE = partial_taken or (unrealized_R >= cfg_trail_activate_r)
    allow_trail = (unrealized_R >= cfg_trail_activate_r)

    # --- Logic ---
    if allow_trail:
        # Trailing stop logic (Chandelier Exit)
        try:
            chand_high = high.rolling(20).max().iloc[-1]
            chand_low = low.rolling(20).min().iloc[-1]
            if side == 1 and not math.isnan(chand_high):
                trail_price = float(chand_high) - trail_atr_mult * atr_now
                new_sl = max(old_sl, trail_price)
            elif side == -1 and not math.isnan(chand_low):
                trail_price = float(chand_low) + trail_atr_mult * atr_now
                new_sl = min(old_sl, trail_price)
        except IndexError:
            pass # Not enough data for rolling window

    elif allow_BE and not state.be_set:
        # Break-even logic (only if not already trailing)
        eps = state.partial_be_eps_atr * state.entry_atr if state.entry_atr else 0.0
        be_price = entry_px + side * eps
        if side == 1:
            new_sl = max(old_sl, be_price)
        else:
            new_sl = min(old_sl, be_price)
        state.be_set = True

    # --- Logging ---
    if new_sl != old_sl:
        logger.info(
            f"[SL MOVE] i={bar_count_since_entry} from={old_sl:.2f} to={new_sl:.2f} "
            f"R={unrealized_R:.2f} partial_taken={partial_taken} "
            f"trail_on={allow_trail}"
        )

    # --- Time Stop ---
    if time_stop_bars and bar_count_since_entry >= time_stop_bars and unrealized_R < 1.0:
        state.force_exit = True

    tp_px = state.tp
    return new_sl, tp_px, state
```

## utils/perf.py
```python
def track_mfe_mae(ohlcv, atr_series, side, entry_idx, exit_idx, entry_px, atr_at_entry):
    """
    ohlcv: DataFrame con columnas high, low (y opcional close)
    atr_series: Serie ATR alineada
    side: +1 long, -1 short
    entry_idx, exit_idx: índices absolutos
    entry_px: precio de entrada
    atr_at_entry: ATR en la barra de entrada
    """
    window = slice(entry_idx, exit_idx + 1)
    highs = ohlcv["high"].iloc[window]
    lows  = ohlcv["low"].iloc[window]

    if side == 1:  # long
        mfe_abs = highs.max() - entry_px
        mae_abs = entry_px - lows.min()
    else:          # short
        mfe_abs = entry_px - lows.min()
        mae_abs = highs.max() - entry_px

    if not atr_at_entry or atr_at_entry == 0:
        mfe_atr = 0.0
        mae_atr = 0.0
    else:
        mfe_atr = float(mfe_abs / atr_at_entry)
        mae_atr = float(mae_abs / atr_at_entry)

    return float(mfe_abs), float(mae_abs), mfe_atr, mae_atr

```

## utils/common.py
```python
import logging
import sys
import time
import requests
from functools import wraps
import numpy as np

def setup_logger():
    """Configura el logger para que escriba en un archivo y en la consola."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("trading_bot.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def retry_with_backoff(max_retries=3, initial_delay=5, backoff_factor=2):
    """
    Decorador que reintenta una función con un tiempo de espera exponencial
    si lanza una excepción.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        logging.error(f"Error final en {func.__name__} tras {max_retries} intentos: {e}")
                        raise
                    logging.warning(f"Error en {func.__name__}: {e}. Reintentando en {delay}s...")
                    time.sleep(delay)
                    delay *= backoff_factor
            return None
        return wrapper
    return decorator

def validate_config(config):
    """
    Valida que el archivo de configuración contenga todas las claves necesarias.
    Lanza ValueError si falta alguna clave o los valores son incorrectos.
    """
    required_keys = {
        'okx_credentials': ['apiKey', 'secret', 'password'],
        'telegram': ['bot_token', 'chat_id'],
        'trading_settings': ['symbol', 'timeframe', 'position_risk_percentage', 'paper_trading'],
        'strategy_params': [
            'sma_period', 'rsi_period', 'atr_period', 'volume_period',
            'support_resistance_period', 'rsi_oversold', 'rsi_overbought',
            'volume_factor', 'atr_multiplier_sl', 'risk_reward_ratio'
        ]
    }
    
    for section, keys in required_keys.items():
        if section not in config:
            raise ValueError(f"Falta la sección '{section}' en la configuración")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Falta la clave '{key}' en la sección '{section}'")

    if not (0 < config['trading_settings']['position_risk_percentage'] <= 100):
        raise ValueError("'position_risk_percentage' debe estar entre 0 y 100")
    logging.info("Configuración validada exitosamente.")

def send_telegram_notification(message, config):
    """Envía una notificación a través de Telegram."""
    bot_token = config['telegram']['bot_token']
    chat_id = config['telegram']['chat_id']

    if not bot_token or not chat_id or bot_token == "TU_TELEGRAM_BOT_TOKEN_AQUI":
        logging.warning("Credenciales de Telegram no configuradas. Omitiendo notificación.")
        return

    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"Error al enviar notificación de Telegram: {e}")


```

## risk/risk_manager.py
```python
from dataclasses import dataclass

@dataclass
class RiskManager:
    max_dd_pct_session: float = 5.0
    max_consecutive_losses: int = 3
    max_daily_trades: int = 20
    
    _hard_stopped: bool = False
    _consecutive_losses: int = 0

    def check_limits(self, equity: float, trades: list, current_dt: any):
        """
        Checks session-level risk limits.
        If a limit is breached, it sets the internal hard-stop flag.
        (Logic to be fully implemented later)
        """
        # Placeholder for max drawdown check
        # if new_drawdown > self.max_dd_pct_session:
        #     self._hard_stopped = True
        #     logging.warning(f"RISK_MANAGER: Hard stop triggered due to max session drawdown.")

        # Placeholder for consecutive losses check
        # if new_consecutive_losses > self.max_consecutive_losses:
        #     self._hard_stopped = True
        #     logging.warning(f"RISK_MANAGER: Hard stop triggered due to max consecutive losses.")
        
        pass

    def is_hard_stopped(self) -> bool:
        """Returns True if a hard stop has been triggered."""
        return self._hard_stopped


```

## risk/position_sizing.py
```python
# risk/position_sizing.py

from dataclasses import dataclass

@dataclass
class InstrumentSpec:
    symbol: str
    linear: bool = True          # True: contrato lineal (USDT margined); False: inverso (coin margined)
    contract_size: float = 1.0   # p.ej., 1 para swaps lineales; 0.001 para algunos inversos
    lot_step: float = 0.0001     # paso mínimo de qty
    min_qty: float = 0.001

@dataclass
class RiskConfig:
    risk_usd_per_trade: float = 25.0   # riesgo fijo por trade
    max_notional_usd: float = 1_000.0  # tope de nocional
    max_leverage: float = 1.0          # no aplicar dos veces
    slippage_usd: float = 0.0

def floor_to_step(x: float, step: float) -> float:
    if step <= 0: 
        return x
    return (int(x / step)) * step

def clamp_qty(qty: float, min_qty: float, step: float) -> float:
    q = floor_to_step(max(qty, min_qty), step)
    return q

def compute_qty_for_stop(
    entry_price: float,
    stop_price: float,
    inst: InstrumentSpec,
    risk: RiskConfig,
    side: int,  # +1 long, -1 short
) -> float:
    """
    qty = risk_usd / (USD loss per unit at SL)
    Maneja contratos lineales e inversos y aplica topes de nocional y leverage *una sola vez*.
    """

    # 1) Distancia al stop en USD por unidad de qty
    dist_px = abs(entry_price - stop_price)
    if dist_px <= 0:
        return 0.0

    if inst.linear:
        # Pérdida por 1 unidad de qty ≈ dist_px USD (si contract_size=1 equivale a 1 nocional)
        loss_per_qty_usd = dist_px * inst.contract_size
        notional_per_qty_usd = entry_price * inst.contract_size
    else:
        # Inverso: PnL ≈ (1/entry - 1/exit) * contract_size * USD_per_coin
        # Aproximación local: d(1/p) ≈ dist_px / (entry^2)
        loss_per_qty_usd = (dist_px / max(entry_price**2, 1e-12)) * inst.contract_size * entry_price
        notional_per_qty_usd = (inst.contract_size * entry_price)  # prox.

    if loss_per_qty_usd <= 0:
        return 0.0

    # 2) Qty inicial por riesgo fijo
    qty = (risk.risk_usd_per_trade + risk.slippage_usd) / loss_per_qty_usd

    # 3) Tope por nocional y leverage (no duplicar leverage en otra capa)
    #    nocional = qty * notional_per_qty_usd  => limitar por max_notional_usd * max_leverage
    max_notional = risk.max_notional_usd * max(risk.max_leverage, 1.0)
    if max_notional > 0:
        qty_cap = max_notional / max(notional_per_qty_usd, 1e-12)
        qty = min(qty, qty_cap)

    # 4) Clamp a paso y mínimo
    qty = clamp_qty(qty, inst.min_qty, inst.lot_step)

    return qty

```

## features/core.py
```python
import pandas as pd
import numpy as np

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, low_price, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr1 = h - low_price
    tr2 = (h - prev_c).abs()
    tr3 = (low_price - prev_c).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    diff = s.diff()
    gain = diff.where(diff > 0, 0)
    loss = -diff.where(diff < 0, 0)
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return (100 - (100 / (1 + rs))).fillna(50)

def _zscore(s: pd.Series, n: int = 50) -> pd.Series:
    mean = s.rolling(n).mean()
    std = s.rolling(n).std().replace(0, 1e-9)
    return ((s - mean) / std).bfill()

def add_adx(df, n=14):
    h = df["high"].astype(float)
    low_price = df["low"].astype(float)
    c = df["close"].astype(float)
    up = h.diff()
    dn = -low_price.diff()
    plus_dm  = np.where((up > dn) & (up > 0), up, 0.0).astype(float)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0).astype(float)

    tr1 = (h - low_price).abs()
    tr2 = (h - c.shift()).abs()
    tr3 = (low_price - c.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).astype(float)

    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    pdm = pd.Series(plus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean()
    mdm = pd.Series(minus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean()

    plus_di  = (100.0 * (pdm / atr.replace(0, np.nan))).fillna(0)
    minus_di = (100.0 * (mdm / atr.replace(0, np.nan))).fillna(0)
    dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) ) * 100.0
    adx = dx.ewm(alpha=1/n, adjust=False).mean().fillna(0.0)
    return adx.clip(0, 100), plus_di, minus_di

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["close"]

    out["ret"] = c.pct_change().fillna(0.0)
    out["vol"] = out["ret"].rolling(30).std().bfill()
    out["rng_pct"] = ((out["high"] - out["low"]) / c).rolling(14).mean().bfill()

    out['score'] = out['ret'].rolling(14).mean() * 100

    atr_abs = _atr(out, 14).bfill()
    out["atr"] = atr_abs
    out["atr_pct"] = (atr_abs / c).bfill()

    out["rsi14"] = _rsi(c, 14)
    out["z_score_50"] = _zscore(c, 50)

    out["ema_10"] = c.ewm(span=10, adjust=False).mean().bfill()
    out["sma20"] = c.rolling(20).mean().bfill()
    out["ema_50"] = c.ewm(span=50, adjust=False).mean().bfill()
    out["ema_200"] = c.ewm(span=200, adjust=False).mean().bfill()
    out["std20"] = c.rolling(20).std().bfill()

    from .ict import add_fvg_columns
    from .candles import add_basic_candles
    out = add_fvg_columns(out)
    out = add_basic_candles(out)

    out["adx"], out["di_plus"], out["di_minus"] = add_adx(out)

    out["ret"] = out["ret"].clip(-0.1, 0.1)
    out["vol"] = out["vol"].clip(0, out["vol"].quantile(0.99))
    out["rng_pct"] = out["rng_pct"].clip(0, out["rng_pct"].quantile(0.99))
    return out
```

## configs/btc_30m.toml
```toml
[market]
symbols = ["BTC-USDT-SWAP"]
timeframes = ["30m"]
tick_value = 0.01 # Assuming BTC-USDT has 0.01 tick size

[data]
root = "data_alt"

[risk]
starting_equity = 10000.0
risk_per_trade_pct = 0.01
max_dd_pct_session = 0.06
max_consecutive_losses = 4
max_dd_pct_trend = 0.05
max_dd_pct_mr = 0.02
max_consecutive_losses_trend = 2
max_consecutive_losses_mr = 2

[costs]
taker_fee = 0.0005
maker_fee = 0.0002
slippage_bps = 1.0 # 1 basis point = 0.01%

[hmm]
n_states = 3

[selector]
enter_th = 0.45
exit_th = 0.35
persistence = 2 # For 30m

[strategy_mapping]
trend = ["TrendV2"]   # deja Trend original disponible pero desactivado
mr    = ["MeanRevert"]
high_vol = []

[strategy_params]
Trend_risk_mult = 0.6
MR_risk_mult    = 0.25

# Trend Strategy Params (example values, adjust as needed)
Trend_arm_rbody_th = 0.6
Trend_arm_adx_th = 25.0
Trend_arm_adx_delta_th = 5.0
Trend_arm_timeout = 5
Trend_retest_eps_pct = 0.001
Trend_reconfirm_mult_1 = 0.0003
Trend_reconfirm_mult_2 = 0.0008
Trend_sl_mult_atr = 1.5
Trend_tp_mult_sl = 2.0
Trend_tp_mult_atr = 3.0
Trend_partial_tp_atr_mult = 1.0
Trend_partial_sl_offset_atr_mult = 0.3
Trend_trail_trigger_atr_mult   = 1.5
Trend_trail_sl_offset_atr_mult = 0.3
Trend_sl_cooldown_duration = 5
Trend_allow_shorts = true
Trend_min_pmax = 0.6
Trend_max_dist_ma20_atr = 2.0
Trend_time_stop_bars = 6
Trend_time_stop_mfe_atr = 0.5
Trend_ma_fast_len = 20
Trend_ma_slow_len = 50
Trend_sep_min = 0.0010

# Trend caps (como MR)
Trend_risk_usd_per_trade = 10
Trend_max_loss_usd       = 25
Trend_max_notional_usd   = 10000
Trend_leverage_max       = 1
Trend_min_qty            = 0.001

# MeanRevert caps
MR_risk_usd_per_trade = 20
MR_max_loss_usd       = 25
MR_max_notional_usd   = 10000
MR_leverage_max       = 1
MR_min_qty            = 0.001

# MeanRevert Strategy Params (example values, adjust as needed)
MeanRevert_risk_mult = 0.5
MeanRevert_gate_pmax_th = 0.6
MeanRevert_gate_adx_th = 20.0
MeanRevert_gate_atr_pct_th = 0.005
MeanRevert_dist_sma_mult = 1.0
MeanRevert_signal_z_th = 1.5
MeanRevert_signal_rsi_th = 30.0
MeanRevert_signal_rbody_th = 0.5
MeanRevert_signal_lower_wick_th = 0.5
MeanRevert_sl_mult_atr = 1.0
MeanRevert_sl_swing_low_bars = 5
MeanRevert_tp_mult_sl = 1.5
MeanRevert_tp_mult_atr = 1.0
MeanRevert_partial_tp_atr_mult = 0.5
MeanRevert_partial_sl_offset_atr_mult = 0.2
MeanRevert_local_cooldown_duration = 14
MeanRevert_gate_adx_max = 22
MeanRevert_time_stop_bars = 6
MeanRevert_time_stop_mfe_atr = 0.5
MeanRevert_min_dist_sma_atr = 1.0
MeanRevert_rr_min = 1.4
MeanRevert_partial_atr = 0.8
MeanRevert_partial_sl_offset_atr = 0.1
smoke_bypass_trendv2 = true

[trend_v2]
# Desencadenantes
n_don = 20
require_trend = false
require_adx_rising = false
adx_rise_lookback = 3
adx_rise_min_delta = 1.0
recent_breakout_k = 3
ema_fast = 9
ema_slow = 21
min_adx             = 15
break_eps_atr = 0.0024
vol_ratio = 0.95
slope_lookback = 1
volatility_min_ratio = 0.2
pullback_atr = 0.2
volatility_lookback = 50
vola_min_ratio = 0.5
vol_lookback = 20
vol_ratio_min = 0.80
vol_ratio_pct = 0.10
min_signals_per_window = 8
relax_order = ["volume", "adx_rising"]
relax_progress_pct = 0.50
diag_triggers = true
allow_when_bias_neutral = true
htf_neutral_k_atr = 0.6

sl_atr_mult = 1.7
sl_swing_lookback = 5
trail_atr_mult = 1.7
trail_activate_r = 1.0
partial_take_r = 1.0
partial_take_frac = 0.33
reentry_size_mult = 0.5
atr_pct_high = 0.015
atr_pct_low  = 0.006

adx_rising_bars = 3
structure_confirm = true

reentry_enabled = true
reentry_on_last_sl = true
reentry_max_per_signal = 1
reentry_window = 16
reentry_sl_bars_open_max = 16
reentry_sl_bars_open_relax = 4
reentry_eps_atr = 0.0010
reentry_rr_min = 0.7
reentry_adx_uplift = true
reentry_exec_enabled = false
sl_swing_pad_atr = 0.2
don_body_min_atr = 0.18
sl_mode = "capped"
sl_swing_extra_atr_cap = 0.70
ema_min_sep_atr = 0.18
adx_slope_n     = 3
ema_need_any    = true
don_adx_slope_n     = 2
don_need_adx_rise = true
don_ema_sep_atr = 0.15
relaxer_enabled = false

[exits]
# Stop ni muy estrecho ni inmenso:
sl_atr          = 0.9
# TP moderado para romper el 0% de aciertos:
tp_final_r      = 1.2
# Break-even temprano para proteger avance:
be_trigger_atr  = 0.5
# Trailing lógico (al SL, no al TP):
trail_atr_mult  = 1.7
time_stop_bars  = 60
# (opcional parcial, si ya lo tenías)
tp_r_primary     = 0.8
tp_primary_ratio = 0.5


[cooldown]
after_loss_streak = 6   # velas sin operar tras racha >=2
after_kill_switch = 24  # velas sin operar tras kill-switch

[telemetry]
audit_csv = true
trades_csv = true
stats_console = true
# contadores para diagnosticar “cuello de botella”
count_trigger_raw = true
count_passed_filters = true
log_regime_bias = true
log_exit_mgmt = true

[export]
# Plantillas de nombre unificadas
audit_pattern  = "audit_${symbol}_${timeframe}_window_${win}_${i0}-${i1}.csv"
trades_pattern = "trades_${symbol}_${timeframe}_window_${win}_${i0}-${i1}.csv"
summary_dir = "reports"

[validation_targets]
# Metas de esta tanda (para que tu runner avise en consola)
min_trades_per_window = 8
min_hit_rate_pct = 20
min_median_mfe_atr = 1.0
max_median_mae_atr = 1.2
min_pass_ratio_pct = 20   # passed_filters >= 20% de trigger_raw

[selectors]
use_high_vol = false

```

## pyproject.toml
```
[tool.ruff]
exclude = ["venv", ".venv", "**/site-packages/**", "**/dist-packages/**"]
line-length = 100
target-version = "py39"
```

## requirements.txt
```
ccxt
pandas
pandas-ta
numpy
toml
hmmlearn
scikit-learn
```

## data/okx_client.py
```python
from __future__ import annotations
import os
import pandas as pd

def _path(root: str, symbol: str, timeframe: str) -> str:
    p = symbol.replace("/", "_").replace(":", "_")
    return os.path.join(root, p, timeframe, "ohlcv.parquet")

def get_ohlcv(symbol: str, timeframe: str, limit: int = 3000,
              root: str = "data/okx") -> pd.DataFrame:
    fp = _path(root, symbol, timeframe)
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Parquet not found: {fp}")
    df = pd.read_parquet(fp)
    # normaliza columnas esperadas por build_features/main
    df = df.rename(columns={"ts":"ts","open":"open","high":"high","low":"low","close":"close","volume":"volume"})
    df = df.sort_values("ts").reset_index(drop=True)

    return df
```

## okx_client.py
```python
import ccxt
import pandas as pd
import logging

class OKXClient:
    """
    Cliente para interactuar con la API de OKX.
    Maneja la autenticación, obtención de datos y ejecución de órdenes.
    """
    def __init__(self, config: dict):
        self.config = config
        self.exchange = self._init_exchange()

    def _init_exchange(self):
        """Inicializa la instancia de ccxt con las credenciales."""
        try:
            exchange = ccxt.okx({
                'apiKey': self.config['okx_credentials']['apiKey'],
                'secret': self.config['okx_credentials']['secret'],
                'password': self.config['okx_credentials']['password'],
                'options': {
                    'defaultType': 'swap', # Usar SWAP para derivados (futuros/perpetuos)
                },
            })
            # Habilitar modo de prueba (sandbox) si está en la configuración
            if self.config['trading_settings']['paper_trading']:
                logging.info("Modo Paper Trading (Sandbox) activado.")
                exchange.set_sandbox_mode(True)
            return exchange
        except Exception as e:
            logging.error(f"Error al inicializar el cliente de OKX: {e}")
            raise

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        Obtiene datos OHLCV y los convierte a un DataFrame de pandas.
        """
        try:
            logging.info(f"Obteniendo datos OHLCV para {symbol} en timeframe {timeframe}...")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logging.error(f"Error al obtener datos OHLCV: {e}")
            return pd.DataFrame() # Devuelve un DataFrame vacío en caso de error

    def get_balance(self, currency: str = 'USDT') -> float:
        """
        Obtiene el saldo disponible para una moneda específica.
        """
        try:
            balance = self.exchange.fetch_balance()
            # La estructura puede variar, busca en 'free', 'total', o la específica del activo.
            # Para derivados, el balance relevante suele ser el de margen en USDT.
            if currency in balance['total']:
                return balance[currency]['free'] or 0.0
            return 0.0
        except Exception as e:
            logging.error(f"Error al obtener el saldo: {e}")
            return 0.0

    def place_order(self, symbol: str, side: str, amount: float, price: float, params: dict):
        """
        Coloca una orden de mercado con Stop Loss y Take Profit.
        """
        order_side = 'buy' if side == 'BUY' else 'sell'
        try:
            logging.info(f"Intentando colocar orden {side} para {symbol}...")
            logging.info(f"Cantidad: {amount}, Precio Entrada Aprox: {price}, SL: {params['stop_loss']}, TP: {params['take_profit']}")

            if self.config['trading_settings']['paper_trading']:
                logging.warning("MODO SIMULACIÓN: La orden no se ejecutará en el mercado real.")
                # Simula una respuesta exitosa
                return {
                    'info': {'ordId': 'simulated_12345'},
                    'status': 'open'
                }

            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=order_side,
                amount=amount,
                params={
                    'tdMode': 'isolated', # o 'cross'
                    'slPrice': self.exchange.price_to_precision(symbol, params['stop_loss']),
                    'tpPrice': self.exchange.price_to_precision(symbol, params['take_profit'])
                }
            )
            logging.info(f"Orden colocada exitosamente: {order['info']['ordId']}")
            return order
        except Exception as e:
            logging.error(f"Error al colocar la orden: {e}")
            return None
```

