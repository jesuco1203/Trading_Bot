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
            _ = run_single_backtest(cfg, df_win, symbol, timeframe, label, run_ts)
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
