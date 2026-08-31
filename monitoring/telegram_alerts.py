"""Alertas de cierre de velas 4h por Telegram.

El módulo no coloca órdenes. Solo obtiene velas públicas de OKX, evalúa una
envolvente confirmada y envía una notificación opcional al bot de Telegram.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/candles"
OKX_TICKER_URL = "https://www.okx.com/api/v5/market/ticker"
TELEGRAM_SEND_URL = "https://api.telegram.org/bot{token}/sendMessage"


@dataclass(frozen=True)
class Candle:
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open


@dataclass(frozen=True)
class Alert:
    symbol: str
    candle: Candle
    pattern: str
    direction: str
    level: float | None = None
    timeframe: str = "4h"
    mode: str = "close"


def parse_okx_candle(row: list[Any]) -> Candle:
    """Parse an OKX candle row; OKX returns newest candles first."""
    if len(row) < 5:
        raise ValueError(f"Fila OHLCV incompleta: {row!r}")
    return Candle(
        timestamp_ms=int(row[0]),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
    )


def confirmed_candles(payload: dict[str, Any]) -> list[Candle]:
    """Return confirmed candles sorted oldest to newest."""
    if payload.get("code") != "0":
        raise RuntimeError(f"OKX respondió con error: {payload}")
    rows = payload.get("data") or []
    parsed = []
    for row in rows:
        # OKX's ninth field is confirm: 1 means the candle is closed.
        if len(row) >= 9 and str(row[8]) != "1":
            continue
        parsed.append(parse_okx_candle(row))
    return sorted(parsed, key=lambda candle: candle.timestamp_ms)


def fetch_confirmed_candles(symbol: str, timeframe: str = "4h", limit: int = 5) -> list[Candle]:
    """Fetch only closed candles from OKX's public market endpoint."""
    query = urllib.parse.urlencode({"instId": symbol, "bar": timeframe.upper(), "limit": limit})
    request = urllib.request.Request(
        f"{OKX_CANDLES_URL}?{query}",
        headers={"User-Agent": "Trading_Bot/4h-alerts"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return confirmed_candles(payload)


def fetch_current_price(symbol: str) -> float:
    """Fetch the latest traded price from OKX for display in the UI."""
    query = urllib.parse.urlencode({"instId": symbol})
    request = urllib.request.Request(f"{OKX_TICKER_URL}?{query}", headers={"User-Agent": "Trading_Bot/alerts"})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("code") != "0" or not payload.get("data"):
        raise RuntimeError(f"OKX respondió con error de ticker: {payload}")
    return float(payload["data"][0]["last"])


def engulfing_direction(previous: Candle, current: Candle) -> str | None:
    """Match the project's classic body-engulfing definition."""
    if current.is_bullish and previous.is_bearish:
        if current.close >= previous.open and current.open <= previous.close:
            return "bullish"
    if current.is_bearish and previous.is_bullish:
        if current.close <= previous.open and current.open >= previous.close:
            return "bearish"
    return None


def crosses_level(previous: Candle, current: Candle, level: float, direction: str) -> bool:
    """Require a close crossing, avoiding repeated alerts above/below a level."""
    if direction == "above":
        return previous.close < level <= current.close
    if direction == "below":
        return previous.close > level >= current.close
    raise ValueError("direction debe ser 'above' o 'below'")


def crosses_live_level(previous_price: float, current_price: float, level: float) -> bool:
    """Return whether a live price crossed a level in either direction."""
    return (previous_price < level <= current_price) or (previous_price > level >= current_price)


def evaluate_candles(
    symbol: str,
    candles: list[Candle],
    *,
    level: float | None = None,
    level_direction: str | None = None,
    require_engulfing: bool = True,
    timeframe: str = "4h",
) -> Alert | None:
    """Evaluate the newest confirmed candle exactly once at the caller's tick."""
    if len(candles) < 2:
        return None
    previous, current = candles[-2:]
    pattern_direction = engulfing_direction(previous, current)
    if require_engulfing and pattern_direction is None:
        return None
    if level is not None:
        if level_direction not in ("above", "below"):
            raise ValueError("level_direction debe ser 'above' o 'below'")
        if not crosses_level(previous, current, level, level_direction):
            return None
    direction = pattern_direction or level_direction or "close"
    return Alert(symbol, current, pattern_direction or "level_close", direction, level, timeframe)


def format_alert(alert: Alert) -> str:
    """Format a compact Spanish notification for a phone screen."""
    when = datetime.fromtimestamp(alert.candle.timestamp_ms / 1000, tz=timezone.utc)
    pattern = {
        "bullish": "envolvente alcista",
        "bearish": "envolvente bajista",
        "level_close": "cierre de nivel",
        "live_level": "precio alcanzado",
    }.get(alert.pattern, alert.pattern)
    lines = [
        f"🚨 ALERTA {'EN VIVO' if alert.mode == 'live' else alert.timeframe.upper()} — {alert.symbol}",
        f"Tipo: {pattern}",
        f"{'Precio' if alert.mode == 'live' else 'Cierre'}: {alert.candle.close:g}",
        f"Vela cerrada: {when:%Y-%m-%d %H:%M UTC}",
    ]
    if alert.level is not None:
        lines.append(f"Nivel cruzado: {alert.level:g} ({alert.direction})")
    return "\n".join(lines)


def send_telegram(message: str, token: str | None = None, chat_id: str | None = None) -> None:
    """Send a Telegram message using environment variables by default."""
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("Faltan TELEGRAM_BOT_TOKEN y/o TELEGRAM_CHAT_ID")
    body = urllib.parse.urlencode({"chat_id": chat_id, "text": message}).encode("utf-8")
    request = urllib.request.Request(
        TELEGRAM_SEND_URL.format(token=token),
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not result.get("ok"):
        raise RuntimeError(f"Telegram respondió con error: {result}")
