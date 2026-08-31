"""Monitor de alertas 4h: una consulta exactamente después de cada cierre."""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta, timezone

from monitoring.telegram_alerts import (
    evaluate_candles,
    fetch_confirmed_candles,
    format_alert,
    send_telegram,
)


def next_candle_close(timeframe: str = "4h", now: datetime | None = None) -> datetime:
    """Return the next UTC boundary for the supported timeframe."""
    now = now or datetime.now(timezone.utc)
    hours = {"3h": 3, "4h": 4}.get(timeframe)
    if hours is None:
        raise ValueError("timeframe debe ser '3h' o '4h'")
    next_hour = ((now.hour // hours) + 1) * hours
    day = now.date()
    if next_hour >= 24:
        return datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    return datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).replace(hour=next_hour)


def next_4h_close(now: datetime | None = None) -> datetime:
    """Backward-compatible 4h boundary helper."""
    return next_candle_close("4h", now)


def wait_until_close(delay_seconds: int = 10) -> None:
    target = next_4h_close() + timedelta(seconds=delay_seconds)
    seconds = max(0.0, (target - datetime.now(timezone.utc)).total_seconds())
    logging.info("Siguiente cierre 4h: %s (espera %.0f s)", target.isoformat(), seconds)
    time.sleep(seconds)


def run_once(args: argparse.Namespace) -> bool:
    candles = fetch_confirmed_candles(args.symbol, args.timeframe, limit=5)
    alert = evaluate_candles(
        args.symbol,
        candles,
        level=args.price,
        level_direction=args.direction,
        require_engulfing=args.require_engulfing,
        timeframe=args.timeframe,
    )
    if alert is None:
        logging.info("Sin alerta para %s en la última vela cerrada", args.symbol)
        return False
    message = format_alert(alert)
    if args.dry_run:
        print(message)
    else:
        send_telegram(message)
        logging.info("Alerta enviada para %s", args.symbol)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Alertas de velas 4h de OKX por Telegram")
    parser.add_argument("--symbol", required=True, help="Ej. BTC-USDT-SWAP")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--price", type=float, help="Nivel de cierre opcional")
    parser.add_argument("--direction", choices=("above", "below"), help="Dirección del cruce")
    parser.add_argument(
        "--no-engulfing",
        dest="require_engulfing",
        action="store_false",
        help="Alertar solo por cierre del nivel, sin exigir envolvente",
    )
    parser.add_argument("--once", action="store_true", help="Consultar una vez y salir")
    parser.add_argument("--dry-run", action="store_true", help="Mostrar mensaje sin enviarlo")
    parser.add_argument("--delay-seconds", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.price is not None and args.direction is None:
        parser.error("--price requiere --direction above|below")
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    if args.once:
        run_once(args)
        return
    while True:
        wait_until_close(args.delay_seconds)
        try:
            run_once(args)
        except Exception:
            logging.exception("Falló la consulta/envío de la alerta 4h; se reintentará en el próximo cierre")


if __name__ == "__main__":
    main()
