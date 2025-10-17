#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import os
from typing import List

import pandas as pd

try:
    import ccxt
except ImportError as exc:
    raise SystemExit("ccxt is required to run this script") from exc


def parse_date(value: str) -> dt.datetime:
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date '{value}', expected YYYY-MM-DD") from exc


def fetch_ohlcv(exchange: ccxt.okx, symbol: str, timeframe: str, since: dt.datetime, until: dt.datetime) -> List[List[float]]:
    ms_since = int(since.timestamp() * 1000)
    ms_until = int(until.timestamp() * 1000)
    ohlcv: List[List[float]] = []
    limit = 200
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=ms_since, limit=limit)
        if not batch:
            break
        ohlcv.extend(batch)
        ms_last = batch[-1][0]
        if ms_last >= ms_until:
            break
        ms_since = ms_last + exchange.parse_timeframe(timeframe) * 1000
        if ms_since >= ms_until:
            break
    return [row for row in ohlcv if row[0] <= ms_until]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OKX OHLCV data to a parquet file")
    parser.add_argument("--symbol", required=True, help="OKX instrument id, e.g. BTC-USDT-SWAP")
    parser.add_argument("--timeframe", required=True, help="Timeframe, e.g. 30m")
    parser.add_argument("--since", required=True, type=parse_date, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--until", required=True, type=parse_date, help="End date (YYYY-MM-DD)")
    parser.add_argument("--out", required=True, help="Output parquet path")
    args = parser.parse_args()

    exchange = ccxt.okx({"enableRateLimit": True})

    data = fetch_ohlcv(exchange, args.symbol, args.timeframe, args.since, args.until)
    if not data:
        raise SystemExit("No data fetched")

    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(data, columns=columns)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_parquet(args.out)
    print(f"Saved {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
