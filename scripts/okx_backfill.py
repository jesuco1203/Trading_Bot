from __future__ import annotations
import os
import time
import argparse
import pathlib
import datetime as dt
import pandas as pd

try:
    import ccxt
except ImportError:
    raise SystemExit("Instala: pip install ccxt pandas pyarrow")

def _ms(d: dt.datetime) -> int: return int(d.timestamp()*1000)
def _mk(p: str): pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def fetch_ohlcv_full(ex, symbol, timeframe, since_ms, until_ms, limit=200):
    out, cursor = [], since_ms
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not batch:
            break
        out += batch
        last = batch[-1][0]
        cursor = last + 1
        if cursor >= until_ms:
            break
        time.sleep(ex.rateLimit/1000.0)
    if not out:
        return pd.DataFrame(columns=["ts","open","high","low","close","volume"])
    df = pd.DataFrame(out, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def save_parquet(df, path):
    if df.empty:
        return
    _mk(os.path.dirname(path))
    if os.path.exists(path):
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates("ts").sort_values("ts")
    df.to_parquet(path, index=False)

def norm_sym(s: str) -> str:
    if ":" in s or "/" in s:
        return s
    base = s.split("-")[0]
    return f"{base}/USDT:USDT"  # OKX perps lineales en ccxt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--symbols", required=True)   # "BTC-USDT-SWAP,ETH-USDT-SWAP"
    ap.add_argument("--timeframes", default="30m,1h,2h,4h")
    ap.add_argument("--months", type=int, default=12)
    args = ap.parse_args()

    ex = ccxt.okx({"enableRateLimit": True})
    until = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    since = until - dt.timedelta(days=30*args.months)

    for sym in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        ccxt_sym = norm_sym(sym)
        for tf in [t.strip() for t in args.timeframes.split(",") if t.strip()]:
            print(f"[backfill] {ccxt_sym} {tf} ({args.months}m)")
            df = fetch_ohlcv_full(ex, ccxt_sym, tf, _ms(since), _ms(until))
            out = os.path.join(args.root, sym.replace("/", "_").replace(":", "_"), tf, "ohlcv.parquet")
            save_parquet(df, out)
            print(f"  -> {len(df):,} rows -> {out}")

if __name__ == "__main__":
    main()