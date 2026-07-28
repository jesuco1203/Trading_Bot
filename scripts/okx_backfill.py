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

# OKX usa minuscula por debajo de 1H y mayuscula a partir de 1H.
_BAR = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "12h": "12H",
    "1d": "1D", "1w": "1W",
    # OKX cierra 1D/1W en horario Hong Kong (16:00 UTC). Estos alias cierran en UTC.
    "1dutc": "1Dutc", "1wutc": "1Wutc", "6hutc": "6Hutc", "12hutc": "12Hutc",
}

def _ms(d: dt.datetime) -> int: return int(d.timestamp()*1000)
def _mk(p: str): pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def inst_id(s: str) -> str:
    """'BTC-USDT-SWAP' o 'BTC/USDT:USDT' -> instId de OKX."""
    if "/" in s or ":" in s:
        base = s.split("/")[0]
        return f"{base}-USDT-SWAP"
    return s

def fetch_ohlcv_full(ex, symbol, timeframe, since_ms, until_ms, limit=100):
    """
    Descarga histórico profundo vía /market/history-candles.

    El endpoint por defecto (fetch_ohlcv / market/candles) solo sirve las ~1440
    velas más recientes y devuelve vacío ante un `since` antiguo. history-candles
    pagina hacia ATRÁS con `after` (ms exclusivo) y llega hasta el listado del
    instrumento (~dic-2019 para BTC-USDT-SWAP).
    """
    bar = _BAR.get(timeframe.lower())
    if bar is None:
        raise SystemExit(f"Timeframe no soportado por OKX: {timeframe}")

    iid = inst_id(symbol)
    rows, cursor = [], until_ms
    while cursor > since_ms:
        resp = ex.publicGetMarketHistoryCandles({
            "instId": iid, "bar": bar, "after": str(cursor), "limit": str(limit),
        })
        batch = resp.get("data") or []
        if not batch:
            break
        # OKX devuelve [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm], nuevo->viejo.
        for r in batch:
            if len(r) > 8 and r[8] != "1":
                continue  # vela aún abierta
            rows.append([int(r[0])] + [float(x) for x in r[1:6]])
        oldest = int(batch[-1][0])
        if oldest >= cursor:
            break  # sin progreso: corta en vez de girar en vacío
        cursor = oldest
        time.sleep(ex.rateLimit/1000.0)

    if not rows:
        return pd.DataFrame(columns=["ts","open","high","low","close","volume"])
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
    df = df[df["ts"] >= since_ms]
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)

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
    until = dt.datetime.now(dt.timezone.utc)
    since = until - dt.timedelta(days=30*args.months)

    for sym in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        for tf in [t.strip() for t in args.timeframes.split(",") if t.strip()]:
            print(f"[backfill] {inst_id(sym)} {tf} ({args.months}m)", flush=True)
            df = fetch_ohlcv_full(ex, sym, tf, _ms(since), _ms(until))
            out = os.path.join(args.root, sym.replace("/", "_").replace(":", "_"), tf, "ohlcv.parquet")
            save_parquet(df, out)
            span = f"{df['ts'].iloc[0]} -> {df['ts'].iloc[-1]}" if not df.empty else "vacío"
            print(f"  -> {len(df):,} rows | {span} -> {out}", flush=True)

if __name__ == "__main__":
    main()
