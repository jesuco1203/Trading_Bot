"""
Descarga el histórico de funding de perpetuos de OKX.

El backtest no modela funding, y en posiciones que duran ~36h de mediana se
pagan/cobran unas 4-5 liquidaciones por operación. Con un margen de coste de
~13 bps, eso no es despreciable a priori y hay que medirlo, no estimarlo.

El endpoint pagina hacia atrás con `before`/`after` sobre fundingTime.
Guarda un parquet por símbolo en <root>/<símbolo>/funding.parquet.
"""
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


def fetch_funding(ex, inst_id: str, since_ms: int, until_ms: int, limit: int = 100) -> pd.DataFrame:
    rows, cursor = [], until_ms
    while cursor > since_ms:
        resp = ex.publicGetPublicFundingRateHistory({
            "instId": inst_id, "before": "", "after": str(cursor), "limit": str(limit),
        })
        batch = resp.get("data") or []
        if not batch:
            break
        for r in batch:
            rows.append([int(r["fundingTime"]), float(r["fundingRate"])])
        oldest = int(batch[-1]["fundingTime"])
        if oldest >= cursor:
            break
        cursor = oldest
        time.sleep(ex.rateLimit / 1000.0)

    if not rows:
        return pd.DataFrame(columns=["ts", "rate"])
    df = pd.DataFrame(rows, columns=["ts", "rate"])
    df = df[df.ts >= since_ms]
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--months", type=int, default=84)
    args = ap.parse_args()

    ex = ccxt.okx({"enableRateLimit": True})
    until = dt.datetime.now(dt.timezone.utc)
    since = until - dt.timedelta(days=30 * args.months)

    for sym in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        print(f"[funding] {sym}", flush=True)
        df = fetch_funding(ex, sym, int(since.timestamp() * 1000), int(until.timestamp() * 1000))
        out = os.path.join(args.root, sym, "funding.parquet")
        pathlib.Path(os.path.dirname(out)).mkdir(parents=True, exist_ok=True)
        if not df.empty:
            df.to_parquet(out, index=False)
            print(f"  -> {len(df):,} pagos | {df.ts.iloc[0]} -> {df.ts.iloc[-1]} | "
                  f"tasa media {df.rate.mean()*100:.4f}% | -> {out}", flush=True)
        else:
            print("  -> vacío", flush=True)


if __name__ == "__main__":
    main()
