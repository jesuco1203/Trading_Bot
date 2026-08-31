"""
Filtro ML sobre las entradas de TrendV2 — validación walk-forward.

CÓMO REGENERAR LOS INPUTS
    for s in btc eth sol xrp doge ada ltc bnb; do
      python main.py --config configs/${s}_4h.toml --start-index 0 --limit-bars 20000 \
        | grep '^CSV:' | sed "s|^|SYM=$s |"
    done > oos.log
y apuntar OOS_LOG a ese fichero.

RESULTADO (2026-07-28, 3.830 operaciones, 8 símbolos)
    PF 1.22 -> 1.49, esperanza +0.1056R -> +0.2118R, mejora en 6/6 periodos.
    OJO: el R TOTAL es el mismo (242.7R vs 244.0R). El filtro concentra el
    beneficio en la mitad de operaciones, no lo aumenta. La ganancia se explota
    vía sizing, no vía "operar menos".

NOTAS DE METODOLOGÍA
  - Split temporal siempre (entrenar con el pasado, evaluar el futuro).
  - El recorte de colas usa cuantiles SÓLO de train: hacerlo sobre el dataset
    completo es lookahead.
  - 'vol' y 'rng_pct' quedan excluidas a propósito: features/core.py:82-83 las
    recorta con quantile() de la serie entera, lo que también es lookahead.
  - La logística generaliza (AUC 0.576 train / 0.575 test); el gradient boosting
    memoriza (0.850 / 0.568). Con ~3.800 muestras, más capacidad = más overfit.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/Users/jesuco1203/Docs_mac/Trading_Bot")
from features.core import build_features

OOS_LOG = "oos.log"  # ver cabecera
D = ""
runs = {r.split()[0][4:]: r.split()[2] for r in open(D + OOS_LOG)}

FEATS = ["adx", "di_plus", "di_minus", "atr_pct", "rsi14", "z_score_50",
         "d_ema10", "d_ema50", "d_ema200", "ema_sep", "body_frac",
         "hour", "side", "ret5", "ret20", "atr_ratio"]

rows = []
for sym, csv in runs.items():
    df = pd.read_parquet(f"data/okx/{sym.upper()}-USDT-SWAP/4h/ohlcv.parquet").set_index("ts")
    f = build_features(df); c = f["close"]
    f["d_ema10"] = (c - f.ema_10) / f.atr
    f["d_ema50"] = (c - f.ema_50) / f.atr
    f["d_ema200"] = (c - f.ema_200) / f.atr
    f["ema_sep"] = (f.ema_50 - f.ema_200) / f.atr
    f["body_frac"] = (c - f.open).abs() / (f.high - f.low).replace(0, np.nan)
    f["hour"] = f.index.hour
    f["ret5"] = c.pct_change(5); f["ret20"] = c.pct_change(20)
    f["atr_ratio"] = f.atr_pct / f.atr_pct.rolling(50).mean()
    d = pd.read_csv(csv, parse_dates=["entry_ts"])
    g = d.groupby("trade_id").agg(pnl=("pnl", "sum"), ts=("entry_ts", "first"), side=("side", "first"))
    j = g.join(f.drop(columns=["side"], errors="ignore"), on="ts"); j["sym"] = sym
    rows.append(j)

a = pd.concat(rows).sort_values("ts")
a[FEATS] = a[FEATS].replace([np.inf, -np.inf], np.nan)
a = a.dropna(subset=FEATS)
a["y"] = (a.pnl > 0).astype(int); a["R"] = a.pnl / 100.0
# recorte SOLO con cuantiles de train (se hace dentro del fold)

pf = lambda s: s[s > 0].sum() / abs(s[s < 0].sum()) if (s < 0).any() else np.inf
N = len(a); FOLDS = 6; start = int(N * 0.40); step = (N - start) // FOLDS

print(f"walk-forward: {FOLDS} periodos, entrenamiento expansivo, {N} operaciones\n")
print(f"{'periodo':22} {'train':>6} {'test':>6} {'AUC':>6} {'PF base':>8} {'PF top50%':>10} {'delta':>7}")
print("-" * 72)
mej = 0
allbase, alltop = [], []
for k in range(FOLDS):
    i0 = start + k * step
    i1 = N if k == FOLDS - 1 else start + (k + 1) * step
    tr, te = a.iloc[:i0].copy(), a.iloc[i0:i1].copy()
    if len(te) < 30: continue
    # recorte de colas con cuantiles calculados SOLO en train
    for col in FEATS:
        lo, hi = tr[col].quantile([0.005, 0.995])
        tr[col] = tr[col].clip(lo, hi); te[col] = te[col].clip(lo, hi)
    p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    p.fit(tr[FEATS].values, tr.y.values)
    sc = p.predict_proba(te[FEATS].values)[:, 1]
    try: auc = roc_auc_score(te.y.values, sc)
    except ValueError: auc = np.nan
    base = te.R; top = te.R[sc >= np.median(sc)]
    d = pf(top) - pf(base)
    if d > 0: mej += 1
    allbase.append(base); alltop.append(top)
    per = f"{te.ts.min():%Y-%m}->{te.ts.max():%Y-%m}"
    print(f"{per:22} {len(tr):6d} {len(te):6d} {auc:6.3f} {pf(base):8.2f} {pf(top):10.2f} {d:+7.2f}")

B, T = pd.concat(allbase), pd.concat(alltop)
print("-" * 72)
print(f"{'AGREGADO':22} {'':6} {len(B):6d} {'':6} {pf(B):8.2f} {pf(T):10.2f} {pf(T)-pf(B):+7.2f}")
print(f"\nperiodos en que el filtro mejora: {mej}/{FOLDS}")
print(f"esperanza: base {B.mean():+.4f}R -> filtrada {T.mean():+.4f}R  ({len(T)} de {len(B)} ops)")

# ¿el modelo aprende algo trivial (símbolo/lado) o algo real?
p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
p.fit(a[FEATS].values, a.y.values)
co = pd.Series(p[-1].coef_[0], index=FEATS).sort_values(key=abs, ascending=False)
print("\npeso de cada feature (coeficientes estandarizados):")
for k, v in co.head(8).items():
    print(f"   {k:12} {v:+.3f}")
