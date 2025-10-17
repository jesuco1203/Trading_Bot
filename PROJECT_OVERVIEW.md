# 📘 Proyecto Trading_Bot – Resumen Técnico General

## 🧠 Propósito
Desarrollar un **sistema modular de backtesting y validación de estrategias de trading algorítmico** (BTC/USDT 30m) con capacidad de optimización progresiva y posterior transición a *paper trading* en tiempo real.

---

## 🏗️ Arquitectura General

| Módulo | Función | Estado |
|--------|----------|--------|
| `main.py` | Runner principal, orquesta ventanas de backtesting, imprime telemetría y gestiona CSVs. | ✅ Estable |
| `execution/paper.py` | Broker simulado (“PaperBroker”), ejecuta operaciones con control de fees, slippage y stop-loss dinámico. | ✅ Estable |
| `strategies/trend_v2.py` | Estrategia principal TrendV2, basada en ADX, Donchian y EMA separations. | ✅ Optimizable |
| `configs/btc_30m.toml` | Configuración paramétrica (filtros, stops, reentry, relaxer). | ✅ Estricta y auditada |
| `utils/perf.py` | Herramientas de performance y métricas auxiliares (nuevo, pendiente de trackeo en git). | 🟡 Revisar |
| `shared_context` | Diccionario que pasa datos entre módulos (`df_base`, ATR actual, high/low, etc.). | ✅ Obligatorio mantener |

**Contrato clave:**  
```python
broker.mark_to_market(
    close[i], ts=idx[i],
    high=df_base['high'].iloc[:i+1],
    low=df_base['low'].iloc[:i+1],
    current_atr=float(atr.iloc[i]),
    i=i, shared_context={"df_base": df_base}
)
```
⚠️ *No debe modificarse sin coordinación del consultor.*

---

## ⚙️ Configuración Productiva Actual (`configs/btc_30m.toml`)

| Parámetro | Valor | Descripción |
|------------|--------|-------------|
| `min_adx` | 15 | Filtro de tendencia mínima |
| `don_body_min_atr` | 0.18 | Longitud mínima del cuerpo Donchian |
| `don_ema_sep_atr` | 0.15 | Separación mínima EMA/Donchian |
| `ema_min_sep_atr` | 0.18 | Separación EMA-EMA |
| `break_eps_atr` | 0.0024 | Umbral de ruptura |
| `sl_mode` | "capped" | Stop limitado a ATR máximo |
| `sl_swing_extra_atr_cap` | 0.7 | Extensión del stop swing |
| `partial_take_r` | 1.0 | Nivel de toma parcial |
| `partial_take_frac` | 0.33 | Porcentaje parcial |
| `trail_activate_r` | 1.0 | Activación del trailing |
| `relaxer_enabled` | false | Relaxer deshabilitado (productivo) |

---

## 📊 Resultados de Validación

### A/B Test (`partial_take_r`)
- Comparativa entre 1.0R vs 0.85R (4 ventanas)
- Resultado:
  - PF: 0.43–0.59
  - Expectancy: -18 a -24
  - 0.85R no mejora PF ni reduce SL tempranos.
- ✅ Se mantiene `partial_take_r = 1.0`.

### Ajuste Donchian (`don_body_min_atr`)
- Subida 0.15 → 0.18.
- Resultado neutro (mismo #trades, mismos SL tempranos).
- ✅ Mantener 0.18 como valor estable.

### Próxima prueba sugerida
- Subir `break_eps_atr = 0.0026` (de 0.0024) solo para W3, ver impacto en ruido de entradas débiles.

---

## 📈 Telemetría Estándar

Cada corrida válida debe incluir:
```
[CFG EFFECTIVE] {...}
[SUMMARY run_ts=... window=... trades=... SL_le_2=... partials=... max_rr≈...]
[SL DEBUG] i=... mode=capped chosen=...
TriggerCounts {don_L:..., ema_L:...}
```

CSV generado por run:
```
trades_BTC-USDT-SWAP_30m_window_<start>-<end>_<run_ts>.csv
```

---

## 📂 Registros y Artefactos Clave
- `README.md`: resumen de experimentos y resultados A/B.
- `PROJECT_PROGRESS.md`: changelog detallado de tandas.
- `configs/btc_30m.toml`: perfil productivo “estricto”.
- CSVs en `/data/trades_*.csv` con `run_ts` únicos.

---

## 🧩 Recomendaciones para el próximo consultor

1. **No romper contratos.**
   - Mantener firma de `mark_to_market` y `shared_context`.
   - No mover telemetría estándar.
2. **Modificar solo una perilla por tanda.**
   - Cada ajuste debe aislar un efecto y registrarse con `run_ts`.
3. **Mantener relaxer OFF** hasta la fase de real-time testing.
4. **Próxima meta:** probar `break_eps_atr = 0.0026` y luego evaluar mejora en `SL_le_2`.
5. **Trackear utils/** en git antes de cerrar sprint.

---

## 🧭 Nivel de Madurez Actual

| Área | Estado | Comentario |
|------|---------|------------|
| Arquitectura modular | 🟢 Estable |
| Telemetría / Logging | 🟢 Completa |
| Backtesting multi-ventana | 🟢 Validado |
| Control de riesgo | 🟢 Correcto |
| Persistencia (DB) | 🟠 Parcial (solo CSV) |
| Paper trading real-time | 🔴 Pendiente |
