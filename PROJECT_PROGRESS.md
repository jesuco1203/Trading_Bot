# Project Goal
The primary goal is to enhance the existing OKX trading bot by incorporating Artificial Intelligence (AI) to dynamically select trading strategies based on market cycles, aiming to maximize risk-adjusted returns.

## Implemented Features (Hito 1 - Initial Setup)
(This section remains largely as is, covering the foundational setup)

### 1. Project Structure & Configuration
-   **Directory Structure:** Created modular directories (`data`, `features`, `regime`, `strategies`, `selector`, `risk`, `execution`, `monitoring`, `scripts`).
-   **Configuration (`config.toml`):** Migrated from `config.json` to `config.toml`, incorporating new parameters for HMM, risk management, and run configurations.

### 2. Data Acquisition & Processing
-   **Historical Data Backfill (`scripts/okx_backfill.py`):** Implemented a script to download historical OHLCV data from OKX (BTC, ETH, ATOM across multiple timeframes) and store it in Parquet files.
-   **Data Loading (`data/okx_client.py`):** Updated to read OHLCV data from local Parquet files.
-   **Feature Engineering (`features/core.py`):** Implemented calculation of `ret`, `vol`, `rng_pct`, `atr`, and `atr_pct`. Added FVG and basic candle pattern detection. Implemented outlier clipping for robustness.

### 3. Market Regime Detection (HMM)
-   **RegimeHMM (`regime/hmm.py`):** Implemented a Gaussian Hidden Markov Model (HMM) for market regime detection.
    -   Uses `StandardScaler` for feature normalization.
    -   Employs KMeans for robust initialization of HMM means.
    -   Configured for 3 states (`trend`, `mr`, `high_vol`).
    -   Includes warm-start for retraining.
    -   `covariance_type` set to `"diag"` for stability.
-   **RegimeSelector (`selector/regime_switch.py`):** Selects active strategies based on HMM probabilities.
    -   Now supports configurable `enter_th`, `exit_th`, and `persistence`.
    -   Improved logging to differentiate reasons for strategy inactivity (e.g., `mapping_empty` instead of generic `no_mapping_for_*`).

### 4. Trading Strategy (`strategies/trend.py`)
-   **Trend Strategy:** Implemented a refined MA-cross strategy.
    -   Uses MA 20/50 periods.
    -   Incorporates dynamic separation based on volatility (`sep_min`).
    -   Includes alternative entry by price displacement if MA cross is "sticky".
    -   Features a cooldown period after a Stop-Loss (SL) event.
    -   Soft confluence from `score_multiTF` (adjusts strength, not blocking).

### 5. Paper Trading Execution & Metrics
-   **PaperBroker (`execution/paper.py`):** Simulates trade execution with commissions and slippage.
    -   Checks Stop-Loss (SL) and Take-Profit (TP) using high/low of the candle.
    -   Calls `on_stop` callback on SL.
-   **Position Sizing (`risk/position_sizing.py`):** Calculates position size based on ATR and risk percentage.
-   **RollingMetrics (`monitoring/metrics.py`):** Tracks rolling Sharpe, Max Drawdown, and other performance metrics.

### 6. Main Loop Orchestration (`main.py`)
-   Orchestrates the entire backtesting pipeline.
-   Fetches multi-timeframe data.
-   Calculates `score_multiTF` based on weighted scores from different timeframes.
-   Passes relevant context to strategies.
-   Implements dynamic risk adjustment based on HMM confidence and signal strength.

---

## Hito 2 - Risk Management & Strategy Refinement (Implemented during current interaction)

This section details significant enhancements and refactorings implemented to stabilize the backtesting framework, harden risk controls, and introduce a new strategy.

### 1. Core Backtesting Framework Enhancements
-   **Argparse Flags**: Added command-line flags (`--mr-only`, `--fixed-qty`, `--no-fees`, `--taker-fee`) for flexible backtest configuration.
-   **Robust Walk-Forward Validation**: Refactored `main.py` to correctly implement non-overlapping walk-forward windows, ensuring each backtest runs on a distinct data segment.
-   **Unified PnL Accounting**: Implemented a single, unified `_close_position` method in `PaperBroker` to centralize PnL calculation, fee handling, and logging for all trade exits (SL, TP, time-stops, session-end).
-   **Corrected Partial Fill PnL**: Fixed the PnL calculation for partial fills to prevent anomalous losses.
-   **Enhanced Reporting**: Improved the final backtest report to accurately reflect PnL per strategy, entry counts, and PnL broken down by long/short trades, all derived directly from the trade ledger.

### 2. Risk Management System (Hardened)
-   **Unit-Safe Position Sizing**: Replaced basic ATR-based sizing with a robust `compute_qty_for_stop` function that handles linear/inverse contracts and applies strict caps on notional value and estimated loss per trade.
-   **Pre-Trade Risk Enforcement**: Implemented hard caps on `qty` based on `max_notional_usd` and `max_loss_usd` *before* a trade is opened, preventing oversized positions.
-   **Post-Trade PnL Audit**: Added `AssertionError` checks at trade closure to immediately halt backtests if actual losses exceed configured limits, ensuring risk compliance.
-   **Adaptive Time-Stops**: Implemented adaptive time-stop logic for `Trend` strategy to close stagnant trades or extend promising ones.
-   **Soft Break-Even**: Implemented a soft break-even mechanism after partial fills to protect profits.

### 3. Strategy Enhancements
-   **`MeanRevert` Strategy (Defensive)**:
    -   **HMM-Aware Gates**: Implemented strict gates to block `MeanRevert` trades during strong trend regimes (HMM-detected trend or high ADX) and high volatility (ATR percentile).
    -   **Stricter Signal Confluence**: Tightened thresholds for Z-score, RSI, distance from SMA, and wick size to require clearer reversal patterns.
-   **`TrendV2` Strategy (New & Active)**:
    -   **New Strategy Implementation**: Created `strategies/trend_v2.py` from scratch, based on Donchian Channel breakouts with pullback confirmation.
    -   **Symmetrical Long/Short Logic**: Fully implemented symmetrical logic for both long and short trades.
    -   **HMM-Driven Dynamic Gates**: Implemented adaptive thresholds for `sep_min`, `adx_min`, and `bar_expand_k` based on HMM regime, making the strategy more reactive in trending markets.
    -   **Stateful Momentum Override**: Implemented a controlled `momentum_override` mechanism with strict activation conditions, budgeting (cooldowns, minimum gap between trades), and detailed telemetry.
    -   **Adaptive Pullback Window**: Extended the pullback window for Setup A dynamically based on HMM trend mode.

### 4. Telemetry & Debugging
-   2024-11-16: TrendV2 strict profile – bumped `don_body_min_atr` to 0.18 (W3 trades=59, SL_le_2=8) and tested tighter breakout gate `break_eps_atr=0.0026`; run_ts=1760680095 logged TriggerCounts `{'don_L': 58, 'don_S': 48, 'ema_L': 9, 'ema_S': 7}`, SL debug confirms `mode='capped'`, CSV `trades_BTC-USDT-SWAP_30m_window_4000-6000_1760680095.csv`.
-   2024-11-16: Reverted `break_eps_atr` to 0.0024 (neutral impact); W3 run_ts=1760681747 → trades=59, SL_le_2=8, TriggerCounts `{'don_L': 58, 'don_S': 48, 'ema_L': 9, 'ema_S': 7}`, CSV `trades_BTC-USDT-SWAP_30m_window_4000-6000_1760681747.csv`.
-   2024-11-16: EMA sweep – `ema_min_sep_atr = 0.20` → run_ts=1760681759, trades=58, SL_le_2=7, TriggerCounts `{'don_L': 58, 'don_S': 48, 'ema_L': 7, 'ema_S': 7}`, CSV `trades_BTC-USDT-SWAP_30m_window_4000-6000_1760681759.csv` (neutral).
-   2024-11-16: EMA sweep – `ema_min_sep_atr = 0.22` → run_ts=1760681770, trades=57, SL_le_2=6, TriggerCounts `{'don_L': 58, 'don_S': 48, 'ema_L': 6, 'ema_S': 7}`, CSV `trades_BTC-USDT-SWAP_30m_window_4000-6000_1760681770.csv` (improved SL_le_2 by ~25% with 3% trade reduction). (Baseline restablecida a 0.18 tras la tanda 2024-11-20.)
-   2024-11-20: Donchian EMA separation `don_ema_sep_atr` sweep (0.12–0.21) → run_ts=1760717269/1760717286/1760717291 (W3), 1760717376/1760717386/1760717391/1760717395 (W1), 1760717403/1760717424 (W2). TriggerCounts con `don_sep_rejects_*` activos, `DON_SEP_ATR_STATS` p50≈0.38. PF, Expectancy y SL≤2/trade sin cambios; baseline se mantiene en 0.15.
-   2024-11-20: Reentry sweep – `reentry_eps_atr` ∈ {0.0007, 0.0010, 0.0013} (W3) → run_ts=1760719767/1760719755/1760719773. Armado visible (reentry_armed>0) pero sin ejecución (reentry_exec=0). Telemetría `REENTRY_COUNTS` activa (`reject_sl_too_late` dominante). baseline queda en 0.0010.
-   2024-11-21: Reentry diagnostic (W3, exec OFF) run_ts=1760727301 → baseline `reentry_exec_enabled=false`; lateness p50=0.0, RR p75=1.41, `reject_sl_too_late=46`, `passes_all=7`, `reentry_exec=0`.
-   2024-11-21: Adaptive gate test (W3, overrides ON) run_ts=1760727308 → `reentry_exec_enabled=true`, `reentry_exec=6`, `relax_gate_used=0`; PF≈0.0034 (vs 0.0036 OFF), Expectancy≈-103.94 (↓0.41). Resultado: ejecución funcional pero sin mejora → flag se mantiene OFF por defecto.
-   2024-11-21: W1 ON check (run_ts=1760727323) → `reentry_exec=7`, `relax_gate_used=0`, PF≈0.0146, Expectancy≈-93.95 (≈baseline−0.09); flag OFF por defecto.
-   2024-11-21: W2 ON check (run_ts=1760727330) → `reentry_exec=5`, `relax_gate_used=0`, PF≈0.0067, Expectancy≈-99.13 (≈baseline−0.12); flag OFF por defecto.
-   2024-11-16: Cross-check W1 (`ema_min_sep_atr = 0.22`) – run_ts=1760682036, TriggerCounts `{'don_L': 54, 'don_S': 39, 'ema_L': 4, 'ema_S': 4}`, trades=61, SL_le_2=7, CSV `trades_BTC-USDT-SWAP_30m_window_0-2000_1760682036.csv` (consistent).
-   2024-11-16: Cross-check W2 (`ema_min_sep_atr = 0.22`) – run_ts=1760682041, TriggerCounts `{'don_L': 70, 'don_S': 40, 'ema_L': 4, 'ema_S': 11}`, trades=61, SL_le_2=7, CSV `trades_BTC-USDT-SWAP_30m_window_2000-4000_1760682041.csv` (consistent).
-   Added extensive logging (`WF SLICE`, `MR SIZE DEBUG`, `MR PARTIAL DEBUG`, `MR CLOSE DEBUG`, `TREND SIZE DEBUG`, `TREND MO BLOCK/ALLOW`, `TREND MO ENTER/CLOSE`) to provide granular insight into strategy decisions, risk enforcement, and trade lifecycle.

### 5. Technical Housekeeping
-   **`pkg_resources` Warning Mitigation**: Implemented a deferred import for `pandas_ta` in `strategies/trend_v2.py` using `importlib`. This prevents the `pkg_resources` `UserWarning` from being triggered on startup, keeping CI/CD logs clean. This is the cleaner, more robust solution compared to pinning dependencies.

---

## Hito 3: Frozen Parameters & Final Validation

This section documents the final, validated parameters used for the successful walk-forward analysis, establishing a baseline for future robustness tests.

-   **`TrendV2` Strategy:**
    -   **Core Tuning:** `n_donchian=12`, `adx_min(trend)=15`, `sep_min(trend)=0.0015`.
    -   **Setup Logic:** `PULLBACK_MAX_BARS` set to 8 (Setup A) and 6 (Setup B).
    -   **Trade Management:** Partial TP at 1xATR, followed by a soft Break-Even stop at `entry + 0.3*ATR`. Includes an adaptive time-stop.
    -   **Risk:** Unit-safe position sizing with pre-trade notional and max loss caps.

-   **`MeanRevert` (MR) Strategy:**
    -   **Status:** Active but defensive.
    -   **Gates:** Strict trend/volatility gates are active, which resulted in zero entries during the final validation windows.
    -   **Risk:** Per-strategy kill-switch is active (`max_consecutive_losses=3`).

-   **Cost Model:**
    -   `taker_fee = 0.0005` (5 bps)
    -   `maker_fee = 0.0002` (2 bps)
    -   `slippage_bps = 1.0` (1 bps, adverse)

---

## Current Profile Status & Validation Results

This section summarizes the performance and configuration status of each trading profile based on recent backtests.

-   **BTC 30m (Combined Baseline):**
    -   **Status:** Profitable.
    -   **Net PnL:** +4.87 (from initial baseline).
    -   **Configuration:** Uses a combination of Trend and MeanRevert strategies with optimized parameters.
-   **ETH 30m (MR-only):**
    -   **Status:** Profitable.
    -   **Net PnL:** +4.15 (after recent tuning).
    -   **Configuration:** Currently runs MeanRevert strategy only (`Trend` temporarily disabled via `strategy_mapping` and `ETH30m_enable_trend` flag). MeanRevert parameters are tuned for ETH 30m.
    -   **Key Insight:** `gate_adx_high` is the dominant reason for MeanRevert not generating signals (approx. 73% of blocks).
-   **BTC 4H (Trend-only):**
    -   **Status:** Profitable.
    -   **Net PnL:** +10.48.
    -   **Configuration:** Runs Trend strategy only (`MeanRevert` disabled). Trend parameters are robustly tuned for 4H timeframe.

---

## Pending Tasks & Next Steps

This section outlines the remaining development tasks and the immediate roadmap for the project.

### 1. Finalize Reporting Consistency
-   **Action**: Ensure `PnL Long Trades` and `PnL Short Trades` are correctly displayed in the final report for all windows, and that the `TrendV2 Entries (Total/L/S)` count is accurate. This requires a final pass on the reporting pipeline in `main.py`.

### 2. Optimize `TrendV2` Frequency
-   **Goal**: Achieve ≥3 entries per window consistently across all walk-forward windows.
-   **Current Status**: Frequency is 1, 3, 2 entries in the last validation.
-   **Next Micro-adjustment**: Based on `Top Blocks` analysis (to be performed after fixing reporting), apply a single, controlled adjustment to `TrendV2`'s filters (e.g., `N_DON`, `PULLBACK_MAX_BARS`, `sep_min`, `adx_min`) to increase entry frequency without compromising risk.

### 3. Paper Trading Preparation (Original Pending Task)
-   **Risk Limits:** Implement runtime risk limits (`max_dd_pct_session`, `max_consecutive_losses`, `max_daily_trades`) in `PaperBroker` or a dedicated `RiskManager`.
-   **Hard Stop Logic:** Implement logic to hard stop the bot and log alerts if any risk limits are exceeded.
-   **Commission/Slippage Verification:** Ensure OKX-specific commission and slippage are accurately applied in the broker simulation.

### 4. Further Strategy Tuning (Original Pending Task)
-   **ETH 30m Trend:** Re-evaluate and tune the Trend strategy for ETH 30m separately, potentially re-enabling it if it can contribute positively.
-   **MeanRevert ADX Gate:** Optionally, explore increasing `MeanRevert_gate_adx_max` (e.g., to 22) to potentially increase MeanRevert signal frequency, and measure impact.
-   **General Optimization:** Continue refining parameters for all strategies based on walk-forward results.

### 5. Documentation & Maintenance (Original Pending Task)
-   **Project Progress Update:** Maintain this `PROJECT_PROGRESS.md` document with ongoing updates.
-   **Codebase Refinement:** Continuous review and refactoring for code quality and maintainability.
