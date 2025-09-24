# Project Progress Summary: AI-Driven Trading Bot

This document summarizes the progress made on the AI-driven trading bot project, highlighting key implementations and the current development roadmap.

## Project Goal
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

## Recent Implementations & Refinements

This section details the significant enhancements and refactorings implemented since the initial setup.

### 1. Dynamic Configuration Profiles
-   **Dedicated Configs:** Introduced separate TOML configuration files (`configs/btc_30m.toml`, `configs/eth_30m.toml`, `configs/btc_4h.toml`) for different trading pairs and timeframes.
-   **Parameterization:** `main.py` now dynamically loads all strategy and selector parameters from these TOML files, allowing for flexible tuning without code changes.
-   **Trend Enable Flag:** Added `ETH30m_enable_trend` flag in `configs/eth_30m.toml` to conditionally enable/disable the Trend strategy for ETH 30m directly from config.

### 2. Enhanced Strategy Logic & Control
-   **BaseStrategy Metrics:** `strategies/base.py` `print_summary` method now accurately calculates and displays `RR Avg` (Risk-Reward Ratio) for each strategy based on entry parameters.
-   **Trend Strategy (`strategies/trend.py`):**
    -   **Configurable ARM:** Parameters for `rbody`, `ADX` thresholds, and `arm_timeout` are now configurable.
    -   **Dynamic SL/TP:** SL (`sl_mult_atr`) and TP (`tp_mult_sl`, `tp_mult_atr`) calculations are now configurable.
    -   **Partial Exits & Trailing:** `partial_tp_atr_mult` and `partial_sl_offset_atr_mult` are configurable.
    -   **New Trailing Stop:** Implemented configurable `trail_trigger_atr_mult` and `trail_sl_offset_atr_mult` for dynamic trailing stop adjustments.
    -   **Entry Filters:** Added `min_pmax` and `max_dist_ma20_atr` guards to filter fragile entries.
    -   **Time-Stop:** Implemented `time_stop_bars` and `time_stop_mfe_atr` to exit trades that do not progress within a set time/MFE.
    -   **Shorts Control:** `allow_shorts` parameter added for explicit control over short entries.
-   **MeanRevert Strategy (`strategies/mean_revert.py`):**
    -   **Configurable Gates:** Parameters for `pmax`, `ADX` (`gate_adx_max`), `ATR%`, and `dist_sma` are now configurable.
    -   **Detailed Block Telemetry:** `print_summary` now provides a detailed breakdown of reasons for strategy inactivity (e.g., `gate_not_mr`, `gate_pmax_low`, `gate_adx_high`, `gate_atr_high`, `too_close_to_sma`, `cooldown`, `no_setup`) with percentage rates.

### 3. Improved PaperBroker & Trade Telemetry
-   **Position Dataclass:** Extended `Position` to track `rr`, `bars_open`, `mfe_atr`, `mae_atr`, `symbol`, and `tf` per trade.
-   **Trade Recording:** `enter_or_flip` and `mark_to_market` methods now correctly capture and store these new metrics in the `trades` list.
-   **CSV Export:** Added `export_trades_to_csv` method to `PaperBroker` for detailed trade analysis, including all captured metrics.

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

### 1. Walk-Forward Validation
-   **Implementation:** Refactor `main.py` to implement a walk-forward validation framework.
-   **Methodology:** Execute backtests across 3 sliding windows (shift 500–750 candles) for each profile (BTC 30m, ETH 30m, BTC 4H).
-   **Reporting:** Generate detailed reports per window, including total PnL, PnL per strategy, Hit%, RR Avg, and percentage of MeanRevert blockages.

### 2. Paper Trading Preparation
-   **Risk Limits:** Implement runtime risk limits (`max_dd_pct_session`, `max_consecutive_losses`, `max_daily_trades`) in `PaperBroker` or a dedicated `RiskManager`.
-   **Hard Stop Logic:** Implement logic to hard stop the bot and log alerts if any risk limits are exceeded.
-   **Commission/Slippage Verification:** Ensure OKX-specific commission and slippage are accurately applied in the broker simulation.

### 3. Further Strategy Tuning (Post Walk-Forward)
-   **ETH 30m Trend:** Re-evaluate and tune the Trend strategy for ETH 30m separately, potentially re-enabling it if it can contribute positively.
-   **MeanRevert ADX Gate:** Optionally, explore increasing `MeanRevert_gate_adx_max` (e.g., to 22) to potentially increase MeanRevert signal frequency, and measure impact.
-   **General Optimization:** Continue refining parameters for all strategies based on walk-forward results.

### 4. Documentation & Maintenance
-   **Project Progress Update:** Maintain this `PROJECT_PROGRESS.md` document with ongoing updates.
-   **Codebase Refinement:** Continuous review and refactoring for code quality and maintainability.