# Project Progress Summary: AI-Driven Trading Bot

This document summarizes the progress made on the AI-driven trading bot project, highlighting key implementations, challenges, and the current blocking issue.

## Project Goal
The primary goal is to enhance the existing OKX trading bot by incorporating Artificial Intelligence (AI) to dynamically select trading strategies based on market cycles, aiming to maximize risk-adjusted returns.

## Implemented Features (Hito 1 - Initial Setup)

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
    -   Filters out signals with low confidence (`pmax < 0.45`).
    -   Currently configured to only activate strategies if the regime is "trend" (for testing purposes).

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

## Current Status & Blocking Issue

The project has made significant progress, with the end-to-end backtesting pipeline largely implemented. The HMM is now learning distinct states, and the `Trend` strategy has been refined with more robust entry conditions.

**However, a critical blocking issue persists:**

-   **Inability to Read/Modify `main.py`:** I am currently encountering a persistent technical issue that prevents me from reliably reading or modifying the `main.py` file. Despite multiple attempts using different tools (`read_file`, `replace`, `read_many_files`), I am unable to access its content or perform the requested modifications. This has led to a loop where I cannot apply necessary changes to `main.py` to continue debugging and refining the bot.

This limitation prevents further progress on implementing the dynamic `adj_risk` calculation and other necessary changes in `main.py`.

## Next Steps (Pending Resolution of Blocking Issue)

Once the `main.py` file becomes accessible and modifiable, the immediate next steps would be to:
1.  **Implement dynamic `adj_risk` calculation in `main.py`:** This was the last pending change from the consultant's previous instructions.
2.  **Run validation:** Execute `main.py` with the specified configuration (BTC 30m, --limit-bars 2000) to evaluate the impact of all recent changes.
3.  **Continue refinement:** Based on the validation results, further refine strategy parameters, HMM configuration, and explore additional strategies as per the consultant's roadmap.
