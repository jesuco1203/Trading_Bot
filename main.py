import argparse
import logging
import pandas as pd
import toml
import numpy as np
from datetime import datetime, timedelta

from data.okx_client import get_ohlcv
from features.core import build_features
from regime.hmm import RegimeHMM
from selector.regime_switch import RegimeSelector
from strategies.trend import Trend
from strategies.mean_revert import MeanRevert
from strategies.vol_breakout import VolBreakout
from execution.paper import PaperBroker
from risk.position_sizing import atr_position_size
from monitoring.metrics import RollingMetrics

logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler()])

def main(config_path: str, limit_bars: int):
    """
    Main function to run the backtesting pipeline.
    """
    # 1. Load Configuration
    try:
        config = toml.load(config_path)
        logging.info(f"Configuration loaded from {config_path}")
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        return

    # 3. Load Data
    symbol = config['market']['symbols'][0]
    base_tf = config['market']['timeframes'][0]
    
    df_base = get_ohlcv(symbol, base_tf, limit=limit_bars, root=config['data']['root'])
    if df_base.empty:
        logging.error("No data loaded. Exiting.")
        return
        
    df_base = build_features(df_base)
    feats_base = df_base.copy()

    # Sanity check for ADX
    logging.info(f"ADX min/max: {feats_base['adx'].min():.1f} / {feats_base['adx'].max():.1f}")

    # === ATR% percentiles ===
    atr_pct_series = (feats_base["atr"] / feats_base["close"]).clip(lower=0).fillna(0.0)
    atr_pct_85_percentile = float(atr_pct_series.quantile(0.85))
    atr_pct_70_percentile = float(atr_pct_series.quantile(0.70))

    # 4. HMM Training & Regime Prediction
    features_for_hmm = ['ret', 'vol', 'rng_pct']
    hmm_model = RegimeHMM(n_states=config['hmm']['n_states'])
    eps = 1e-4
    X_fit_j = df_base[features_for_hmm].values + np.random.normal(0.0, eps, df_base[features_for_hmm].values.shape)
    hmm_model.fit(X_fit_j)
    logging.info(f"HMM state counts: {np.bincount(hmm_model.model.predict(hmm_model.scaler.transform(df_base[features_for_hmm].values)))}")
    regime_probas = hmm_model.predict_proba(df_base[features_for_hmm].values)
    for i, proba_dict_item in enumerate(regime_probas):
        for state, proba in proba_dict_item.items():
            df_base.loc[df_base.index[i], f'{state}_proba'] = proba

    # 2. Initialize Components
    all_strategies = {
        "Trend": Trend(
            atr_pct_80_percentile=atr_pct_85_percentile,
            risk_mult=config['strategy_params']['Trend_risk_mult'],
            arm_rbody_th=config['strategy_params']['Trend_arm_rbody_th'],
            arm_adx_th=config['strategy_params']['Trend_arm_adx_th'],
            arm_adx_delta_th=config['strategy_params']['Trend_arm_adx_delta_th'],
            arm_timeout=config['strategy_params']['Trend_arm_timeout'],
            retest_eps_pct=config['strategy_params']['Trend_retest_eps_pct'],
            reconfirm_mult_1=config['strategy_params']['Trend_reconfirm_mult_1'],
            reconfirm_mult_2=config['strategy_params']['Trend_reconfirm_mult_2'],
            sl_mult_atr=config['strategy_params']['Trend_sl_mult_atr'],
            tp_mult_sl=config['strategy_params']['Trend_tp_mult_sl'],
            tp_mult_atr=config['strategy_params']['Trend_tp_mult_atr'],
            partial_tp_atr_mult=config['strategy_params']['Trend_partial_tp_atr_mult'],
            partial_sl_offset_atr_mult=config['strategy_params']['Trend_partial_sl_offset_atr_mult'],
            sl_cooldown_duration=config['strategy_params']['Trend_sl_cooldown_duration'],
            allow_shorts=config['strategy_params']['Trend_allow_shorts']
        ),
        "MeanRevert": MeanRevert(
            risk_mult=config['strategy_params']['MeanRevert_risk_mult'],
            gate_pmax_th=config['strategy_params']['MeanRevert_gate_pmax_th'],
            gate_adx_th=config['strategy_params']['MeanRevert_gate_adx_th'],
            gate_atr_pct_th=config['strategy_params']['MeanRevert_gate_atr_pct_th'],
            dist_sma_mult=config['strategy_params']['MeanRevert_dist_sma_mult'],
            signal_z_th=config['strategy_params']['MeanRevert_signal_z_th'],
            signal_rsi_th=config['strategy_params']['MeanRevert_signal_rsi_th'],
            signal_rbody_th=config['strategy_params']['MeanRevert_signal_rbody_th'],
            signal_lower_wick_th=config['strategy_params']['MeanRevert_signal_lower_wick_th'],
            sl_mult_atr=config['strategy_params']['MeanRevert_sl_mult_atr'],
            sl_swing_low_bars=config['strategy_params']['MeanRevert_sl_swing_low_bars'],
            tp_mult_sl=config['strategy_params']['MeanRevert_tp_mult_sl'],
            tp_mult_atr=config['strategy_params']['MeanRevert_tp_mult_atr'],
            partial_tp_atr_mult=config['strategy_params']['MeanRevert_partial_tp_atr_mult'],
            partial_sl_offset_atr_mult=config['strategy_params']['MeanRevert_partial_sl_offset_atr_mult'],
            local_cooldown_duration=config['strategy_params']['MeanRevert_local_cooldown_duration']
        ),
        "VolBreakout": VolBreakout()
    }
    strategy_mapping = config['strategy_mapping']
    broker = PaperBroker(
        initial_capital=config['risk']['starting_equity'],
        comm_rate=config['costs']['commission_rate'],
        slippage_min=config['costs']['slippage_min'],
        all_strategies=all_strategies
    )
    regime_selector = RegimeSelector(
        hmm_model,
        strategy_mapping,
        enter_th=config['selector']['enter_th'],
        exit_th=config['selector']['exit_th'],
        persistence=config['selector']['persistence']
    )
    metrics = RollingMetrics()
    
    # 5. Backtesting Loop
    consecutive_trend_count = 0
    last_entry_i = -10**9
    last_exit_i  = -10**9
    entry_cooldown = 3
    exit_cooldown  = 3
    warmup_period = max(s.warmup_bars() for s in all_strategies.values())
    equity = broker.get_equity()
    strategies = all_strategies.values()
    tick_value = config['market']['tick_value']
    p_prev = {label: 1.0 / hmm_model.n_states for label in hmm_model.labels}

    for i in range(warmup_period, len(df_base)):
        current_dt = df_base.index[i]
        row = df_base.iloc[i]
        atr_t = float(row['atr'])
        prev_exposure = broker.exposure()

        # HMM Probs
        proba_dict_now = {label: row.get(f"{label}_proba", 0.0) for label in hmm_model.labels}
        tot = sum(proba_dict_now.values()) or 1.0
        proba_dict_now = {k: v/tot for k, v in proba_dict_now.items()}
        proba_dict_smooth = {k: 0.2*p_prev[k] + 0.8*proba_dict_now[k] for k in hmm_model.labels}
        p_prev = proba_dict_smooth
        K = len(hmm_model.labels)
        proba_dict = {k: 0.85*proba_dict_smooth[k] + 0.15*(1.0/K) for k in hmm_model.labels}
        pmax = max(proba_dict.values())
        pmax_used = min(pmax, 0.75)
        lab = max(proba_dict, key=proba_dict.get)

        # MTM and Position Update
        pnl_bar = broker.mark_to_market(float(row["close"]), ts=current_dt, high=float(row["high"]), low=float(row["low"]))
        equity += pnl_bar
        metrics.update(current_dt, equity, broker.exposure())
        if prev_exposure != 0 and broker.exposure() == 0:
            last_exit_i = i

        active_names, selector_reason = regime_selector.active_strategies_with_reason(proba_dict)
        if i % 200 == 0:
            logging.info(
                f"[{i}] SELECTOR lab={lab} pmax={pmax:.2f} reason={selector_reason} active={sorted(active_names)} entries={broker.entries_count} exits={broker.exits_count} partials={broker.partials_count} flips={broker.flips_count} eq={broker.get_equity():.2f}"
            )

        # Strategy Gating & Signal Generation
        allow_new_entry = (broker.exposure() == 0)
        if not allow_new_entry:
            continue

        consecutive_trend_count = consecutive_trend_count + 1 if (lab == "trend" and pmax >= regime_selector.enter_th) else 0

        if not active_names:
            continue
        if (i - last_entry_i) < entry_cooldown or (i - last_exit_i) < exit_cooldown:
            continue

        signals = []
        for s in strategies:
            if s.name not in active_names:
                continue
            if s.name == "Trend" and consecutive_trend_count < 2:
                continue
            
            context = {
                "i": i, "ts": current_dt, "df": df_base.iloc[:i+1], "feats": feats_base.iloc[:i+1],
                "equity": equity, "score_multiTF": float(row.get("score_multiTF", 0.0)),
                "atr_pct_p85": atr_pct_85_percentile, "atr_pct_p70": atr_pct_70_percentile,
                "regime_label": lab, "pmax": pmax
            }
            sig = s.signal(context)
            if sig and sig.side != "flat":
                signals.append((s.name, sig))

        if not signals:
            continue

        # Order Execution
        strat_name, sig = signals[0]
        was_flat = (prev_exposure == 0)
        strat = all_strategies[strat_name]
        base_risk = config['risk']['risk_per_trade_pct']
        adj_risk = base_risk * strat.risk_mult * (0.5 + 0.5*pmax_used) * (0.5 + 0.5*sig.strength)
        qty = atr_position_size(equity=equity, atr=atr_t, risk_pct=adj_risk, tick_value=tick_value)
        
        if qty > 0:
            price = float(row["close"])
            broker.enter_or_flip(
                side=1 if sig.side == "long" else -1, qty=qty, price=price,
                sl_pts=sig.sl_pts, tp_pts=sig.tp_pts, partial_tp_pts=sig.partial_tp_pts, 
                ts=current_dt, strategy_name=strat_name, atr=atr_t, partial_sl_offset_atr_mult=sig.partial_sl_offset_atr_mult
            )
            if was_flat:
                broker.entries_count += 1
                last_entry_i = i

            adx = feats_base['adx'].iloc[i]
            atr_pct = atr_t / price
            rr = sig.tp_pts / sig.sl_pts if sig.sl_pts > 0 else 0
            logging.info(f"ENTER i={i} strat={strat_name} pmax={pmax:.2f} adx={adx:.1f} atr%={atr_pct:.3f} rr={rr:.2f} sl={sig.sl_pts:.1f} tp={sig.tp_pts:.1f} reason={sig.reason}")

    # 6. Print Results
    logging.info("Backtest finished.")
    all_strategies["Trend"].print_summary(broker.trades)
    all_strategies["MeanRevert"].print_summary(broker.trades)
    broker.print_summary()

    long_trades = [t for t in broker.trades if t.get('side') == 1]
    short_trades = [t for t in broker.trades if t.get('side') == -1]
    logging.info(f"PnL Long Trades: {sum(t['pnl'] for t in long_trades):.2f}")
    logging.info(f"PnL Short Trades: {sum(t['pnl'] for t in short_trades):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-Driven Trading Bot Backtester")
    parser.add_argument("--config", type=str, default="config.toml", help="Path to the configuration file")
    parser.add_argument("--limit-bars", type=int, default=2000, help="Number of historical bars to load")
    args = parser.parse_args()
    main(config_path=args.config, limit_bars=args.limit_bars)