import pandas as pd
import json
import logging
import time
import os
import numpy as np
from strategy import TradingStrategy
from okx_client import OKXClient
from utils import setup_logger, validate_config

def fetch_all_historical_data(client, symbol, timeframe, since, max_candles=15000):
    all_ohlcv = []
    limit = 500
    
    while len(all_ohlcv) < max_candles:
        try:
            logging.info(f"Obteniendo lote de datos desde {pd.to_datetime(since, unit='ms', utc=True)}...")
            ohlcv = client.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                logging.info("No hay más datos históricos disponibles.")
                break
            
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            time.sleep(client.exchange.rateLimit / 1000)

        except Exception as e:
            logging.error(f"Error durante la obtención de datos paginados: {e}")
            break
            
    return all_ohlcv

def simulate_trade_outcome(entry_index, df, trade_params, signal, entry_price):
    stop_loss, take_profit, max_favorable_price = trade_params['stop_loss'], trade_params['take_profit'], entry_price
    for i in range(entry_index + 1, len(df)):
        candle = df.iloc[i]
        if signal == 'BUY':
            if candle['high'] > max_favorable_price: max_favorable_price = candle['high']
            if candle['low'] <= stop_loss: return 'LOSS', stop_loss, candle.name, stop_loss - entry_price, max_favorable_price
            if candle['high'] >= take_profit: return 'WIN', take_profit, candle.name, take_profit - entry_price, max_favorable_price
        elif signal == 'SELL':
            if candle['low'] < max_favorable_price: max_favorable_price = candle['low']
            if candle['high'] >= stop_loss: return 'LOSS', stop_loss, candle.name, entry_price - stop_loss, max_favorable_price
            if candle['low'] <= take_profit: return 'WIN', take_profit, candle.name, entry_price - take_profit, max_favorable_price
    return 'OPEN', None, None, 0, max_favorable_price

def run_backtest(config, start_date_str):
    logging.info(f"Iniciando backtest para {config['trading_settings']['symbol']} desde {start_date_str}")
    
    client = OKXClient(config)
    strategy = TradingStrategy(config)
    symbol, timeframe = config['trading_settings']['symbol'], config['trading_settings']['timeframe']
    
    cache_filename = f"{symbol.replace('/', '_')}_{timeframe}_data.csv"
    requested_start_date = pd.Timestamp(start_date_str, tz='UTC')
    
    # --- LÓGICA DE CACHÉ CORREGIDA ---
    df = pd.DataFrame()
    if os.path.exists(cache_filename):
        logging.info(f"Archivo de caché '{cache_filename}' encontrado.")
        df = pd.read_csv(cache_filename, index_col='timestamp', parse_dates=True)
        if df.index.tz is None: df.index = df.index.tz_localize('UTC')

    # Si no hay caché, o si el caché no contiene la fecha solicitada, descargamos.
    if df.empty or requested_start_date < df.index[0] or requested_start_date > df.index[-1]:
        logging.info("El caché es inválido o no cubre el rango solicitado. Se descargarán nuevos datos.")
        since = int(requested_start_date.timestamp() * 1000)
        all_ohlcv = fetch_all_historical_data(client, symbol, timeframe, since)
        if not all_ohlcv:
            logging.error("No se pudieron obtener datos. Finalizando."); return

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC')
        df.set_index('timestamp', inplace=True)
        logging.info(f"Guardando {len(df)} velas en caché: '{cache_filename}'")
        df.to_csv(cache_filename)
    else:
        logging.info("Caché válido. Usando datos locales.")

    analysis_df = df[df.index >= requested_start_date].copy()
    logging.info(f"Datos listos para el backtest: {len(analysis_df)} velas.")

    # --- SIMULACIÓN (sin cambios) ---
    results, start_index = [], config['strategy_params']['support_resistance_period'] + 3
    if len(analysis_df) <= start_index: logging.error("No hay suficientes datos para iniciar el backtest."); return

    i = start_index
    while i < len(analysis_df):
        current_df_slice = analysis_df.iloc[0:i+1].copy()
        signal = strategy.analyze(current_df_slice)
        if signal != 'HOLD':
            entry_price = current_df_slice['close'].iloc[-1]
            trade_params = strategy.get_trade_params(current_df_slice, signal, entry_price)
            outcome, exit_price, exit_date, pnl, max_favorable_price = simulate_trade_outcome(i, analysis_df, trade_params, signal, entry_price)
            if signal == 'BUY': max_potential_pnl = max_favorable_price - entry_price
            else: max_potential_pnl = entry_price - max_favorable_price
            results.append({'entry_date': current_df_slice.index[-1].strftime('%Y-%m-%d %H:%M'),'signal': signal, 'entry_price': entry_price,'stop_loss': trade_params['stop_loss'], 'take_profit': trade_params['take_profit'],'outcome': outcome, 'pnl': pnl, 'max_potential_pnl': max_potential_pnl})
            if exit_date:
                exit_idx_loc = analysis_df.index.get_loc(exit_date)
                if exit_idx_loc > i: i = exit_idx_loc; continue
        i += 1
    
    if not results: print(f"\n--- Backtest Finalizado sobre {len(analysis_df)} velas. No se generaron operaciones ---"); return
    
    results_df = pd.DataFrame(results)
    wins, losses = results_df[results_df['outcome'] == 'WIN'], results_df[results_df['outcome'] == 'LOSS']
    win_rate = (len(wins) / (len(wins) + len(losses)) * 100) if (len(wins) + len(losses)) > 0 else 0
    total_pnl, total_wins_pnl, total_losses_pnl = results_df['pnl'].sum(), wins['pnl'].sum(), abs(losses['pnl'].sum())
    profit_factor = total_wins_pnl / total_losses_pnl if total_losses_pnl > 0 else np.inf

    print("\n--- Resultados Detallados del Backtest ---"); print(results_df.to_string())
    print("\n--- 📊 Resumen de Rendimiento ---")
    print(f"Período Analizado: {analysis_df.index[0].strftime('%Y-%m-%d')} a {analysis_df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Total de Operaciones: {len(wins) + len(losses)}"); print(f"Ganadoras: {len(wins)}"); print(f"Perdedoras: {len(losses)}")
    print(f"Tasa de Acierto (Win Rate): {win_rate:.2f}%"); print("---")
    print(f"Ganancia/Pérdida Neta (PnL): ${total_pnl:,.2f}"); print(f"Factor de Beneficio (Profit Factor): {profit_factor:.2f}")

if __name__ == '__main__':
    setup_logger()
    try:
        with open('config.json') as f: config = json.load(f)
        validate_config(config)
    except (FileNotFoundError, ValueError) as e: logging.error(f"Error de configuración: {e}"); exit()
    
    start_date = '2023-01-01' # Cambia esta fecha para tus pruebas
    run_backtest(config, start_date)