import pandas as pd
import pandas_ta as ta
import numpy as np
import logging

class TradingStrategy:
    """
    Encapsula la lógica de la estrategia de trading.
    Puede operar en modo 'CONSERVADORA', 'AGRESIVA' o 'SIMPLE'.
    Ahora también puede operar en modo ADAPTATIVO para ajustarse al régimen de mercado.
    """
    def __init__(self, config: dict):
        self.params = config.get('strategy_params', {})
        self.trading_settings = config.get('trading_settings', {})
        self.adaptive_settings = config.get('adaptive_strategy_settings', {'enabled': False})
        
        self.mode = self.trading_settings.get('strategy_mode', 'CONSERVADORA')
        
        log_message = f"Estrategia inicializada en modo: {self.mode}."
        if self.adaptive_settings.get('enabled', False):
            log_message += " El modo adaptativo está HABILITADO."
        else:
            log_message += " El modo adaptativo está DESHABILITADO."
        logging.info(log_message)

    def analyze(self, df: pd.DataFrame) -> str:
        """
        Analiza los datos del mercado y devuelve una señal ('BUY', 'SELL', 'HOLD').
        """
        self._calculate_indicators(df)
        
        min_periods = max(
            self.params.get('support_resistance_period', 50),
            self.adaptive_settings.get('trend_sma_period', 200)
        )
        if len(df) < min_periods + 3:
            return 'HOLD'

        last_candle = df.iloc[-2]
        prev_candle = df.iloc[-3]
        
        if 'ATR_avg' in df.columns and 'ATR' in last_candle:
            avg_atr = df['ATR_avg'].iloc[-2]
            volatility_threshold = avg_atr * self.params.get('volatility_filter_multiplier', 2.5)
            if last_candle['ATR'] > volatility_threshold:
                return 'HOLD'

        if self.adaptive_settings.get('enabled', False):
            market_regime = self._get_market_regime(df)
            logging.info(f"Régimen de mercado detectado: {market_regime}")
            
            base_signal = self._get_base_signal(df, last_candle, prev_candle)

            if market_regime == 'BULLISH' and base_signal == 'BUY':
                return 'BUY'
            elif market_regime == 'BEARISH' and base_signal == 'SELL':
                return 'SELL'
            elif market_regime == 'RANGING' and base_signal != 'HOLD':
                return base_signal
            else:
                return 'HOLD'
        else:
            return self._get_base_signal(df, last_candle, prev_candle)

    def _get_base_signal(self, df, last_candle, prev_candle) -> str:
        if self.mode == 'AGRESIVA':
            return self._analyze_aggressive(df, last_candle, prev_candle)
        elif self.mode == 'SIMPLE':
            return self._analyze_simple(df, last_candle, prev_candle)
        else: # CONSERVADORA
            return self._analyze_conservative(df, last_candle, prev_candle)

    def _get_market_regime(self, df: pd.DataFrame) -> str:
        trend_sma_col = f"SMA_{self.adaptive_settings['trend_sma_period']}"
        
        if trend_sma_col not in df.columns:
            return 'RANGING'

        last_close = df['close'].iloc[-1]
        trend_sma = df[trend_sma_col].iloc[-1]
        
        if last_close > trend_sma:
            return 'BULLISH'
        elif last_close < trend_sma:
            return 'BEARISH'
        else:
            return 'RANGING'

    def _analyze_conservative(self, df, last_candle, prev_candle):
        if (self._is_bullish_engulfing(last_candle, prev_candle) and
            last_candle['RSI'] < self.params['rsi_oversold'] and
            last_candle['volume'] > last_candle['volume_sma'] * self.params['volume_factor'] and
            self._is_near_support(last_candle, df.iloc[-self.params['support_resistance_period']:-2]) and
            last_candle['close'] > last_candle['SMA']):
            return 'BUY'

        if (self._is_bearish_engulfing(last_candle, prev_candle) and
            last_candle['RSI'] > self.params['rsi_overbought'] and
            last_candle['volume'] > last_candle['volume_sma'] * self.params['volume_factor'] and
            self._is_near_resistance(last_candle, df.iloc[-self.params['support_resistance_period']:-2]) and
            last_candle['close'] < last_candle['SMA']):
            return 'SELL'
            
        return 'HOLD'

    def _analyze_aggressive(self, df, last_candle, prev_candle):
        if (self._is_bullish_engulfing(last_candle, prev_candle) and
            last_candle['volume'] > last_candle['volume_sma'] * self.params['volume_factor'] and
            last_candle['close'] > last_candle['SMA']):
            return 'BUY'

        if (self._is_bearish_engulfing(last_candle, prev_candle) and
            last_candle['volume'] > last_candle['volume_sma'] * self.params['volume_factor'] and
            last_candle['close'] < last_candle['SMA']):
            return 'SELL'
            
        return 'HOLD'
        
    def _analyze_simple(self, df, last_candle, prev_candle):
        if (self._is_bullish_engulfing(last_candle, prev_candle) and
            last_candle['close'] > last_candle['SMA']):
            return 'BUY'

        if (self._is_bearish_engulfing(last_candle, prev_candle) and
            last_candle['close'] < last_candle['SMA']):
            return 'SELL'
            
        return 'HOLD'

    def _calculate_indicators(self, df: pd.DataFrame):
        df['SMA'] = ta.sma(df['close'], length=self.params['sma_period'])
        df['RSI'] = ta.rsi(df['close'], length=self.params['rsi_period'])
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=self.params['atr_period'])
        df['volume_sma'] = ta.sma(df['volume'], length=self.params['volume_period'])
        df['support'] = df['low'].rolling(window=self.params['support_resistance_period']).min()
        df['resistance'] = df['high'].rolling(window=self.params['support_resistance_period']).max()
        df['ATR_avg'] = ta.sma(df['ATR'], length=self.params['sma_period'])
        
        if self.adaptive_settings.get('enabled', False):
            trend_sma_period = self.adaptive_settings.get('trend_sma_period', 200)
            df[f'SMA_{trend_sma_period}'] = ta.sma(df['close'], length=trend_sma_period)

        df.dropna(inplace=True)

    def _is_bullish_engulfing(self, current, previous) -> bool:
        return (current['close'] > current['open'] and previous['close'] < previous['open'] and current['close'] > previous['open'] and current['open'] < previous['close'])

    def _is_bearish_engulfing(self, current, previous) -> bool:
        return (current['close'] < current['open'] and previous['close'] > previous['open'] and current['close'] < previous['open'] and current['open'] > previous['close'])

    def _is_near_support(self, candle, window_df) -> bool:
        support_level = window_df['low'].min()
        return candle['low'] <= support_level * 1.005

    def _is_near_resistance(self, candle, window_df) -> bool:
        resistance_level = window_df['high'].max()
        return candle['high'] >= resistance_level * 0.995

    def get_trade_params(self, df: pd.DataFrame, side: str, entry_price: float) -> dict:
        stop_loss = 0
        sl_mode = self.params.get('stop_loss_mode', 'atr')
        max_sl_pct = self.params.get('max_stop_loss_percentage', 10.0) / 100

        if sl_mode == 'structural':
            lookback = self.params.get('structural_stop_lookback', 20)
            window = df.iloc[-lookback:-1] 

            if side == 'BUY':
                structural_sl = window['low'].min()
                stop_loss = min(entry_price * (1 - max_sl_pct), structural_sl)
            elif side == 'SELL':
                structural_sl = window['high'].max()
                stop_loss = max(entry_price * (1 + max_sl_pct), structural_sl)
        
        else: # Fallback to ATR
            last_atr = df.iloc[-2]['ATR']
            if pd.isna(last_atr) or last_atr == 0:
                last_atr = entry_price * 0.01
            
            sl_distance_atr = last_atr * self.params.get('atr_multiplier_sl', 1.5)
            if side == 'BUY':
                stop_loss = entry_price - sl_distance_atr
            elif side == 'SELL':
                stop_loss = entry_price + sl_distance_atr

        if side == 'BUY':
            max_sl_price = entry_price * (1 - max_sl_pct)
            stop_loss = max(stop_loss, max_sl_price) 
        elif side == 'SELL':
            max_sl_price = entry_price * (1 + max_sl_pct)
            stop_loss = min(stop_loss, max_sl_price)

        sl_distance = abs(entry_price - stop_loss)
        tp_distance = sl_distance * self.params.get('risk_reward_ratio', 2.0)

        if side == 'BUY':
            take_profit = entry_price + tp_distance
        elif side == 'SELL':
            take_profit = entry_price - tp_distance
        else:
            return {}

        return {'stop_loss': stop_loss, 'take_profit': take_profit}