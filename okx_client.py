import ccxt
import pandas as pd
import logging

class OKXClient:
    """
    Cliente para interactuar con la API de OKX.
    Maneja la autenticación, obtención de datos y ejecución de órdenes.
    """
    def __init__(self, config: dict):
        self.config = config
        self.exchange = self._init_exchange()

    def _init_exchange(self):
        """Inicializa la instancia de ccxt con las credenciales."""
        try:
            exchange = ccxt.okx({
                'apiKey': self.config['okx_credentials']['apiKey'],
                'secret': self.config['okx_credentials']['secret'],
                'password': self.config['okx_credentials']['password'],
                'options': {
                    'defaultType': 'swap', # Usar SWAP para derivados (futuros/perpetuos)
                },
            })
            # Habilitar modo de prueba (sandbox) si está en la configuración
            if self.config['trading_settings']['paper_trading']:
                logging.info("Modo Paper Trading (Sandbox) activado.")
                exchange.set_sandbox_mode(True)
            return exchange
        except Exception as e:
            logging.error(f"Error al inicializar el cliente de OKX: {e}")
            raise

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        Obtiene datos OHLCV y los convierte a un DataFrame de pandas.
        """
        try:
            logging.info(f"Obteniendo datos OHLCV para {symbol} en timeframe {timeframe}...")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logging.error(f"Error al obtener datos OHLCV: {e}")
            return pd.DataFrame() # Devuelve un DataFrame vacío en caso de error

    def get_balance(self, currency: str = 'USDT') -> float:
        """
        Obtiene el saldo disponible para una moneda específica.
        """
        try:
            balance = self.exchange.fetch_balance()
            # La estructura puede variar, busca en 'free', 'total', o la específica del activo.
            # Para derivados, el balance relevante suele ser el de margen en USDT.
            if currency in balance['total']:
                return balance[currency]['free'] or 0.0
            return 0.0
        except Exception as e:
            logging.error(f"Error al obtener el saldo: {e}")
            return 0.0

    def place_order(self, symbol: str, side: str, amount: float, price: float, params: dict):
        """
        Coloca una orden de mercado con Stop Loss y Take Profit.
        """
        order_side = 'buy' if side == 'BUY' else 'sell'
        try:
            logging.info(f"Intentando colocar orden {side} para {symbol}...")
            logging.info(f"Cantidad: {amount}, Precio Entrada Aprox: {price}, SL: {params['stop_loss']}, TP: {params['take_profit']}")

            if self.config['trading_settings']['paper_trading']:
                logging.warning("MODO SIMULACIÓN: La orden no se ejecutará en el mercado real.")
                # Simula una respuesta exitosa
                return {
                    'info': {'ordId': 'simulated_12345'},
                    'status': 'open'
                }

            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=order_side,
                amount=amount,
                params={
                    'tdMode': 'isolated', # o 'cross'
                    'slPrice': self.exchange.price_to_precision(symbol, params['stop_loss']),
                    'tpPrice': self.exchange.price_to_precision(symbol, params['take_profit'])
                }
            )
            logging.info(f"Orden colocada exitosamente: {order['info']['ordId']}")
            return order
        except Exception as e:
            logging.error(f"Error al colocar la orden: {e}")
            return None