import logging
import sys
import time
import requests
from functools import wraps
import numpy as np

def setup_logger():
    """Configura el logger para que escriba en un archivo y en la consola."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("trading_bot.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def retry_with_backoff(max_retries=3, initial_delay=5, backoff_factor=2):
    """
    Decorador que reintenta una función con un tiempo de espera exponencial
    si lanza una excepción.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        logging.error(f"Error final en {func.__name__} tras {max_retries} intentos: {e}")
                        raise
                    logging.warning(f"Error en {func.__name__}: {e}. Reintentando en {delay}s...")
                    time.sleep(delay)
                    delay *= backoff_factor
            return None
        return wrapper
    return decorator

def validate_config(config):
    """
    Valida que el archivo de configuración contenga todas las claves necesarias.
    Lanza ValueError si falta alguna clave o los valores son incorrectos.
    """
    required_keys = {
        'okx_credentials': ['apiKey', 'secret', 'password'],
        'telegram': ['bot_token', 'chat_id'],
        'trading_settings': ['symbol', 'timeframe', 'position_risk_percentage', 'paper_trading'],
        'strategy_params': [
            'sma_period', 'rsi_period', 'atr_period', 'volume_period',
            'support_resistance_period', 'rsi_oversold', 'rsi_overbought',
            'volume_factor', 'atr_multiplier_sl', 'risk_reward_ratio'
        ]
    }
    
    for section, keys in required_keys.items():
        if section not in config:
            raise ValueError(f"Falta la sección '{section}' en la configuración")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Falta la clave '{key}' en la sección '{section}'")

    if not (0 < config['trading_settings']['position_risk_percentage'] <= 100):
        raise ValueError("'position_risk_percentage' debe estar entre 0 y 100")
    logging.info("Configuración validada exitosamente.")

def send_telegram_notification(message, config):
    """Envía una notificación a través de Telegram."""
    bot_token = config['telegram']['bot_token']
    chat_id = config['telegram']['chat_id']

    if not bot_token or not chat_id or bot_token == "TU_TELEGRAM_BOT_TOKEN_AQUI":
        logging.warning("Credenciales de Telegram no configuradas. Omitiendo notificación.")
        return

    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"Error al enviar notificación de Telegram: {e}")

