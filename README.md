# Bot de Trading para OKX con Estrategia de Vela Envolvente

Este es un bot de trading automatizado en Python que implementa una estrategia intradía basada en velas envolventes, RSI, volumen y otros filtros para operar en el exchange OKX.

**🚨 ADVERTENCIA: USAR BAJO SU PROPIO RIESGO 🚨**
El trading de criptomonedas es altamente riesgoso. Este software se proporciona "tal cual", sin garantías. El autor no se hace responsable de ninguna pérdida financiera. Se recomienda encarecidamente probar exhaustivamente en modo **Paper Trading** (`"paper_trading": true` en `config.json`) antes de arriesgar capital real.

## Características

-   **Estrategia Configurable**: Basada en velas envolventes, RSI, SMA, Volumen y Soportes/Resistencias.
-   **Gestión de Riesgo Integrada**: Stop-Loss basado en ATR y Take-Profit con ratio Riesgo/Recompensa.
-   **Cálculo de Tamaño de Posición**: Ajusta el tamaño de la operación basado en un porcentaje de riesgo del capital total.
-   **Conexión Segura a OKX**: Utiliza `ccxt` para una interacción robusta con la API.
-   **Modo de Simulación**: Incluye un modo de *paper trading* para probar la estrategia sin arriesgar fondos.
-   **Automatización**: Se sincroniza automáticamente con el cierre de las velas del timeframe configurado.
-   **Logging Detallado**: Registra todas las acciones, decisiones y errores en `trading_bot.log`.
-   **Estructura Modular**: El código está separado en módulos lógicos (cliente, estrategia, principal) para facilitar el mantenimiento y la expansión.

## Requisitos Previos

1.  **Python 3.8 o superior.**
2.  Una cuenta en **OKX**.
3.  **Credenciales API** de OKX (API Key, Secret Key, Passphrase).
    -   Al crear la API, asegúrate de darle permisos de **Trading**.
    -   Para el modo de prueba (Paper Trading), obtén las credenciales desde la sección "Demo Trading" de OKX.

## Instalación

1.  **Clona este repositorio:**
    ```bash
    git clone <url-del-repositorio>
    cd <nombre-del-repositorio>
    ```

2.  **Crea y activa un entorno virtual (recomendado):**
    ```bash
    python -m venv venv
    # En Windows
    venv\Scripts\activate
    # En macOS/Linux
    source venv/bin/activate
    ```

3.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuración

1.  **Renombra `config.example.json` a `config.json`** si es necesario.
2.  **Abre el archivo `config.json` y edita los siguientes campos:**

    -   `apiKey`, `secret`, `password`: Introduce tus credenciales API de OKX.
    -   `symbol`: El par a operar (ej. `BTC/USDT`). Asegúrate que sea un contrato de SWAP si operas derivados.
    -   `timeframe`: El marco de tiempo (ej. `30m`, `1h`, `4h`).
    -   `position_risk_percentage`: El porcentaje de tu capital a arriesgar por operación (ej. `1.0` para 1%).
    -   `paper_trading`:
        -   `true`: **Modo simulación**. El bot registrará las operaciones pero no enviará órdenes reales al mercado. **Usa esto para probar.**
        -   `false`: **Modo real**. El bot ejecutará operaciones con fondos reales.
    -   **`strategy_params`**: Ajusta los parámetros de los indicadores según tus preferencias.

## Ejecución

Una vez configurado, simplemente ejecuta el script principal desde tu terminal:

```bash
python main.py
```

El bot comenzará a funcionar. Verás mensajes en la consola y toda la actividad quedará registrada en el archivo `trading_bot.log`.

Para detener el bot de forma segura, presiona `Ctrl + C` en la terminal.

## Ejemplo de Log (`trading_bot.log`)

Un archivo de log típico se vería así:

```log
2025-08-20 11:30:00,123 - INFO - Iniciando el bot de trading...
2025-08-20 11:30:00,456 - INFO - Modo Paper Trading (Sandbox) activado.
2025-08-20 11:30:00,457 - INFO - Próxima ejecución en 1799.54 segundos a las 2025-08-20 12:00:00 UTC.
2025-08-20 12:00:05,111 - INFO - Comprobando nueva señal...
2025-08-20 12:00:05,112 - INFO - Obteniendo datos OHLCV para BTC/USDT en timeframe 30m...
2025-08-20 12:00:06,223 - INFO - Señal generada: HOLD
2025-08-20 12:00:06,224 - INFO - Próxima ejecución en 1799.89 segundos a las 2025-08-20 12:30:00 UTC.
...
2025-08-20 14:30:05,333 - INFO - Comprobando nueva señal...
2025-08-20 14:30:05,334 - INFO - Obteniendo datos OHLCV para BTC/USDT en timeframe 30m...
2025-08-20 14:30:06,444 - INFO - Señal generada: BUY
2025-08-20 14:30:06,555 - INFO - Calculando orden: Balance=1000.00 USDT, Riesgo=10.00 USDT, Tamaño Posición=0.0833 BTC/USDT
2025-08-20 14:30:06,666 - INFO - Intentando colocar orden BUY para BTC/USDT...
2025-08-20 14:30:06,667 - INFO - Cantidad: 0.0833, Precio Entrada Aprox: 71500.0, SL: 71380.0, TP: 71740.0
2025-08-20 14:30:06,668 - WARNING - MODO SIMULACIÓN: La orden no se ejecutará en el mercado real.
2025-08-20 14:30:06,669 - INFO - Posición abierta. El bot no buscará nuevas señales hasta ser reiniciado.
```

## Próximos Pasos y Mejoras Potenciales

-   **Detección de Divergencias RSI**: Implementar una función para detectar divergencias alcistas/bajistas, añadiendo un filtro más a la estrategia.
-   **Backtesting**: Crear un script separado (`backtest.py`) que utilice la misma clase `TradingStrategy` para evaluar su rendimiento con datos históricos.
-   **Gestión de Estado Avanzada**: En lugar de una simple variable `position_open`, consultar activamente a la API de OKX (`fetch_positions`) para saber si hay posiciones abiertas.
-   **Notificaciones**: Integrar un módulo para enviar alertas vía Telegram o email cuando se abre una operación o ocurre un error.