# Bot de Trading AI para OKX

Este es un bot de trading algorítmico avanzado en Python que utiliza un enfoque basado en Inteligencia Artificial para operar en el exchange OKX. El sistema identifica dinámicamente el régimen de mercado (tendencia, rango, alta volatilidad) mediante un **Modelo Oculto de Markov (HMM)** y activa las estrategias de trading más adecuadas para las condiciones actuales.

**🚨 ADVERTENCIA: USAR BAJO SU PROPIO RIESGO 🚨**
El trading de criptomonedas es altamente riesgoso. Este software se proporciona "tal cual", sin garantías. El autor no se hace responsable de ninguna pérdida financiera. Se recomienda encarecidamente probar exhaustivamente en modo de backtesting y paper trading antes de arriesgar capital real.

## Arquitectura y Características

-   **Selección de Estrategias por IA**: Utiliza un Modelo Oculto de Markov (HMM) para analizar características del mercado (`ret`, `vol`, `rng_pct`) y determinar el régimen actual.
-   **Multi-Estrategia Dinámica**: Activa y desactiva estrategias automáticamente según el régimen detectado por el HMM.
    -   **`Trend`**: Estrategia de seguimiento de tendencia basada en cruces de medias móviles y filtros de ADX.
    -   **`MeanRevert`**: Estrategia de reversión a la media que opera en mercados de rango.
    -   **`VolBreakout`**: (Placeholder) Estrategia para operar en rupturas de volatilidad.
-   **Configuración por Perfiles**: Gestiona todos los parámetros a través de archivos de configuración `.toml` dedicados para cada par/timeframe (ej. `configs/btc_30m.toml`), permitiendo una experimentación y ajuste flexibles.
-   **Backtesting Robusto**: El script `main.py` funciona como un motor de backtesting que permite la validación de estrategias sobre datos históricos.
-   **Gestión de Riesgo Avanzada**:
    -   Cálculo de tamaño de posición basado en ATR.
    -   Ajuste de riesgo dinámico según la confianza del modelo HMM y la fuerza de la señal.
    -   Soporte para Stop-Loss, Take-Profit, salidas parciales y trailing stops.
-   **Simulación de Ejecución (Paper Broker)**: Un broker simulado (`execution/paper.py`) que modela comisiones, deslizamiento (slippage) y exporta un registro detallado de todas las operaciones a un archivo CSV.
-   **Adquisición de Datos**: Incluye un script (`scripts/okx_backfill.py`) para descargar y almacenar datos históricos de OKX en formato Parquet, optimizado para lecturas rápidas.
-   **Estructura Modular**: El código está organizado en módulos claros y cohesivos (`data`, `features`, `regime`, `strategies`, `selector`, `risk`, `execution`, `monitoring`).

## Notas de backtesting (nov 2024)
-   TrendV2 AB (windows 2000-4000 y 4000-6000) confirma que `partial_take_r=1.0` supera a 0.85R; artefactos: `trades_BTC-USDT-SWAP_30m_window_{2000-4000,4000-6000}_{1760678994,1760678999,1760679005,1760679010}.csv`.

## Requisitos Previos

1.  **Python 3.8 o superior.**
2.  Un entorno virtual de Python (recomendado).
3.  Una cuenta en **OKX** (para trading real o para obtener datos).

## Instalación

1.  **Clona este repositorio:**
    ```bash
    git clone <url-del-repositorio>
    cd Trading_Bot
    ```

2.  **Crea y activa un entorno virtual:**
    ```bash
    python -m venv venv
    # En macOS/Linux
    source venv/bin/activate
    # En Windows
    venv\Scripts\activate
    ```

3.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuración

El sistema ya no usa un único `config.json`. Toda la configuración se gestiona a través de perfiles `.toml` en el directorio `configs/`.

1.  **Datos Históricos**: Antes de ejecutar un backtest, necesitas descargar los datos. Usa el script `okx_backfill.py`.
    ```bash
    # Ejemplo para descargar 12 meses de datos para BTC y ETH en varios timeframes
    python scripts/okx_backfill.py --root data/okx --symbols "BTC-USDT-SWAP,ETH-USDT-SWAP" --months 12
    ```

2.  **Perfil de Backtesting**: Abre y modifica un perfil existente en `configs/`, por ejemplo, `configs/btc_30m.toml`.
    -   **`[market]`**: Define el símbolo (`symbols`) y el timeframe (`timeframes`) para el backtest.
    -   **`[risk]`**: Ajusta el capital inicial (`starting_equity`) y el riesgo por operación (`risk_per_trade_pct`).
    -   **`[selector]`**: Configura los umbrales de probabilidad (`enter_th`, `exit_th`) para que el HMM active/desactive estrategias.
    -   **`[strategy_mapping]`**: **Aquí decides qué estrategia se ejecuta en cada régimen de mercado**. Para activar una estrategia, añade su nombre (ej. `"Trend"`, `"MeanRevert"`) a la lista del régimen correspondiente (`trend`, `mr`, `high_vol`). Para desactivarla, deja la lista vacía.
        ```toml
        [strategy_mapping]
        trend = ["Trend"]  # <-- La estrategia Trend se activará en régimen de tendencia
        mr    = []         # <-- La estrategia MeanRevert está desactivada
        high_vol = []
        ```
    -   **`[strategy_params]`**: Ajusta los parámetros específicos de cada estrategia. Cada parámetro está prefijado con el nombre de la estrategia (ej. `Trend_sl_mult_atr`).

## Ejecución de Backtesting

Para ejecutar un backtest, utiliza `main.py` y especifica el archivo de configuración del perfil que deseas probar.

```bash
# Ejemplo de ejecución de un backtest para el perfil de BTC en 30m
python main.py --config configs/btc_30m.toml --limit-bars 2000
```

-   `--config`: Ruta al archivo de configuración del perfil (obligatorio).
-   `--limit-bars`: Número de velas (barras) históricas que se usarán en el backtest.

El script ejecutará el backtest, mostrando logs de la actividad del selector de régimen y las estrategias. Al finalizar, imprimirá un resumen de rendimiento y exportará un archivo `trades_*.csv` con el detalle de todas las operaciones simuladas.

## Archivos Desactualizados

Los siguientes archivos corresponden a una versión anterior del bot y ya no son utilizados por el flujo principal de `main.py`. Se conservan como referencia o para posibles usos futuros:
- `config.json`
- `strategy.py`
- `backtest.py`
- `okx_client.py` (el cliente CCXT en la raíz)

Legacy moved to archive/, la fuente de verdad es main.py, configs/*.toml, strategies/, execution/, features/.
