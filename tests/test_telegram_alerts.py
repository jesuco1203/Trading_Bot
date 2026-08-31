from datetime import timezone

from monitoring.telegram_alerts import (
    Candle,
    aggregate_hourly_candles,
    confirmed_candles,
    crosses_level,
    engulfing_direction,
    evaluate_candles,
    format_alert,
)


def candle(ts: int, open_: float, high: float, low: float, close: float) -> Candle:
    return Candle(ts, open_, high, low, close)


def test_bullish_engulfing_is_detected():
    previous = candle(1, 105, 107, 99, 100)
    current = candle(2, 99, 112, 98, 110)
    assert engulfing_direction(previous, current) == "bullish"


def test_bearish_engulfing_is_detected():
    previous = candle(1, 100, 107, 99, 105)
    current = candle(2, 106, 108, 94, 98)
    assert engulfing_direction(previous, current) == "bearish"


def test_level_requires_a_cross_not_just_being_above():
    previous = candle(1, 100, 105, 99, 104)
    current = candle(2, 104, 110, 103, 108)
    assert not crosses_level(previous, current, 100, "above")
    assert crosses_level(previous, current, 105, "above")


def test_evaluate_requires_both_engulfing_and_level_when_configured():
    previous = candle(1, 105, 107, 99, 100)
    current = candle(2, 99, 112, 98, 110)
    alert = evaluate_candles(
        "BTC-USDT-SWAP", [previous, current], level=105, level_direction="above"
    )
    assert alert is not None
    assert alert.pattern == "bullish"
    assert "envolvente alcista" in format_alert(alert)


def test_unconfirmed_okx_candle_is_ignored():
    payload = {
        "code": "0",
        "data": [
            ["2000", "2", "3", "1", "2.5", "0", "0", "0", "0"],
            ["1000", "1", "2", "0", "1.5", "0", "0", "0", "1"],
        ],
    }
    candles = confirmed_candles(payload)
    assert [item.timestamp_ms for item in candles] == [1000]


def test_three_hour_candle_is_built_from_three_complete_utc_hours():
    hour = 60 * 60 * 1000
    candles = [
        candle(0, 100, 105, 99, 101),
        candle(hour, 101, 108, 100, 107),
        candle(2 * hour, 107, 109, 104, 105),
    ]

    result = aggregate_hourly_candles(candles)

    assert result == [candle(0, 100, 109, 99, 105)]
