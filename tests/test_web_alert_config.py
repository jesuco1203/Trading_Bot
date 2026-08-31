import pytest

from monitoring.web_app import normalize_alert


def test_editing_live_alert_rearms_and_discards_stale_price():
    alert = normalize_alert({
        "symbol": "BTC-USDT-SWAP", "mode": "live", "timeframe": "4h",
        "price": "100000", "live_last_price": 99999, "live_armed": False,
    })

    assert alert["price"] == 100000.0
    assert alert["live_armed"] is True
    assert "live_last_price" not in alert
    assert "direction" not in alert


def test_close_alert_requires_direction():
    with pytest.raises(ValueError):
        normalize_alert({"symbol": "BTC-USDT-SWAP", "mode": "close", "timeframe": "4h", "price": 1})


def test_one_minute_close_alert_is_valid_for_testing():
    alert = normalize_alert({
        "symbol": "BTC-USDT-SWAP", "mode": "close", "timeframe": "1m",
        "price": 1, "direction": "above",
    })

    assert alert["timeframe"] == "1m"
