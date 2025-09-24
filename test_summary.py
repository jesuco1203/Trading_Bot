import sys
sys.path.append('.')

from strategies.base import BaseStrategy
from datetime import datetime, timedelta

# Create a dummy BaseStrategy instance
strategy = BaseStrategy(name="MeanRevert")

# Create synthetic trades data
now = datetime.now()
fake_trades = [
    # Entry 1
    {"ts": now, "entry_ts": now, "strategy": "MeanRevert", "partial": False, "pnl": 0.0, "rr": 1.5, "event": "entry"},
    {"ts": now + timedelta(minutes=10), "entry_ts": now, "strategy": "MeanRevert", "partial": True, "pnl": 50.0, "event": "partial"},
    {"ts": now + timedelta(minutes=20), "entry_ts": now, "strategy": "MeanRevert", "partial": False, "pnl": 100.0, "exit_reason": "tp", "event": "close"},

    # Entry 2 (Loss)
    {"ts": now + timedelta(minutes=30), "entry_ts": now + timedelta(minutes=30), "strategy": "MeanRevert", "partial": False, "pnl": 0.0, "rr": 2.0, "event": "entry"},
    {"ts": now + timedelta(minutes=40), "entry_ts": now + timedelta(minutes=30), "strategy": "MeanRevert", "partial": False, "pnl": -75.0, "exit_reason": "sl", "event": "close"},

    # Entry 3 (Partial win, then loss)
    {"ts": now + timedelta(minutes=50), "entry_ts": now + timedelta(minutes=50), "strategy": "MeanRevert", "partial": False, "pnl": 0.0, "rr": 1.0, "event": "entry"},
    {"ts": now + timedelta(minutes=60), "entry_ts": now + timedelta(minutes=50), "strategy": "MeanRevert", "partial": True, "pnl": 20.0, "event": "partial"},
    {"ts": now + timedelta(minutes=70), "entry_ts": now + timedelta(minutes=50), "strategy": "MeanRevert", "partial": False, "pnl": -30.0, "exit_reason": "sl", "event": "close"},

    # Entry 4 (No close yet)
    {"ts": now + timedelta(minutes=80), "entry_ts": now + timedelta(minutes=80), "strategy": "MeanRevert", "partial": False, "pnl": 0.0, "rr": 1.8, "event": "entry"},
]

# Test the print_summary method
p, r, e, w, h = strategy.print_summary(fake_trades)

# Assertions
assert isinstance(p, float), f"PnL is not float: {type(p)}"
assert isinstance(r, float), f"RR Avg is not float: {type(r)}"
assert isinstance(e, int), f"Entries is not int: {type(e)}"
assert isinstance(w, int), f"Wins is not int: {type(w)}"
assert isinstance(h, float), f"Hit Rate is not float: {type(h)}"

print("All assertions passed for print_summary!")
print(f"PnL: {p}, RR Avg: {r}, Entries: {e}, Wins: {w}, Hit Rate: {h}")
