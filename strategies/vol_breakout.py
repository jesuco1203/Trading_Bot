from __future__ import annotations
from strategies.base import BaseStrategy, Signal
from typing import Dict, Any
import pandas as pd

class VolBreakout(BaseStrategy):
    def __init__(self):
        super().__init__("VolBreakout")

    def signal(self, ctx: Dict[str, Any]) -> Signal | None:
        # Conservative implementation: return flat until properly defined
        return Signal(side="flat", strength=0.0, sl_pts=None, tp_pts=None, reason="disabled")

    def warmup_bars(self) -> int:
        return 20 # Needs at least 20 bars for range calculation
