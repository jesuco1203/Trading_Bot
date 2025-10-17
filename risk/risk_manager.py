from dataclasses import dataclass

@dataclass
class RiskManager:
    max_dd_pct_session: float = 5.0
    max_consecutive_losses: int = 3
    max_daily_trades: int = 20
    
    _hard_stopped: bool = False
    _consecutive_losses: int = 0

    def check_limits(self, equity: float, trades: list, current_dt: any):
        """
        Checks session-level risk limits.
        If a limit is breached, it sets the internal hard-stop flag.
        (Logic to be fully implemented later)
        """
        # Placeholder for max drawdown check
        # if new_drawdown > self.max_dd_pct_session:
        #     self._hard_stopped = True
        #     logging.warning(f"RISK_MANAGER: Hard stop triggered due to max session drawdown.")

        # Placeholder for consecutive losses check
        # if new_consecutive_losses > self.max_consecutive_losses:
        #     self._hard_stopped = True
        #     logging.warning(f"RISK_MANAGER: Hard stop triggered due to max consecutive losses.")
        
        pass

    def is_hard_stopped(self) -> bool:
        """Returns True if a hard stop has been triggered."""
        return self._hard_stopped

