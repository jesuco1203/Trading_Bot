from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class RiskManager:
    max_dd_pct_session: float = 5.0
    max_consecutive_losses: int = 5
    max_daily_trades: int = 30
    cooldown_after_loss_bars: int = 0
    # Espera tras CUALQUIER cierre (gane o pierda), no sólo tras pérdida.
    # Motivo: al quedar plano el bot toma la siguiente señal disponible, y las
    # entradas inmediatas rinden mucho peor que las que llegan tras una pausa.
    cooldown_after_close_bars: int = 0

    # internal state
    _hard_stopped: bool = False
    _consecutive_losses: int = 0
    _cooldown_until_bar: Optional[int] = None
    # Deliberadamente separado de _cooldown_until_bar: ese lo borra
    # _reset_daily_state en cada día nuevo, lo que en 4h cancelaría el cooldown
    # a media vigencia.
    _close_cooldown_until_bar: Optional[int] = None
    _risk_halt: bool = False
    _halt_reason: Optional[str] = None
    _halt_announced: bool = False
    _trades_today: int = 0
    _current_day: Optional[str] = None
    _session_start_equity: Optional[float] = None
    _session_peak_equity: Optional[float] = None

    def check_limits(self, equity: float, trades: list, current_dt: Any) -> None:
        """Placeholder for future session-level risk checks."""
        pass

    def can_open_new_trade(self, idx: int) -> bool:
        if self._risk_halt:
            if not self._halt_announced and self._halt_reason:
                print(f"[RISK_STOP] reason={self._halt_reason}")
                self._halt_announced = True
            return False

        if (
            self.max_daily_trades > 0
            and self._trades_today >= self.max_daily_trades
        ):
            self._trigger_halt("max_daily_trades")
            if not self._halt_announced:
                print("[RISK_STOP] reason=max_daily_trades")
                self._halt_announced = True
            return False

        if self._cooldown_until_bar is not None and idx < self._cooldown_until_bar:
            self._blocked_trades = getattr(self, "_blocked_trades", 0) + 1
            return False

        if (
            self._close_cooldown_until_bar is not None
            and idx < self._close_cooldown_until_bar
        ):
            self._blocked_trades = getattr(self, "_blocked_trades", 0) + 1
            self._blocked_by_close_cd = getattr(self, "_blocked_by_close_cd", 0) + 1
            return False
        return True

    def on_trade_close(
        self,
        idx: int,
        pnl_r_multiple: float,
        pnl: Optional[float] = None,
        equity: Optional[float] = None,
        ts: Optional[Any] = None,
    ) -> None:
        day_key = self._extract_day(ts)
        if day_key is not None and day_key != self._current_day:
            self._reset_daily_state(equity, day_key)

        self._trades_today += 1

        if pnl_r_multiple < 0 and self.cooldown_after_loss_bars > 0:
            self._cooldown_until_bar = idx + self.cooldown_after_loss_bars + 1
            self._losses_seen = getattr(self, "_losses_seen", 0) + 1

        if self.cooldown_after_close_bars > 0:
            self._close_cooldown_until_bar = idx + self.cooldown_after_close_bars + 1

        if pnl_r_multiple < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        if equity is not None:
            if self._session_start_equity is None:
                self._session_start_equity = equity
            if self._session_peak_equity is None:
                self._session_peak_equity = equity
            else:
                self._session_peak_equity = max(self._session_peak_equity, equity)

            if self.max_dd_pct_session > 0 and self._session_peak_equity:
                drawdown = self._session_peak_equity - equity
                drawdown_pct = (
                    0.0
                    if self._session_peak_equity <= 0
                    else (drawdown / self._session_peak_equity) * 100.0
                )
                if drawdown_pct >= self.max_dd_pct_session:
                    self._trigger_halt("max_dd_pct_session")

        if (
            self.max_consecutive_losses > 0
            and self._consecutive_losses >= self.max_consecutive_losses
        ):
            self._trigger_halt("max_consecutive_losses")

    def is_hard_stopped(self) -> bool:
        """Returns True if a hard stop has been triggered."""
        return self._hard_stopped or self._risk_halt

    def debug_summary(self) -> None:
        print(
            f"[RM_DEBUG] cooldown_after_loss_bars={self.cooldown_after_loss_bars}, "
            f"cooldown_until={self._cooldown_until_bar}, "
            f"blocked={getattr(self, '_blocked_trades', 0)}, "
            f"losses={getattr(self, '_losses_seen', 0)}, "
            f"risk_halt={self._risk_halt}, "
            f"reason={self._halt_reason}"
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _trigger_halt(self, reason: str) -> None:
        if self._risk_halt and self._halt_reason == reason:
            return
        self._risk_halt = True
        self._halt_reason = reason
        self._halt_announced = False
        print(f"[RISK_STOP] reason={reason}")

    def _extract_day(self, ts: Any) -> Optional[str]:
        if ts is None:
            return None
        if hasattr(ts, "date"):
            try:
                return str(ts.date())
            except Exception:
                pass
        if isinstance(ts, str):
            return ts[:10]
        return None

    def _reset_daily_state(self, equity: Optional[float], day_key: str) -> None:
        self._current_day = day_key
        self._trades_today = 0
        self._consecutive_losses = 0
        self._risk_halt = False
        self._halt_reason = None
        self._halt_announced = False
        self._cooldown_until_bar = None
        if equity is not None:
            self._session_start_equity = equity
            self._session_peak_equity = equity
