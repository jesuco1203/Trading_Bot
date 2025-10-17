
COOLDOWN_AFTER_LOSS_STREAK = 6
COOLDOWN_AFTER_KILL        = 24

def check_cooldown(metrics, session):
    if metrics.consecutive_losses() >= 2:
        session.cooldown_bars = COOLDOWN_AFTER_LOSS_STREAK

    if session.kill_switch:
        session.cooldown_bars = max(session.cooldown_bars, COOLDOWN_AFTER_KILL)

    if session.cooldown_bars > 0:
        session.cooldown_bars -= 1
        return True
    else:
        return False
