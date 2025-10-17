import logging
import math

def manage_exit(state, side, entry_px, sl_px, atr_now, bar_count_since_entry, high, low,
                sl_atr, tp_r_primary, tp_primary_ratio, tp_final_r,
                be_trigger_atr, trail_atr_mult, time_stop_bars, trail_activate_r, bar_rr):

    logger = logging.getLogger()
    old_sl = sl_px
    new_sl = old_sl

    # --- Guards ---
    unrealized_R = bar_rr
    partial_taken = state.partial_done
    cfg_trail_activate_r = state.trail_activate_r

    allow_BE = partial_taken or (unrealized_R >= cfg_trail_activate_r)
    allow_trail = (unrealized_R >= cfg_trail_activate_r)

    # --- Logic ---
    if allow_trail:
        # Trailing stop logic (Chandelier Exit)
        try:
            chand_high = high.rolling(20).max().iloc[-1]
            chand_low = low.rolling(20).min().iloc[-1]
            if side == 1 and not math.isnan(chand_high):
                trail_price = float(chand_high) - trail_atr_mult * atr_now
                new_sl = max(old_sl, trail_price)
            elif side == -1 and not math.isnan(chand_low):
                trail_price = float(chand_low) + trail_atr_mult * atr_now
                new_sl = min(old_sl, trail_price)
        except IndexError:
            pass # Not enough data for rolling window

    elif allow_BE and not state.be_set:
        # Break-even logic (only if not already trailing)
        eps = state.partial_be_eps_atr * state.entry_atr if state.entry_atr else 0.0
        be_price = entry_px + side * eps
        if side == 1:
            new_sl = max(old_sl, be_price)
        else:
            new_sl = min(old_sl, be_price)
        state.be_set = True

    # --- Logging ---
    if new_sl != old_sl:
        logger.info(
            f"[SL MOVE] i={bar_count_since_entry} from={old_sl:.2f} to={new_sl:.2f} "
            f"R={unrealized_R:.2f} partial_taken={partial_taken} "
            f"trail_on={allow_trail}"
        )

    # --- Time Stop ---
    if time_stop_bars and bar_count_since_entry >= time_stop_bars and unrealized_R < 1.0:
        state.force_exit = True

    tp_px = state.tp
    return new_sl, tp_px, state