
import numpy as np

def regime_bias(ema_fast, ema_slow):
    slope = ema_slow.diff(3)
    bias = np.where((ema_fast > ema_slow) & (slope > 0), 1, 0)
    bias = np.where((ema_fast < ema_slow) & (slope < 0), -1, bias)
    return bias
