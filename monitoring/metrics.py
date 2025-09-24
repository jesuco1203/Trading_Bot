import numpy as np

class RollingMetrics:
    def __init__(self, window=1000):
        self.window = window
        self.equity = []

    def update(self, ts, equity, exposure):
        self.equity.append(float(equity))
        if len(self.equity) > self.window:
            self.equity = self.equity[-self.window:]

    def summary(self):
        r = np.diff(self.equity) if len(self.equity)>1 else np.array([0.0])
        if r.std() == 0: sharpe = 0.0
        else: sharpe = (r.mean() / r.std()) * np.sqrt(252)
        maxdd = 0.0
        peak = -1e18
        dd = 0.0
        for x in self.equity:
            peak = max(peak, x)
            dd = max(dd, peak - x)
            maxdd = max(maxdd, dd)
        return {"sharpe": float(sharpe), "max_dd": float(maxdd)}