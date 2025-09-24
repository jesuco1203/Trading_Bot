import numpy as np

class RollingMetrics:
    def __init__(self, window=1000):
        self.window = window
        self.equity = []
        self.initial_equity = None # To store initial equity for drawdown_pct
        self.peak_equity = -np.inf # Track peak equity for drawdown calculation
        self.consecutive_loss_count = 0
        self.last_pnl_positive = True

    def update(self, ts, equity, exposure):
        if self.initial_equity is None:
            self.initial_equity = float(equity)
        self.equity.append(float(equity))
        if len(self.equity) > self.window:
            self.equity = self.equity[-self.window:]
        
        current_equity = float(equity)
        self.peak_equity = max(self.peak_equity, current_equity)

        # Update consecutive losses
        if len(self.equity) > 1:
            current_pnl = self.equity[-1] - self.equity[-2]
            if current_pnl < 0:
                if self.last_pnl_positive:
                    self.consecutive_loss_count = 1
                else:
                    self.consecutive_loss_count += 1
                self.last_pnl_positive = False
            elif current_pnl > 0:
                self.consecutive_loss_count = 0
                self.last_pnl_positive = True

    def drawdown_pct(self):
        if not self.equity or self.peak_equity == -np.inf or self.peak_equity == 0:
            return 0.0
        current_equity = self.equity[-1]
        drawdown = self.peak_equity - current_equity
        return - (drawdown / self.peak_equity)

    def consecutive_losses(self):
        return self.consecutive_loss_count

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