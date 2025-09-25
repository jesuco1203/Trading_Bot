import numpy as np

class RollingMetrics:
    def __init__(self, window=1000):
        self.window = window
        self.equity = []
        self.initial_equity = None
        self.peak_equity = -np.inf
        self.consecutive_loss_count = 0
        self.last_pnl_positive = True

        self.peak_equity_strategy = {}
        self.consecutive_losses_strategy = {}
        self.last_pnl_positive_strategy = {}

    def update(self, ts, equity, exposure, pnl=0.0, strategy_name=None):
        if self.initial_equity is None:
            self.initial_equity = float(equity)
        self.equity.append(float(equity))
        if len(self.equity) > self.window:
            self.equity = self.equity[-self.window:]
        
        current_equity = float(equity)
        self.peak_equity = max(self.peak_equity, current_equity)

        if pnl != 0:
            if pnl < 0:
                if self.last_pnl_positive:
                    self.consecutive_loss_count = 1
                else:
                    self.consecutive_loss_count += 1
                self.last_pnl_positive = False
            else:
                self.consecutive_loss_count = 0
                self.last_pnl_positive = True

        if strategy_name:
            if strategy_name not in self.peak_equity_strategy:
                self.peak_equity_strategy[strategy_name] = -np.inf
                self.consecutive_losses_strategy[strategy_name] = 0
                self.last_pnl_positive_strategy[strategy_name] = True

            self.peak_equity_strategy[strategy_name] = max(self.peak_equity_strategy[strategy_name], current_equity)

            if pnl != 0:
                if pnl < 0:
                    if self.last_pnl_positive_strategy[strategy_name]:
                        self.consecutive_losses_strategy[strategy_name] = 1
                    else:
                        self.consecutive_losses_strategy[strategy_name] += 1
                    self.last_pnl_positive_strategy[strategy_name] = False
                else:
                    self.consecutive_losses_strategy[strategy_name] = 0
                    self.last_pnl_positive_strategy[strategy_name] = True

    def drawdown_pct(self, strategy_name=None):
        if strategy_name:
            if strategy_name not in self.peak_equity_strategy or self.peak_equity_strategy[strategy_name] == -np.inf or self.peak_equity_strategy[strategy_name] == 0:
                return 0.0
            current_equity = self.equity[-1]
            drawdown = self.peak_equity_strategy[strategy_name] - current_equity
            return - (drawdown / self.peak_equity_strategy[strategy_name])
        else:
            if not self.equity or self.peak_equity == -np.inf or self.peak_equity == 0:
                return 0.0
            current_equity = self.equity[-1]
            drawdown = self.peak_equity - current_equity
            return - (drawdown / self.peak_equity)

    def consecutive_losses(self, strategy_name=None):
        if strategy_name:
            return self.consecutive_losses_strategy.get(strategy_name, 0)
        else:
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