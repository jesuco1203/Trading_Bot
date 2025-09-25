from typing import Dict, List

class RegimeSelector:
    def __init__(self, hmm, mapping: Dict[str, List[str]], enter_th: float = 0.45, exit_th: float = 0.35, persistence: int = 2):
        self.hmm = hmm
        self.mapping = mapping
        self.enter_th = enter_th
        self.exit_th  = exit_th
        self.persistence = persistence
        self.current_regime = None
        self.regime_duration = 0

    def active_strategies_with_reason(self, proba_dict: Dict[str, float]) -> tuple[List[str], str]:
        if not proba_dict:
            return [], "empty_proba_dict"

        max_proba_val = max(proba_dict.values())
        
        # Determine current most probable regime
        raw_label = max(proba_dict, key=proba_dict.get)
        lab = raw_label.lower()
        if lab in ("trending","trend_up","trend"):
            lab = "trend"
        elif lab in ("mean_revert","meanrevert","range","mr"):
            lab = "mr"
        elif lab in ("highvol","high_vol","vol"):
            lab = "high_vol"
        current_most_probable_regime = lab

        # Update regime duration
        if current_most_probable_regime == self.current_regime:
            self.regime_duration += 1
        else:
            self.current_regime = current_most_probable_regime
            self.regime_duration = 1

        # Check if regime is persistent enough
        if self.regime_duration < self.persistence:
            return [], f"regime_not_persistent_enough_{self.regime_duration}/{self.persistence}"

        # Check entry/exit thresholds
        if max_proba_val < self.enter_th:
            return [], f"below_enter_th_{max_proba_val:.2f}"

        # If current regime is below exit threshold, disengage
        if self.current_regime and proba_dict.get(self.current_regime, 0.0) < self.exit_th:
            self.current_regime = None
            self.regime_duration = 0
            return [], f"below_exit_th_{proba_dict.get(self.current_regime, 0.0):.2f}"

        regime = self.current_regime # Use the persistent regime
        strategies = self.mapping.get(regime, [])
        if not strategies:
            return [], "mapping_empty" # Changed reason
        return strategies, "ok"

    def active_strategies(self, proba_dict: Dict[str, float]) -> List[str]:
        strategies, _ = self.active_strategies_with_reason(proba_dict)
        return strategies
