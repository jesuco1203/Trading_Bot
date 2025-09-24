from typing import Dict, List, Any

class RegimeSelector:
    def __init__(self, hmm, mapping: Dict[str, List[str]]):
        self.hmm = hmm
        self.mapping = mapping
        self.enter_th = 0.45
        self.exit_th  = 0.35

    def active_strategies_with_reason(self, proba_dict: Dict[str, float]) -> tuple[List[str], str]:
        if not proba_dict:
            return [], "empty_proba_dict"

        max_proba_val = max(proba_dict.values())
        
        if max_proba_val < self.enter_th:
            return [], f"below_enter_th_{max_proba_val:.2f}"

        regime = max(proba_dict, key=proba_dict.get)
        strategies = self.mapping.get(regime, [])
        if not strategies:
            return [], f"no_mapping_for_{regime}"
        return strategies, "ok"

    def active_strategies(self, proba_dict: Dict[str, float]) -> List[str]:
        strategies, _ = self.active_strategies_with_reason(proba_dict)
        return strategies
