import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class RegimeHMM:
    def __init__(self, n_states=3, random_state=42):
        self.n_states = n_states # Store n_states as an instance attribute
        self.model  = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",   # más estable que 'full' en 3 features sencillos
            min_covar=5e-3,
            random_state=random_state,
            n_iter=300,
            tol=1e-3
        )
        self.scaler = StandardScaler()
        self.labels = ["trend","mr","high_vol"][:n_states]
        self._fitted_once = False

    def fit(self, X: np.ndarray):
        Xs = self.scaler.fit_transform(X) if not self._fitted_once else self.scaler.transform(X)

        # Init por KMeans para evitar colapso
        k = min(self.model.n_components, len(np.unique(Xs, axis=0)))
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
        means_init = km.cluster_centers_
        self.model.means_init = means_init

        # Warm-start (no reinit después del primer fit)
        self.model.init_params = "" if self._fitted_once else "stmc"
        self.model.fit(Xs)
        self._fitted_once = True

        # Mapeo de estados por varianza (heurístico estable)
        covs  = np.array([np.mean(c) for c in self.model.covars_])
        order = np.argsort(covs)
        
        # Normalize labels to a consistent set
        normalized_labels = []
        for rank, state in enumerate(order):
            raw_label = self.labels[min(rank, len(self.labels)-1)]
            if raw_label in ("trending", "trend_up", "trend"): normalized_labels.append("trend")
            elif raw_label in ("mean_revert", "meanrevert", "range", "mr"): normalized_labels.append("mr")
            elif raw_label in ("highvol", "high_vol", "vol"): normalized_labels.append("high_vol")
            else: normalized_labels.append(raw_label) # Fallback

        self._state2name = {state: normalized_labels[rank]
                            for rank, state in enumerate(order)}
        return self

    def predict_proba(self, X_t: np.ndarray):
        Xs = self.scaler.transform(X_t)
        P  = self.model.predict_proba(Xs)
        out = []
        for p in P:
            d = { self._state2name.get(i, self.labels[-1]): float(p[i])
                  for i in range(self.model.n_components) }
            for k in self.labels: d.setdefault(k, 0.0)
            out.append(d)
        return out