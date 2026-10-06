"""Shared by both experiment folders: the cached TabPFN-3.5 wrapper, the sklearn baselines and the metrics.

Every predictor has the same shape: f(X, Ys: {name: y}, Xt) -> {name: (q05, q50, q95)} arrays of len(Xt).
TabPFN answers are cached in experiments/cache/ (keyed by model, ensemble size and data), so re-runs are free.
"""
import hashlib
import os
import time

import numpy as np

os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", "2")
EXP = os.path.dirname(os.path.abspath(__file__))
RES, CACHE, FIG = (os.path.join(EXP, d) for d in ("results", "cache", "figures"))
for d in (RES, CACHE, FIG):
    os.makedirs(d, exist_ok=True)
QS = np.array([0.05, 0.5, 0.95])
Z90 = 1.6449
MODELS = {"tabpfn": "tabpfn-v3.5-20260909.safetensors", "tabpfn_fast": "tabpfn-v3.5-fast-20260909.safetensors"}


# ============================================================================ predictors
# Every predictor: f(X, Ys: {name: y}, Xt) -> {name: (q05, q50, q95)} arrays of len(Xt)
class TabPFN:
    """TabPFN-3.5 (local weights), targets stacked into one regression exactly like server.py."""
    latency = []

    n_estimators = None                        # None = TabPFN default; set by --n-estimators

    def __init__(self, key, device):
        self.key, self.device = key, device

    def __call__(self, X, Ys, Xt):
        X, Xt = np.asarray(X, float), np.asarray(Xt, float)
        names = list(Ys)
        ys = {n: np.asarray(Ys[n], float) for n in names}
        tag = self.key if TabPFN.n_estimators is None else f"{self.key}|n{TabPFN.n_estimators}"
        h = hashlib.sha1(tag.encode() + X.tobytes() + Xt.tobytes() + str(X.shape).encode()
                         + b"".join(ys[n].tobytes() for n in names) + ",".join(names).encode()).hexdigest()
        path = os.path.join(CACHE, h + ".npz")
        if os.path.exists(path):
            z = np.load(path)
            return {n: tuple(z[f"{n}_{i}"] for i in range(3)) for n in names}
        k, eye = len(names), np.eye(len(names))
        st = {n: (ys[n].mean(), ys[n].std() + 1e-9) for n in names}
        Xs = np.vstack([np.hstack([X, np.tile(eye[j], (len(X), 1))]) for j in range(k)])
        yst = np.concatenate([(ys[n] - st[n][0]) / st[n][1] for n in names])
        Xts = np.vstack([np.hstack([Xt, np.tile(eye[j], (len(Xt), 1))]) for j in range(k)])
        from tabpfn import TabPFNRegressor
        from tabpfn.model_loading import prepend_cache_path
        t0 = time.time()
        kw = {} if TabPFN.n_estimators is None else {"n_estimators": TabPFN.n_estimators}
        reg = TabPFNRegressor(model_path=prepend_cache_path(MODELS[self.key]), device=self.device, random_state=0, **kw)
        reg.fit(Xs, yst)
        q = np.sort(np.stack([np.asarray(a, float).reshape(-1) for a in
                              reg.predict(Xts, output_type="quantiles", quantiles=list(QS))]), axis=0)
        TabPFN.latency.append({"model": self.key, "device": self.device, "n_estimators": TabPFN.n_estimators or "auto", "train_rows": len(Xs), "test_rows": len(Xts),
                               "features": Xs.shape[1], "seconds": time.time() - t0})
        out, n = {}, len(Xt)
        for j, nm in enumerate(names):
            mu, sd = st[nm]
            out[nm] = tuple(q[i, j * n:(j + 1) * n] * sd + mu for i in range(3))
        np.savez(path, **{f"{nm}_{i}": out[nm][i] for nm in names for i in range(3)})
        return out


def _gauss(m, s):
    s = np.maximum(np.asarray(s, float), 1e-4)
    return (m - Z90 * s, m, m + Z90 * s)


def _std(X):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    return lambda A: (np.asarray(A, float) - mu) / sd


def gbr(X, Ys, Xt):
    from sklearn.ensemble import HistGradientBoostingRegressor
    X, Xt = np.asarray(X, float), np.asarray(Xt, float)
    out = {}
    for n, y in Ys.items():
        qs = [HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=150, learning_rate=0.1,
                                            min_samples_leaf=max(1, min(20, len(X) // 10)), random_state=0)
              .fit(X, y).predict(Xt) for q in QS]
        out[n] = tuple(np.sort(np.stack(qs), axis=0))
    return out


def rf(X, Ys, Xt):
    from sklearn.ensemble import RandomForestRegressor
    X, Xt = np.asarray(X, float), np.asarray(Xt, float)
    out = {}
    for n, y in Ys.items():
        y = np.asarray(y, float)
        oob = len(X) >= 15
        m = RandomForestRegressor(200, min_samples_leaf=2, oob_score=oob, random_state=0, n_jobs=-1).fit(X, y)
        res = (y - m.oob_prediction_) if oob else (y - m.predict(X)) * 2
        out[n] = _gauss(m.predict(Xt), np.std(res))
    return out


def knn(X, Ys, Xt, k=7):
    z = _std(np.asarray(X, float)); A, B = z(X), z(Xt)
    d = np.sqrt(((B[:, None, :] - A[None]) ** 2).sum(-1))
    k = min(k, len(A)); idx = np.argsort(d, 1)[:, :k]
    w = 1 / (np.take_along_axis(d, idx, 1) + 0.05); w /= w.sum(1, keepdims=True)
    out = {}
    for n, y in Ys.items():
        yy = np.asarray(y, float)[idx]
        m = (w * yy).sum(1); v = (w * (yy - m[:, None]) ** 2).sum(1)
        out[n] = _gauss(m, np.sqrt(v * (1 + 1 / k)) + 0.01)
    return out


def ridge(X, Ys, Xt):
    from sklearn.linear_model import RidgeCV
    z = _std(np.asarray(X, float)); A, B = z(X), z(Xt)
    out = {}
    for n, y in Ys.items():
        y = np.asarray(y, float)
        m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(A, y)
        out[n] = _gauss(m.predict(B), np.std(y - m.predict(A)) * np.sqrt(1 + 1 / len(A)))
    return out


ML = {"gbr": gbr, "rf": rf, "knn": knn, "ridge": ridge}


def metrics(truth, q):
    """truth (n,), q = (q05, q50, q95) -> per-point abs error, coverage, width, pinball."""
    lo, md, hi = q
    pin = np.mean([np.maximum(a * (truth - b), (a - 1) * (truth - b)) for a, b in zip(QS, q)], axis=0)
    return np.abs(md - truth), ((truth >= lo) & (truth <= hi)).astype(float), hi - lo, pin
