"""Adaptation experiments (lean set): X1 shuttle worlds, X2 world switches, X3 opponent habits.

    node experiments/adaptation/gen_adapt.js --seeds 2
    .venv/bin/python experiments/adaptation/run_adapt.py --device mps --n-estimators 4
    .venv/bin/python experiments/adaptation/analyze_adapt.py

TabPFN answers are cached in experiments/cache (shared with the physics comparison): re-runs are free.
Outputs: experiments/results/adapt_x1.csv.gz, adapt_x2.csv.gz, adapt_x3.csv.gz
"""
import argparse
import glob
import hashlib
import json
import os
import time

import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)
import common as R                              # noqa: E402  TabPFN wrapper + cache, sklearn baselines, metrics
from common import _gauss, metrics              # noqa: E402

DATA = os.path.join(EXP, "data_adapt")
K = 1                                           # landing checkpoint used: 0.4 s after the hit
TYPES = ["clear", "drop", "smash"]


def load(prefix):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(DATA, prefix + "*.json")))]


def last(s):
    return np.array([s["feats"][K]["last"]["x"], s["feats"][K]["last"]["y"]])


def landing_preds(ctx, test, models, age=None):
    """Fit on ctx shots, predict test shots. age: (ctx_ages, test_ages) appended as a feature, or None."""
    X = np.array([s["feats"][K]["x"] for s in ctx], float); Xt = np.array([s["feats"][K]["x"] for s in test], float)
    if age is not None:
        X = np.hstack([X, np.asarray(age[0], float)[:, None]]); Xt = np.hstack([Xt, np.asarray(age[1], float)[:, None]])
    Y = {"dx": [s["landing"]["x"] - last(s)[0] for s in ctx], "dy": [s["landing"]["y"] - last(s)[1] for s in ctx],
         "tr": [s["T"] - 0.4 for s in ctx]}
    L = np.array([last(s) for s in test])
    out = {}
    for name, f in models.items():
        o = f(X, Y, Xt)
        out[name] = (tuple(L[:, 0] + a for a in o["dx"]), tuple(L[:, 1] + a for a in o["dy"]))
    return out


def hybrid_preds(ctx, test, f):
    """TabPFN on top of the default physics guess: the guess (relative to the last observation) is an extra
    feature, and TabPFN predicts truth minus guess. The physics guess is the game's timing+drag fit, tuned for
    the normal world, so in a new world TabPFN learns the correction."""
    def ph(s):
        p = s["phys"][K]; L = last(s)
        return [p["mx"] - L[0], p["my"] - L[1], p["T"] - 0.4]
    X = np.array([s["feats"][K]["x"] + ph(s) for s in ctx], float); Xt = np.array([s["feats"][K]["x"] + ph(s) for s in test], float)
    Y = {"rx": [s["landing"]["x"] - s["phys"][K]["mx"] for s in ctx], "ry": [s["landing"]["y"] - s["phys"][K]["my"] for s in ctx]}
    o = f(X, Y, Xt)
    P = np.array([[s["phys"][K]["mx"], s["phys"][K]["my"]] for s in test])
    return (tuple(P[:, 0] + a for a in o["rx"]), tuple(P[:, 1] + a for a in o["ry"]))


def phys_pred(test):
    P = [s["phys"][K] for s in test]
    return (_gauss(np.array([p["mx"] for p in P]), [p["sx"] for p in P]), _gauss(np.array([p["my"] for p in P]), [p["sy"] for p in P]))


def score(test, preds, rows, meta):
    tx = np.array([s["landing"]["x"] for s in test]); ty = np.array([s["landing"]["y"] for s in test])
    for name, (qx, qy) in preds.items():
        _, cx, wx, px = metrics(tx, qx); _, cy, wy, py = metrics(ty, qy)
        for j, s in enumerate(test):
            rows.append({**meta, "shot": s["i"], "world": s["world"], "model": name,
                         "err": float(np.hypot(qx[1][j] - tx[j], qy[1][j] - ty[j])), "cov90": float((cx[j] + cy[j]) / 2),
                         "width": float((wx[j] + wy[j]) / 2), "pinball": float((px[j] + py[j]) / 2)})


# ----------------------------------------------------------------------------- X1 shuttle worlds, few-shot
def x1(tp):
    rows = []
    normal = {d["seed"]: d for d in load("x1_normal_")}
    for d in load("x1_"):
        pts = [s for s in d["shots"] if s["feats"][K]]
        test = [s for s in pts if s["i"] >= 120]
        for n in (5, 10, 20, 40, 120):
            ctx = [s for s in pts if s["i"] < 120][-n:]
            preds = landing_preds(ctx, test, {"tabpfn": tp, "gbr": R.gbr, "rf": R.rf, "knn": R.knn})
            preds["tabpfn_hybrid"] = hybrid_preds(ctx, test, tp)
            if n == 120:
                preds["physics_normal"] = phys_pred(test)
                if d["world"] != "normal":   # TabPFN that only ever saw the normal world (does not adapt)
                    nctx = [s for s in normal[d["seed"]]["shots"] if s["feats"][K] and s["i"] < 120]
                    preds["tabpfn_normal_ctx"] = landing_preds(nctx, test, {"t": tp})["t"]
            score(test, preds, rows, {"world": d["world"], "seed": d["seed"], "n": n})
        print(f"[x1] {d['world']} s{d['seed']}", flush=True)
    return rows


# ----------------------------------------------------------------------------- X2 sudden world switches
def x2(tp):
    rows = []
    for d in load("x2_"):
        pts = [s for s in d["shots"] if s["feats"][K]]
        frozen = [s for s in pts if s["i"] < 50]                 # what a model trained in the first world knows
        for b in range(10, len(d["shots"]), 10):
            ctx = [s for s in pts if b - 120 <= s["i"] < b]
            test = [s for s in pts if b <= s["i"] < b + 10]
            if len(ctx) < 5 or not test:
                continue
            preds = landing_preds(ctx, test, {"tabpfn": tp, "gbr": R.gbr})
            ages = ([b - s["i"] for s in ctx], [0] * len(test))   # "how many shots ago" (the query is now)
            for nm, v in landing_preds(ctx, test, {"tabpfn_age": tp, "gbr_age": R.gbr}, age=ages).items():
                preds[nm] = v
            preds["tabpfn_frozen"] = landing_preds(frozen, test, {"t": tp})["t"]
            preds["tabpfn_hybrid"] = hybrid_preds(ctx, test, tp)
            preds["physics_normal"] = phys_pred(test)
            score(test, preds, rows, {"seed": d["seed"], "b": b})
        print(f"[x2] s{d['seed']}", flush=True)
    return rows


# ----------------------------------------------------------------------------- X3 opponent habits
class TabPFNClf:
    """TabPFNClassifier (same TabPFN-3.5 checkpoint) with the shared disk cache."""
    def __init__(self, device):
        self.device = device

    def __call__(self, X, y, Xt):
        X, Xt = np.asarray(X, float), np.asarray(Xt, float)
        tag = f"clf|n{R.TabPFN.n_estimators}"
        h = hashlib.sha1(tag.encode() + X.tobytes() + Xt.tobytes() + ",".join(y).encode()).hexdigest()
        path = os.path.join(R.CACHE, h + ".npz")
        if os.path.exists(path):
            return np.load(path)["p"]
        from tabpfn import TabPFNClassifier
        from tabpfn.model_loading import prepend_cache_path
        kw = {} if R.TabPFN.n_estimators is None else {"n_estimators": R.TabPFN.n_estimators}
        t0 = time.time()
        m = TabPFNClassifier(model_path=prepend_cache_path(R.MODELS["tabpfn"]), device=self.device, random_state=0, **kw).fit(X, y)
        pr = m.predict_proba(Xt)
        R.TabPFN.latency.append({"model": "tabpfn_clf", "device": self.device, "n_estimators": R.TabPFN.n_estimators or "auto",
                                 "train_rows": len(X), "test_rows": len(Xt), "features": X.shape[1], "seconds": time.time() - t0})
        p = np.zeros((len(Xt), len(TYPES)))
        for j, c in enumerate(m.classes_):
            p[:, TYPES.index(c)] = pr[:, j]
        np.savez(path, p=p)
        return p


def clf_sklearn(kind, X, y, Xt):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = {"gbr": lambda: HistGradientBoostingClassifier(max_iter=150, min_samples_leaf=max(1, min(20, len(X) // 10)), random_state=0),
         "knn": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(min(7, len(X)), weights="distance")),
         "linear": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))}[kind]().fit(X, y)
    pr = m.predict_proba(Xt); p = np.zeros((len(Xt), len(TYPES)))
    for j, c in enumerate(m.classes_):
        p[:, TYPES.index(c)] = pr[:, j]
    return p


def x3(tp, tc):
    rows = []
    for d in load("x3_"):
        R_ = d["rows"]
        test = R_[120:]
        Xt = np.array([r["x"] for r in test], float)
        tx = np.array([r["landing"]["x"] for r in test]); ty = np.array([r["landing"]["y"] for r in test])
        tt = np.array([TYPES.index(r["type"]) for r in test])
        for n in (5, 10, 20, 40, 80):
            ctx = R_[120 - n:120]
            X = np.array([r["x"] for r in ctx], float)
            Y = {"x": [r["landing"]["x"] for r in ctx], "y": [r["landing"]["y"] for r in ctx]}
            yc = [r["type"] for r in ctx]
            reg = {}
            for name, f in {"tabpfn": tp, "gbr": R.gbr, "knn": R.knn, "linear": R.ridge}.items():
                o = f(X, Y, Xt); reg[name] = (o["x"], o["y"])
            reg["prior"] = (_gauss(np.full(len(test), np.mean(Y["x"])), np.std(Y["x"]) + 0.1), _gauss(np.full(len(test), np.mean(Y["y"])), np.std(Y["y"]) + 0.1))
            # shot type
            cls = {}
            freq = np.array([(np.array(yc) == c).sum() + 0.5 for c in TYPES], float); freq /= freq.sum()
            cls["prior"] = np.tile(freq, (len(test), 1))
            if len(set(yc)) < 2:
                for name in ("tabpfn", "gbr", "knn", "linear"):
                    cls[name] = cls["prior"]
            else:
                cls["tabpfn"] = tc(X, yc, Xt)
                for name in ("gbr", "knn", "linear"):
                    cls[name] = clf_sklearn(name, X, yc, Xt)
            for name in reg:
                (qx, qy), p = reg[name], cls.get(name)
                _, cx, wx, _ = metrics(tx, qx); _, cy, wy, _ = metrics(ty, qy)
                for j, r in enumerate(test):
                    rows.append({"personality": d["personality"], "seed": d["seed"], "n": n, "i": r["i"], "model": name,
                                 "err": float(np.hypot(qx[1][j] - tx[j], qy[1][j] - ty[j])), "cov90": float((cx[j] + cy[j]) / 2),
                                 "width": float((wx[j] + wy[j]) / 2),
                                 "type_correct": float(np.argmax(p[j]) == tt[j]) if p is not None else np.nan,
                                 "type_logloss": float(-np.log(max(1e-6, p[j, tt[j]]))) if p is not None else np.nan})
        print(f"[x3] {d['personality']} s{d['seed']}", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n-estimators", type=int, default=4)
    ap.add_argument("--only", default="x1,x2,x3")
    a = ap.parse_args()
    R.TabPFN.n_estimators = a.n_estimators
    tp, tc = R.TabPFN("tabpfn", a.device), TabPFNClf(a.device)
    t0 = time.time()
    for part in a.only.split(","):
        rows = {"x1": lambda: x1(tp), "x2": lambda: x2(tp), "x3": lambda: x3(tp, tc)}[part]()
        pd.DataFrame(rows).to_csv(os.path.join(R.RES, f"adapt_{part}.csv.gz"), index=False)
    if R.TabPFN.latency:
        pd.DataFrame(R.TabPFN.latency).to_csv(os.path.join(R.RES, "latency_adapt.csv"), index=False)
    print(f"[adapt] done in {(time.time() - t0) / 60:.1f} min, {len(R.TabPFN.latency)} new TabPFN calls", flush=True)
