"""Run the TabPFN vs ML vs physics experiments on the generated data.

    node experiments/physics_comparison/gen_data.js --seeds 3
    .venv/bin/python experiments/physics_comparison/run_experiments.py --part all --device mps --n-estimators 4
    .venv/bin/python experiments/physics_comparison/analyze.py                     # tables + figures

Parts: landing (E1, E2, E4), dynamics (E5, E6), curves (E3), grid (E7b). This earlier study is not part of the
report; the run was stopped partway (all of landing; movement for the court-change and default scenarios).

Every TabPFN answer is cached in experiments/cache (keyed by model + data), so re-runs are free.
Outputs: experiments/results/<part>.csv.gz, one row per (test point, model).
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path[:0] = [HERE, EXP]
from physics_models import NOMINAL, SimplifiedPhysics, calibrate, step_dv, whitebox_landing  # noqa: E402

DATA = os.path.join(EXP, "data")
from common import (CACHE, MODELS, ML, QS, RES, TabPFN, Z90, _gauss, _std, gbr, knn,  # noqa: E402,F401
                    metrics, rf, ridge)


# ============================================================================ landing (E1, E2, E4)
ONLY = None        # optional extra filter on data file names (--only)
OUT_TAG = ""       # suffix for result files (--tag)


def load(pattern):
    files = sorted(glob.glob(os.path.join(DATA, pattern)))
    if ONLY:
        files = [f for f in files if any(o in os.path.basename(f) for o in ONLY.split(","))]
    return [(os.path.basename(f)[:-5], json.load(open(f))) for f in files]


def out(name):
    return os.path.join(RES, f"{name}{OUT_TAG}.csv.gz")


def whitebox_cache(d):
    wb = {}
    for s in d["shots"]:
        for k, c in enumerate(d["checkpoints"]):
            if s["feats"][k]:
                wb[(s["i"], k)] = whitebox_landing(s["samples"], s["p0"], d["shotParams"][s["type"]], c)
    return wb


def landing_eval(d, k, ctx, test, wb, predictors, rows, meta):
    """Fit on ctx shots, predict test shots at checkpoint k with every model; append per-point rows."""
    c = d["checkpoints"][k]
    last = lambda s: np.array([s["feats"][k]["last"]["x"], s["feats"][k]["last"]["y"]])
    X = np.array([s["feats"][k]["x"] for s in ctx]); Xt = np.array([s["feats"][k]["x"] for s in test])
    Y = {"dx": [s["landing"]["x"] - last(s)[0] for s in ctx], "dy": [s["landing"]["y"] - last(s)[1] for s in ctx],
         "tr": [s["T"] - c for s in ctx]}
    # hybrid: physics (white-box) prediction as extra features; target = truth - physics
    wbf = lambda s: [wb[(s["i"], k)][0] - last(s)[0], wb[(s["i"], k)][1] - last(s)[1], wb[(s["i"], k)][4] - c]
    Xh = np.hstack([X, [wbf(s) for s in ctx]]); Xht = np.hstack([Xt, [wbf(s) for s in test]])
    Yh = {"rx": [s["landing"]["x"] - wb[(s["i"], k)][0] for s in ctx], "ry": [s["landing"]["y"] - wb[(s["i"], k)][1] for s in ctx],
          "rT": [s["T"] - wb[(s["i"], k)][4] for s in ctx]}
    L = np.array([last(s) for s in test]); W = np.array([wb[(s["i"], k)] for s in test])
    preds = {}                                          # name -> (qx, qy, qT) in absolute coordinates
    for name, f in predictors.items():
        if name.endswith("_hybrid"):
            o = f(Xh, Yh, Xht)
            preds[name] = (tuple(W[:, 0] + a for a in o["rx"]), tuple(W[:, 1] + a for a in o["ry"]), tuple(W[:, 4] + a for a in o["rT"]))
        else:
            o = f(X, Y, Xt)
            preds[name] = (tuple(L[:, 0] + a for a in o["dx"]), tuple(L[:, 1] + a for a in o["dy"]), tuple(c + a for a in o["tr"]))
    # physics: white-box fit, white-box + constant-wind correction (mean residual of the last 10 shots)
    preds["phys_whitebox"] = (_gauss(W[:, 0], W[:, 2]), _gauss(W[:, 1], W[:, 3]), _gauss(W[:, 4], 0.15))
    rec = ctx[-10:]
    cx = np.mean([s["landing"]["x"] - wb[(s["i"], k)][0] for s in rec]); cy = np.mean([s["landing"]["y"] - wb[(s["i"], k)][1] for s in rec])
    sres = np.sqrt(np.mean([(s["landing"]["x"] - wb[(s["i"], k)][0] - cx) ** 2 + (s["landing"]["y"] - wb[(s["i"], k)][1] - cy) ** 2 for s in rec]) / 2) + 0.05
    preds["phys_whitebox_wind"] = (_gauss(W[:, 0] + cx, sres), _gauss(W[:, 1] + cy, sres), _gauss(W[:, 4], 0.15))
    for name in ("physics", "physics_wind", "ekf", "pf", "linear"):          # precomputed in Node (dev-tools.js)
        P = [s["phys"][name][k] for s in test]
        preds["phys_" + name] = (_gauss(np.array([p["mx"] for p in P]), [p["sx"] for p in P]),
                                 _gauss(np.array([p["my"] for p in P]), [p["sy"] for p in P]),
                                 _gauss(np.array([p["T"] for p in P]), [p["sT"] for p in P]))
    tx = np.array([s["landing"]["x"] for s in test]); ty = np.array([s["landing"]["y"] for s in test]); tT = np.array([s["T"] for s in test])
    for name, (qx, qy, qT) in preds.items():
        ex, cvx, wx, px = metrics(tx, qx); ey, cvy, wy, py = metrics(ty, qy); eT = np.abs(qT[1] - tT)
        for j, s in enumerate(test):
            rows.append({**meta, "k": k, "checkpoint": c, "shot": s["i"], "type": s["type"], "model": name,
                         "err": float(np.hypot(qx[1][j] - tx[j], qy[1][j] - ty[j])), "cov90": float((cvx[j] + cvy[j]) / 2),
                         "width": float((wx[j] + wy[j]) / 2), "pinball": float((px[j] + py[j]) / 2), "err_T": float(eT[j]),
                         "n_ctx": len(ctx)})


def part_landing(device):
    tp, tf = TabPFN("tabpfn", device), TabPFN("tabpfn_fast", device)
    rows = []
    for name, d in load("landing_*.json"):
        scen, seed = d["scenario"], d["seed"]
        preds = {"tabpfn": tp, "tabpfn_hybrid": tp, **ML, "gbr_hybrid": gbr}
        if scen in ("wind1", "wind3"):
            preds["tabpfn_fast"] = tf
        wb = whitebox_cache(d)
        t0 = time.time()
        for k in range(3):
            pts = [s for s in d["shots"] if s["feats"][k]]
            for b in range(40, len(d["shots"]), 40):          # refit every 40 shots, context = last 120
                ctx = [s for s in pts if b - 120 <= s["i"] < b]
                test = [s for s in pts if b <= s["i"] < b + 40]
                if len(ctx) >= 10 and test:
                    landing_eval(d, k, ctx, test, wb, preds, rows, {"scenario": scen, "seed": seed})
        print(f"[landing] {name}: {time.time() - t0:.0f} s", flush=True)
        pd.DataFrame(rows).to_csv(out("landing"), index=False)


# ============================================================================ dynamics (E5, E6)
def dyn_arrays(d):
    T = d["trans"]
    return np.array([t["x"] for t in T], float), np.array([t["dv"] for t in T], float), np.array([t["t"] for t in T], float)


def rls_run(F, dv, lam=0.98):
    """Recursive least squares with forgetting on [features, 1]; returns per-row (pred made before seeing it, running sd)."""
    n, p = len(F), F.shape[1] + 1
    th = np.zeros((p, 2)); P = np.eye(p) * 100; var = np.ones(2) * 0.25
    pred, sd = np.zeros((n, 2)), np.zeros((n, 2))
    for i in range(n):
        x = np.append(F[i], 1.0)
        pred[i] = x @ th; sd[i] = np.sqrt(var)
        e = dv[i] - pred[i]
        g = P @ x / (lam + x @ P @ x)
        th += np.outer(g, e); P = (P - np.outer(g, x @ P)) / lam
        var = 0.98 * var + 0.02 * e ** 2
    return pred, sd


def dyn_eval(d, F, dv, ctx_idx, test_idx, predictors, rows, meta, state):
    B = d["bounds"]
    X, Xt = F[ctx_idx], F[test_idx]
    Y = {"dvx": dv[ctx_idx, 0], "dvy": dv[ctx_idx, 1]}
    nom_c, nom_t = step_dv(X, NOMINAL, B), step_dv(Xt, NOMINAL, B)
    Xh, Xht = np.hstack([X, nom_c]), np.hstack([Xt, nom_t])
    Yh = {"rx": dv[ctx_idx, 0] - nom_c[:, 0], "ry": dv[ctx_idx, 1] - nom_c[:, 1]}
    preds = {}
    for name, f in predictors.items():
        if name == "tabpfn_frozen":
            if state.get("frozen") is None:
                continue
            fi = state["frozen"]
            o = f(F[fi], {"dvx": dv[fi, 0], "dvy": dv[fi, 1]}, Xt)
            preds[name] = (o["dvx"], o["dvy"])
        elif name.endswith("_hybrid"):
            o = f(Xh, Yh, Xht)
            preds[name] = (tuple(nom_t[:, 0] + a for a in o["rx"]), tuple(nom_t[:, 1] + a for a in o["ry"]))
        else:
            o = f(X, Y, Xt)
            preds[name] = (o["dvx"], o["dvy"])
    sd_nom = (dv[ctx_idx] - nom_c).std(0) + 1e-3
    preds["phys_nominal"] = (_gauss(nom_t[:, 0], sd_nom[0]), _gauss(nom_t[:, 1], sd_nom[1]))
    th, sd = calibrate(X, dv[ctx_idx], B, True, state.get("theta", NOMINAL)); state["theta"] = th
    pc = step_dv(Xt, th, B); preds["phys_calibrated"] = (_gauss(pc[:, 0], sd[0]), _gauss(pc[:, 1], sd[1]))
    th0, sd0 = calibrate(X, dv[ctx_idx], B, False, state.get("theta0", NOMINAL)); state["theta0"] = th0
    p0 = step_dv(Xt, th0, B); preds["phys_calibrated_nolag"] = (_gauss(p0[:, 0], sd0[0]), _gauss(p0[:, 1], sd0[1]))
    sp = SimplifiedPhysics().fit(X, dv[ctx_idx]); ps = sp.predict(Xt)
    preds["phys_simplified"] = (_gauss(ps[:, 0], sp.sd), _gauss(ps[:, 1], sp.sd))
    rp, rs = state["rls"]
    preds["rls"] = (_gauss(rp[test_idx, 0], rs[test_idx, 0]), _gauss(rp[test_idx, 1], rs[test_idx, 1]))
    for name, (qx, qy) in preds.items():
        ex, cx, wx, px = metrics(dv[test_idx, 0], qx); ey, cy, wy, py = metrics(dv[test_idx, 1], qy)
        for j, i in enumerate(test_idx):
            rows.append({**meta, "i": int(i), "t": float(state["t"][i]), "model": name,
                         "err": float(np.hypot(qx[1][j] - dv[i, 0], qy[1][j] - dv[i, 1])), "cov90": float((cx[j] + cy[j]) / 2),
                         "width": float((wx[j] + wy[j]) / 2), "pinball": float((px[j] + py[j]) / 2), "n_ctx": len(ctx_idx)})


def window_idx(t, b, rows=300, secs=90):
    lo = max(0, b - rows)
    idx = np.arange(lo, b)
    return idx[t[idx] >= t[b] - secs]


def part_dynamics(device):
    tp = TabPFN("tabpfn", device)
    rows = []
    for name, d in load("dynamics_*.json"):
        F, dv, t = dyn_arrays(d)
        scen, seed = d["scenario"], d["seed"]
        preds = {"tabpfn": tp, "tabpfn_hybrid": tp, **ML, "gbr_hybrid": gbr}
        if scen == "court":
            preds["tabpfn_frozen"] = tp
            preds["tabpfn_fast"] = TabPFN("tabpfn_fast", device)
        state = {"t": t, "rls": rls_run(F, dv), "frozen": None}
        t0 = time.time()
        for b in range(100, len(F), 60):                     # refit every 6 s, window = 300 rows / 90 s
            ci = window_idx(t, b)
            if scen == "court" and t[b] <= 60:
                state["frozen"] = ci                 # the last context the frozen model saw before the change
            ti = np.arange(b, min(b + 60, len(F)))
            dyn_eval(d, F, dv, ci, ti, preds, rows, {"scenario": scen, "seed": seed}, state)
        print(f"[dynamics] {name}: {time.time() - t0:.0f} s", flush=True)
        pd.DataFrame(rows).to_csv(out("dynamics"), index=False)


# ============================================================================ learning curves (E3)
def part_curves(device):
    tp = TabPFN("tabpfn", device)
    rows = []
    for name, d in load("landing_wind1_*.json"):
        wb = whitebox_cache(d); k = 1
        pts = [s for s in d["shots"] if s["feats"][k]]
        test = [s for s in pts if s["i"] >= 200]
        for n in (5, 10, 20, 40, 80, 120, 200):
            ctx = [s for s in pts if s["i"] < 200][-n:]
            landing_eval(d, k, ctx, test, wb, {"tabpfn": tp, "tabpfn_hybrid": tp, **ML, "gbr_hybrid": gbr}, rows,
                         {"task": "landing", "scenario": "wind1", "seed": d["seed"], "n": n})
        print(f"[curves] {name}", flush=True)
    for name, d in load("dynamics_*.json"):
        if d["scenario"] not in ("lag", "wetpatch", "default"):
            continue
        F, dv, t = dyn_arrays(d)
        test = np.arange(900, 1800)
        for n in (10, 25, 50, 100, 200, 300):
            ci = np.arange(900 - n, 900)
            state = {"t": t, "rls": rls_run(F, dv), "frozen": None}
            dyn_eval(d, F, dv, ci, test, {"tabpfn": tp, "tabpfn_hybrid": tp, **ML, "gbr_hybrid": gbr}, rows,
                     {"task": "dynamics", "scenario": d["scenario"], "seed": d["seed"], "n": n}, state)
        print(f"[curves] {name}", flush=True)
    pd.DataFrame(rows).to_csv(out("curves"), index=False)


# ============================================================================ in-game grid vs direct TabPFN (E7b)
def part_grid(device):
    """The game queries TabPFN on a 5x5x5x5 grid at the base position with u_prev = u, then interpolates.
    Compare that surrogate with asking TabPFN directly on the real rows (same context)."""
    tp = TabPFN("tabpfn", device)
    gv, gu = np.linspace(-4.5, 4.5, 5), np.linspace(-1, 1, 5)
    rows = []
    for name, d in load("dynamics_*.json"):
        if d["scenario"] not in ("default", "court", "lag", "wetpatch"):
            continue
        F, dv, t = dyn_arrays(d)
        for b in (300, 600, 900, 1200, 1500):
            ci = window_idx(t, b); ti = np.arange(b, b + 100)
            grid = []
            for vx in gv:
                for vy in gv:
                    for ux in gu:
                        for uy in gu:
                            L = max(1.0, np.hypot(ux, uy)); u = (ux / L, uy / L)
                            grid.append([0.0, -3.3, vx, vy, u[0], u[1], u[0], u[1]])
            Y = {"dvx": dv[ci, 0], "dvy": dv[ci, 1]}
            og = tp(F[ci], Y, np.array(grid)); od = tp(F[ci], Y, F[ti])
            # grid values as the game stores them: mean and std = (q95 - q05) / 3.29
            vals = np.stack([og["dvx"][1], og["dvy"][1], (og["dvx"][2] - og["dvx"][0]) / 3.29, (og["dvy"][2] - og["dvy"][0]) / 3.29], 1)
            V = vals.reshape(5, 5, 5, 5, 4)
            interp = np.array([_multilinear(V, gv, gu, F[i]) for i in ti])
            for label, (qx, qy) in {"grid_surrogate": (_gauss(interp[:, 0], interp[:, 2]), _gauss(interp[:, 1], interp[:, 3])),
                                    "tabpfn_direct": (od["dvx"], od["dvy"])}.items():
                ex, cx, wx, _ = metrics(dv[ti, 0], qx); ey, cy, wy, _ = metrics(dv[ti, 1], qy)
                for j, i in enumerate(ti):
                    rows.append({"scenario": d["scenario"], "seed": d["seed"], "b": b, "i": int(i), "model": label,
                                 "err": float(np.hypot(qx[1][j] - dv[i, 0], qy[1][j] - dv[i, 1])), "cov90": float((cx[j] + cy[j]) / 2),
                                 "width": float((wx[j] + wy[j]) / 2), "pos_x": float(F[i, 0]), "pos_y": float(F[i, 1]),
                                 "u_change": float(np.hypot(F[i, 4] - F[i, 6], F[i, 5] - F[i, 7]))})
        print(f"[grid] {name}", flush=True)
    pd.DataFrame(rows).to_csv(out("grid"), index=False)


def _multilinear(V, gv, gu, f):
    L = max(1.0, np.hypot(f[4], f[5])); u = (f[4] / L, f[5] / L)
    def cell(g, x):
        x = min(max(x, g[0]), g[-1]); fr = (x - g[0]) / (g[1] - g[0]); i = min(len(g) - 2, int(np.floor(fr)))
        return i, fr - i
    (a, wa), (b, wb), (i, wi), (j, wj) = cell(gv, f[2]), cell(gv, f[3]), cell(gu, u[0]), cell(gu, u[1])
    out = np.zeros(4)
    for da in (0, 1):
        for db in (0, 1):
            for di in (0, 1):
                for dj in (0, 1):
                    w = (wa if da else 1 - wa) * (wb if db else 1 - wb) * (wi if di else 1 - wi) * (wj if dj else 1 - wj)
                    out += w * V[a + da, b + db, i + di, j + dj]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["landing", "dynamics", "curves", "grid", "all"], required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-estimators", type=int, default=None, help="TabPFN ensemble size (default: TabPFN's own)")
    ap.add_argument("--only", default=None, help="comma-separated substrings of data file names to include")
    ap.add_argument("--tag", default="", help="suffix for the result files")
    a = ap.parse_args()
    TabPFN.n_estimators, ONLY, OUT_TAG = a.n_estimators, a.only, a.tag
    t0 = time.time()
    parts = ["landing", "dynamics", "curves", "grid"] if a.part == "all" else [a.part]
    for p in parts:
        {"landing": part_landing, "dynamics": part_dynamics, "curves": part_curves, "grid": part_grid}[p](a.device)
    if TabPFN.latency:
        f = os.path.join(RES, f"latency_{a.part}{OUT_TAG}.csv")
        pd.DataFrame(TabPFN.latency).to_csv(f, index=False)
    print(f"[{a.part}] done in {(time.time() - t0) / 60:.1f} min, {len(TabPFN.latency)} new TabPFN calls", flush=True)
