"""Tables, confidence intervals and figures from experiments/results/*.csv.gz.

    .venv/bin/python experiments/physics_comparison/analyze.py
Writes experiments/results/summary.json and experiments/figures/*.png.
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
RES, FIG = os.path.join(EXP, "results"), os.path.join(EXP, "figures")
os.makedirs(FIG, exist_ok=True)
RNG = np.random.default_rng(0)

NAMES = {
    "tabpfn": "TabPFN-3.5", "tabpfn_fast": "TabPFN-3.5-Fast", "tabpfn_hybrid": "Physics + TabPFN (hybrid)",
    "tabpfn_frozen": "TabPFN-3.5, frozen context", "gbr": "Gradient boosting", "gbr_hybrid": "Physics + gradient boosting",
    "rf": "Random forest", "knn": "kNN", "ridge": "Ridge (linear)", "rls": "RLS (online linear)",
    "phys_whitebox": "Physics: white-box trajectory fit", "phys_whitebox_wind": "Physics: white-box + wind correction",
    "phys_physics": "Physics: game's timing+drag fit", "phys_physics_wind": "Physics: timing+drag + wind correction",
    "phys_ekf": "Physics: extended Kalman filter", "phys_pf": "Physics: particle filter", "phys_linear": "Linear extrapolation",
    "phys_nominal": "Physics: exact equations, default parameters", "phys_calibrated": "Physics: exact equations, calibrated",
    "phys_calibrated_nolag": "Physics: calibrated, no lag term", "phys_simplified": "Physics: simplified first-order",
    "grid_surrogate": "In-game 625-point grid", "tabpfn_direct": "TabPFN asked directly",
}
COLORS = {"tabpfn": "#2563eb", "tabpfn_hybrid": "#7c3aed", "tabpfn_fast": "#60a5fa", "tabpfn_frozen": "#94a3b8",
          "gbr": "#ea580c", "gbr_hybrid": "#f59e0b", "rf": "#a16207", "knn": "#65a30d", "ridge": "#0d9488", "rls": "#0891b2",
          "phys_whitebox": "#111827", "phys_whitebox_wind": "#6b7280", "phys_physics": "#b91c1c", "phys_physics_wind": "#f87171",
          "phys_ekf": "#be185d", "phys_pf": "#db2777", "phys_nominal": "#b91c1c", "phys_calibrated": "#111827",
          "phys_calibrated_nolag": "#6b7280", "phys_simplified": "#f87171"}
LS = lambda m: "--" if m.startswith("phys") else "-"


def boot_ci(x, stat=np.mean, n=1000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    b = [stat(x[RNG.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def paired_diff(df, a, b, keys):
    """Mean error of a minus b on the same test points, with a 95% bootstrap CI. Negative = a is better."""
    A = df[df.model == a].set_index(keys)["err"]; B = df[df.model == b].set_index(keys)["err"]
    j = A.index.intersection(B.index)
    d = (A.loc[j] - B.loc[j]).values
    lo, hi = boot_ci(d)
    return {"diff": float(d.mean()), "lo": lo, "hi": hi, "n": int(len(d)),
            "verdict": "better" if hi < 0 else "worse" if lo > 0 else "tie"}


def summarize(df, by):
    g = df.groupby(by)
    out = g.agg(median_err=("err", "median"), mean_err=("err", "mean"), cov90=("cov90", "mean"),
                width=("width", "mean"), pinball=("pinball", "mean"), n=("err", "size")).reset_index()
    ci = g["err"].apply(lambda x: boot_ci(x, np.median)).reset_index(name="ci")
    out["median_lo"] = [c[0] for c in ci["ci"]]; out["median_hi"] = [c[1] for c in ci["ci"]]
    return out


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name), dpi=150); plt.close(fig)


S = {}

# ============================================================================ landing E1, E2, E4
if os.path.exists(os.path.join(RES, "landing.csv.gz")):
    L = pd.read_csv(os.path.join(RES, "landing.csv.gz"))
    keys = ["scenario", "seed", "k", "shot"]
    stat = L[L.scenario != "windshift"]
    tab = summarize(stat, ["scenario", "checkpoint", "model"])
    S["landing_table"] = tab.to_dict("records")
    # win/loss: TabPFN and hybrid vs the best physics model per scenario+checkpoint (best = lowest mean error)
    wl = []
    for (sc, c), g in stat.groupby(["scenario", "checkpoint"]):
        phys = g[g.model.str.startswith("phys_") & (g.model != "phys_linear")].groupby("model")["err"].mean()
        best = phys.idxmin()
        for m in ("tabpfn", "tabpfn_hybrid", "gbr", "gbr_hybrid"):
            r = paired_diff(g, m, best, keys); wl.append({"scenario": sc, "checkpoint": c, "model": m, "vs": best, **r})
    S["landing_winloss"] = wl
    # E1 figure: median error vs wind strength, one panel per checkpoint
    show = ["tabpfn", "tabpfn_hybrid", "gbr", "gbr_hybrid", "knn", "phys_whitebox", "phys_whitebox_wind", "phys_physics", "phys_ekf", "phys_pf"]
    wind = tab[tab.scenario.str.match(r"wind\d")].copy(); wind["w"] = wind.scenario.str[-1].astype(int)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, c in zip(axs, sorted(wind.checkpoint.unique())):
        for m in show:
            g = wind[(wind.model == m) & (wind.checkpoint == c)].sort_values("w")
            ax.plot(g.w, g.median_err, LS(m), marker="o", color=COLORS[m], label=NAMES[m], lw=2 if "tabpfn" in m else 1.4)
            ax.fill_between(g.w, g.median_lo, g.median_hi, color=COLORS[m], alpha=0.08)
        ax.set_title(f"Prediction at {c:.1f} s after the hit"); ax.set_xlabel("Wind strength (0 = calm)"); ax.set_xticks([0, 1, 2, 3]); ax.grid(alpha=0.3)
    axs[0].set_ylabel("Median landing error (m)")
    axs[-1].legend(fontsize=8, loc="upper left")
    save(fig, "e1_landing_vs_wind.png")
    # E4 regime change: rolling error vs shot index at checkpoint 0.4 s
    ws = L[(L.scenario == "windshift") & (L.k == 1)]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for m in ["tabpfn", "tabpfn_hybrid", "gbr", "phys_whitebox", "phys_whitebox_wind", "phys_physics_wind"]:
        g = ws[ws.model == m].groupby("shot")["err"].mean().rolling(25, min_periods=5).mean()
        ax.plot(g.index, g.values, LS(m), color=COLORS[m], label=NAMES[m], lw=2 if "tabpfn" in m else 1.4)
    ax.axvline(200, color="k", lw=1, ls=":"); ax.text(203, ax.get_ylim()[1] * 0.92, "wind reverses", fontsize=9)
    ax.set_xlabel("Shot number"); ax.set_ylabel("Landing error (m), rolling mean of 25 shots"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Wind regime change (prediction at 0.4 s)")
    save(fig, "e4_wind_shift.png")
    blocks = {"before (120-199)": (120, 200), "just after (200-239)": (200, 240), "after (240-319)": (240, 320), "late (320-399)": (320, 400)}
    S["windshift"] = [{"model": m, "block": b, "mean_err": float(ws[(ws.model == m) & ws.shot.between(lo, hi - 1)]["err"].mean())}
                      for m in ws.model.unique() for b, (lo, hi) in blocks.items()]
    # by shot type (E1 detail)
    S["landing_by_type"] = stat[stat.scenario.isin(["wind1", "wind3"])].groupby(["scenario", "type", "model"])["err"].median().reset_index().to_dict("records")

# ============================================================================ dynamics E5, E6
if os.path.exists(os.path.join(RES, "dynamics.csv.gz")):
    D = pd.read_csv(os.path.join(RES, "dynamics.csv.gz"))
    keys = ["scenario", "seed", "i"]
    S["dynamics_table"] = summarize(D, ["scenario", "model"]).to_dict("records")
    court = D[D.scenario == "court"].copy(); court["tb"] = (court.t // 3) * 3
    curve = court.groupby(["model", "tb"])["err"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for m in ["tabpfn", "tabpfn_hybrid", "tabpfn_frozen", "gbr", "rls", "phys_nominal", "phys_calibrated", "phys_simplified"]:
        g = curve[curve.model == m]
        ax.plot(g.tb, g.err, LS(m), color=COLORS[m], label=NAMES[m], lw=2 if "tabpfn" in m else 1.4)
    ax.axvline(60, color="k", lw=1, ls=":"); ax.text(61, 1.05, "court turns slippery", fontsize=9)
    ax.set_yscale("symlog", linthresh=0.05); ax.set_ylim(0, 3)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Velocity-change error (m/s), 3 s bins"); ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
    ax.set_title("Court change: movement prediction error over time")
    save(fig, "e5_court_change.png")
    rec = []
    for m, g in curve.groupby("model"):
        pre = g[g.tb.between(30, 57)]["err"].mean()
        post = g[g.tb >= 60].sort_values("tb")
        ok = post[post.err <= 1.2 * pre + 0.02]
        rec.append({"model": m, "pre_err": float(pre), "err_60_75": float(g[g.tb.between(60, 72)]["err"].mean()),
                    "err_75_105": float(g[g.tb.between(75, 102)]["err"].mean()), "err_105_180": float(g[g.tb >= 105]["err"].mean()),
                    "recovery_s": float(ok.tb.iloc[0] + 3 - 60) if len(ok) else None})
    S["court_recovery"] = rec
    # E6: scenario comparison (skip the first 20 s of each run)
    st = D[D.t >= 20]
    tab = summarize(st, ["scenario", "model"])
    order = ["phys_calibrated", "phys_calibrated_nolag", "phys_nominal", "phys_simplified", "tabpfn", "tabpfn_hybrid", "gbr", "gbr_hybrid", "rf", "knn", "ridge", "rls"]
    scen = ["default", "lag", "slippery", "wetpatch", "court"]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    w = 0.8 / len(order)
    for j, m in enumerate(order):
        vals = [tab[(tab.scenario == s) & (tab.model == m)]["mean_err"].mean() for s in scen]
        ax.bar(np.arange(len(scen)) + j * w - 0.4 + w / 2, vals, w, color=COLORS[m], label=NAMES[m], hatch="//" if m.startswith("phys") else None, edgecolor="white")
    ax.set_xticks(range(len(scen))); ax.set_xticklabels(["default court", "command lag 0.1 s", "low friction", "wet patch (left side)", "court change at 60 s"])
    ax.set_yscale("log"); ax.set_ylabel("Mean velocity-change error (m/s, log)"); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=7, ncol=3)
    ax.set_title("Movement prediction by scenario (t ≥ 20 s)")
    save(fig, "e6_dynamics_scenarios.png")
    wl = []
    for sc, g in st.groupby("scenario"):
        for m in ("tabpfn", "tabpfn_hybrid"):
            for p in ("phys_calibrated", "phys_nominal", "phys_simplified", "phys_calibrated_nolag"):
                wl.append({"scenario": sc, "model": m, "vs": p, **paired_diff(g, m, p, keys)})
    S["dynamics_winloss"] = wl

# ============================================================================ learning curves E3
if os.path.exists(os.path.join(RES, "curves.csv.gz")):
    C = pd.read_csv(os.path.join(RES, "curves.csv.gz"))
    S["curves"] = C.groupby(["task", "scenario", "n", "model"]).agg(median_err=("err", "median"), mean_err=("err", "mean"), cov90=("cov90", "mean")).reset_index().to_dict("records")
    panels = [("landing", "wind1", "Landing at 0.4 s (wind 1)", "Median landing error (m)", "median_err"),
              ("dynamics", "lag", "Movement, command lag 0.1 s", "Mean dv error (m/s)", "mean_err"),
              ("dynamics", "wetpatch", "Movement, wet patch", "Mean dv error (m/s)", "mean_err")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (task, sc, title, yl, col) in zip(axs, panels):
        g = C[(C.task == task) & (C.scenario == sc)]
        agg = g.groupby(["model", "n"])["err"].agg(["median", "mean"]).reset_index()
        for m in agg.model.unique():
            if m in ("phys_linear", "phys_ekf", "phys_pf", "phys_physics_wind", "rf", "ridge", "phys_calibrated_nolag"):
                continue
            a = agg[agg.model == m].sort_values("n")
            ax.plot(a.n, a["median" if col == "median_err" else "mean"], LS(m), marker="o", ms=3, color=COLORS.get(m, "#999"), label=NAMES.get(m, m), lw=2 if "tabpfn" in m else 1.3)
        ax.set_xscale("log"); ax.set_title(title); ax.set_xlabel("Rows in the context / training set"); ax.set_ylabel(yl); ax.grid(alpha=0.3)
        if task == "dynamics":
            ax.set_yscale("symlog", linthresh=0.01)
    axs[0].legend(fontsize=7)
    save(fig, "e3_learning_curves.png")

# ============================================================================ grid vs direct E7b
if os.path.exists(os.path.join(RES, "grid.csv.gz")):
    Gd = pd.read_csv(os.path.join(RES, "grid.csv.gz"))
    Gd["far"] = np.hypot(Gd.pos_x, Gd.pos_y + 3.3) > 1.5
    Gd["turning"] = Gd.u_change > 0.3
    S["grid"] = Gd.groupby(["model"]).agg(mean_err=("err", "mean"), cov90=("cov90", "mean"), width=("width", "mean")).reset_index().to_dict("records")
    S["grid_by_scenario"] = Gd.groupby(["scenario", "model"]).agg(mean_err=("err", "mean"), cov90=("cov90", "mean")).reset_index().to_dict("records")
    S["grid_breakdown"] = Gd.groupby(["model", "far", "turning"]).agg(mean_err=("err", "mean"), cov90=("cov90", "mean"), n=("err", "size")).reset_index().to_dict("records")

# ============================================================================ cost E8
lat = [pd.read_csv(os.path.join(RES, f)) for f in os.listdir(RES) if f.startswith("latency_")]
if lat:
    lat = pd.concat(lat)
    S["latency"] = lat.groupby(["model", "device"]).agg(median_s=("seconds", "median"), p90_s=("seconds", lambda x: np.percentile(x, 90)),
                                                         train_rows=("train_rows", "median"), test_rows=("test_rows", "median"), calls=("seconds", "size")).reset_index().to_dict("records")
sys.path.insert(0, HERE)
from physics_models import NOMINAL, calibrate, step_dv, whitebox_landing  # noqa: E402
d = json.load(open(os.path.join(EXP, "data", "landing_wind1_s1.json"))); s = d["shots"][50]
t0 = time.perf_counter()
for _ in range(200):
    whitebox_landing(s["samples"], s["p0"], d["shotParams"][s["type"]], 0.4)
wb_ms = (time.perf_counter() - t0) / 200 * 1000
dd = json.load(open(os.path.join(EXP, "data", "dynamics_court_s1.json")))
F = np.array([t["x"] for t in dd["trans"]][:300], float); dv = np.array([t["dv"] for t in dd["trans"]][:300], float)
t0 = time.perf_counter(); step_dv(F, NOMINAL, dd["bounds"]); step_ms = (time.perf_counter() - t0) * 1000
t0 = time.perf_counter(); calibrate(F, dv, dd["bounds"]); cal_ms = (time.perf_counter() - t0) * 1000
S["physics_cost_ms"] = {"whitebox_landing_per_shot": wb_ms, "nominal_dynamics_300_rows": step_ms, "calibration_300_rows": cal_ms}

json.dump(S, open(os.path.join(RES, "summary.json"), "w"), indent=1, default=float)
print("wrote", os.path.join(RES, "summary.json"), "and figures:", sorted(os.listdir(FIG)))
