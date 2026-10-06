"""Figures and numbers for the adaptation experiments (X1 worlds, X2 switches, X3 habits, X5 uncertainty).

    .venv/bin/python experiments/adaptation/analyze_adapt.py
Writes experiments/results/adapt_summary.json and experiments/figures/x*.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
RES, FIG = os.path.join(EXP, "results"), os.path.join(EXP, "figures")
os.makedirs(FIG, exist_ok=True)
RNG = np.random.default_rng(0)
WORLD_NAMES = {"normal": "Normal", "heavy": "Heavy shuttle", "feather": "Feather shuttle", "moon": "Low gravity",
               "gale": "Steady gale", "storm": "Gusty storm"}
STYLE_NAMES = {"crosscourt": "Cross-court", "dropper": "Drop artist", "smasher": "Smasher", "away": "Plays away",
               "pattern": "Fixed pattern", "random": "Random (control)"}
NAMES = {"tabpfn": "TabPFN-3.5", "tabpfn_hybrid": "Physics guess + TabPFN-3.5 correction", "tabpfn_age": "TabPFN-3.5 + shot-age feature", "tabpfn_frozen": "TabPFN-3.5, first world only",
         "tabpfn_normal_ctx": "TabPFN-3.5, normal-world context", "gbr": "Gradient boosting", "gbr_age": "Gradient boosting + age",
         "rf": "Random forest", "knn": "kNN", "ridge": "Ridge (linear)", "linear": "Linear (ridge / logistic)", "prior": "Average / most frequent",
         "physics_normal": "Physics tuned for the normal world"}
COLORS = {"tabpfn": "#2563eb", "tabpfn_hybrid": "#0f766e", "tabpfn_age": "#7c3aed", "tabpfn_frozen": "#94a3b8", "tabpfn_normal_ctx": "#94a3b8",
          "gbr": "#ea580c", "gbr_age": "#f59e0b", "rf": "#a16207", "knn": "#65a30d", "ridge": "#0d9488", "linear": "#0d9488",
          "prior": "#6b7280", "physics_normal": "#b91c1c"}
WORLDS = list(WORLD_NAMES)


def boot(x, stat=np.mean, n=2000):
    x = np.asarray(x, float)
    b = [stat(x[RNG.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name), dpi=150); plt.close(fig)


S = {}

# ============================================================================ X1 shuttle worlds + X5 uncertainty
f = os.path.join(RES, "adapt_x1.csv.gz")
if os.path.exists(f):
    X1 = pd.read_csv(f)
    t = X1.groupby(["world", "n", "model"]).agg(median_err=("err", "median"), mean_err=("err", "mean"), cov90=("cov90", "mean"),
                                                width=("width", "mean"), n_pts=("err", "size")).reset_index()
    S["x1"] = t.to_dict("records")
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, w in zip(axs.ravel(), WORLDS):
        g = t[t.world == w]
        for m in ("tabpfn", "tabpfn_hybrid", "gbr", "rf", "knn"):
            a = g[g.model == m].sort_values("n")
            ax.plot(a.n, a.median_err, "-o", ms=4, color=COLORS[m], label=NAMES[m], lw=2.4 if m.startswith("tabpfn") else 1.4)
        for m in ("physics_normal", "tabpfn_normal_ctx"):
            a = g[g.model == m]
            if len(a):
                ax.axhline(a.median_err.iloc[0], color=COLORS[m], ls="--", lw=1.4, label=NAMES[m] + " (does not adapt)")
        ax.set_xscale("log"); ax.set_xticks([5, 10, 20, 40, 120]); ax.set_xticklabels([5, 10, 20, 40, 120]); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(WORLD_NAMES[w]); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)
    for ax in axs[1]:
        ax.set_xlabel("Shots seen in this world (context rows)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Median landing error at 0.4 s (m)")
    h, l = axs[0, 1].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=6, fontsize=9, frameon=False)
    fig.suptitle("X1 · One model, six shuttle worlds: error vs number of shots seen", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96)); fig.savefig(os.path.join(FIG, "x1_worlds.png"), dpi=150); plt.close(fig)
    # "shots needed": smallest n at which TabPFN beats the non-adapting physics, per world
    need = []
    for w in WORLDS:
        g = t[t.world == w]; ph = g[g.model == "physics_normal"].median_err
        tp = g[g.model == "tabpfn"].sort_values("n")
        beat = tp[tp.median_err < ph.iloc[0]] if len(ph) else tp.iloc[0:0]
        gb = g[g.model == "gbr"].sort_values("n")
        need.append({"world": w, "physics_normal": float(ph.iloc[0]) if len(ph) else None,
                     "tabpfn_n5": float(tp[tp.n == 5].median_err.iloc[0]), "tabpfn_n20": float(tp[tp.n == 20].median_err.iloc[0]),
                     "tabpfn_n120": float(tp[tp.n == 120].median_err.iloc[0]),
                     "gbr_n20": float(gb[gb.n == 20].median_err.iloc[0]), "gbr_n120": float(gb[gb.n == 120].median_err.iloc[0]),
                     "normal_ctx": float(g[g.model == "tabpfn_normal_ctx"].median_err.iloc[0]) if (g.model == "tabpfn_normal_ctx").any() else None,
                     "tabpfn_beats_physics_from_n": int(beat.n.min()) if len(beat) else None,
                     **{f"hybrid_n{k}": float(g[(g.model == "tabpfn_hybrid") & (g.n == k)].median_err.iloc[0]) for k in (5, 20, 120)
                        if ((g.model == "tabpfn_hybrid") & (g.n == k)).any()}})
    S["x1_need"] = need
    # X5: 90 % coverage per world (n = 20 and 120)
    cov = X1[X1.n.isin([20, 120]) & X1.model.isin(["tabpfn", "gbr", "rf", "knn", "physics_normal"])]
    S["x5"] = cov.groupby(["n", "model", "world"]).agg(cov90=("cov90", "mean"), width=("width", "mean")).reset_index().to_dict("records")
    c = cov[cov.n == 120].groupby(["model", "world"])["cov90"].mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    order = ["tabpfn", "gbr", "rf", "knn", "physics_normal"]; wdt = 0.16
    for j, m in enumerate(order):
        ax.bar(np.arange(len(WORLDS)) + (j - 2) * wdt, [c.loc[m, w] for w in WORLDS], wdt, color=COLORS[m], label=NAMES[m])
    ax.axhline(0.9, color="k", ls=":", lw=1); ax.text(len(WORLDS) - 0.5, 0.91, "target 90 %", ha="right", fontsize=9)
    ax.set_xticks(range(len(WORLDS))); ax.set_xticklabels([WORLD_NAMES[w] for w in WORLDS]); ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share inside the 90 % interval"); ax.legend(fontsize=8, ncol=5, loc="lower center"); ax.grid(alpha=0.3, axis="y")
    ax.set_title("X5 · Are the uncertainty intervals honest in every world? (120 shots of context)")
    save(fig, "x5_coverage.png")

# ============================================================================ X2 sudden world switches
f = os.path.join(RES, "adapt_x2.csv.gz")
if os.path.exists(f):
    X2 = pd.read_csv(f)
    X2["since"] = X2.shot % 50
    X2["block"] = X2.shot // 50
    after = X2[X2.block >= 1]                         # every switch (the first block has no switch)
    curve = after.groupby(["model", "since"])["err"].mean().reset_index()
    rows = []
    for m, g in after.groupby("model"):
        first = g[g.since < 20]["err"]; steady = g[g.since >= 30]["err"]
        cg = curve[curve.model == m].sort_values("since"); roll = cg.err.rolling(5, min_periods=3).mean()
        ok = cg.since[(roll <= 1.2 * steady.mean()).values]
        rows.append({"model": m, "first20_mean": float(first.mean()), "first20_ci": boot(first), "steady_mean": float(steady.mean()),
                     "steady_ci": boot(steady), "first10_mean": float(g[g.since < 10]["err"].mean()),
                     "recovery_shots": int(ok.iloc[0]) if len(ok) else None})
    S["x2"] = rows
    # paired: age feature vs plain TabPFN on the same shots
    def paired(a, b, sub):
        A = sub[sub.model == a].set_index(["seed", "shot"])["err"]; B = sub[sub.model == b].set_index(["seed", "shot"])["err"]
        j = A.index.intersection(B.index); d = (A.loc[j] - B.loc[j]).values; lo, hi = boot(d)
        return {"a": a, "b": b, "diff": float(d.mean()), "lo": lo, "hi": hi, "n": int(len(d))}
    S["x2_paired"] = [{"window": "first 20 after a switch", **paired("tabpfn_age", "tabpfn", after[after.since < 20])},
                      {"window": "steady (30-49)", **paired("tabpfn_age", "tabpfn", after[after.since >= 30])},
                      {"window": "first 20 after a switch", **paired("tabpfn", "gbr", after[after.since < 20])},
                      {"window": "first 20 after a switch", **paired("tabpfn_hybrid", "tabpfn", after[after.since < 20])},
                      {"window": "steady (30-49)", **paired("tabpfn_hybrid", "tabpfn", after[after.since >= 30])},
                      {"window": "steady (30-49)", **paired("tabpfn", "gbr", after[after.since >= 30])}]
    S["x2_by_world"] = after.groupby(["model", "block"]).apply(lambda g: pd.Series({"first20": g[g.since < 20].err.mean(), "steady": g[g.since >= 30].err.mean()})).reset_index().to_dict("records")
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axs[0]
    for m in ("tabpfn", "tabpfn_age", "tabpfn_hybrid", "gbr", "tabpfn_frozen", "physics_normal"):
        g = curve[curve.model == m].sort_values("since")
        ax.plot(g.since, g.err.rolling(5, min_periods=1).mean(), "-" if "physics" not in m else "--", color=COLORS[m], label=NAMES[m], lw=2.4 if m.startswith("tabpfn") and m != "tabpfn_frozen" else 1.4)
    ax.set_xlabel("Shots since the shuttle world changed"); ax.set_ylabel("Landing error at 0.4 s (m), rolling mean of 5")
    ax.set_title("X2 · Recovery after a world switch (average of all switches)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax = axs[1]
    s1 = X2[X2.seed == 1]; order = json.load(open(os.path.join(EXP, "data_adapt", "x2_switch_s1.json")))["order"]
    for m in ("tabpfn", "tabpfn_age", "tabpfn_hybrid", "gbr", "physics_normal"):
        g = s1[s1.model == m].groupby("shot")["err"].mean().rolling(8, min_periods=2).mean()
        ax.plot(g.index, g.values, "-" if "physics" not in m else "--", color=COLORS[m], label=NAMES[m], lw=2 if m.startswith("tabpfn") else 1.3)
    for b, w in enumerate(order):
        ax.axvspan(b * 50, b * 50 + 50, color="#000" if b % 2 else "#fff", alpha=0.04)
        ax.text(b * 50 + 25, ax.get_ylim()[1] * 0.95, WORLD_NAMES[w], ha="center", fontsize=8)
    ax.set_xlabel("Shot number (seed 1)"); ax.set_ylabel("Landing error (m), rolling mean of 8"); ax.grid(alpha=0.3)
    ax.set_title("X2 · One sequence: the world changes every 50 shots, the model is never told")
    save(fig, "x2_switches.png")

# ============================================================================ X3 opponent habits
f = os.path.join(RES, "adapt_x3.csv.gz")
if os.path.exists(f):
    X3 = pd.read_csv(f)
    t = X3.groupby(["personality", "n", "model"]).agg(median_err=("err", "median"), mean_err=("err", "mean"), cov90=("cov90", "mean"),
                                                      type_acc=("type_correct", "mean"), logloss=("type_logloss", "mean")).reset_index()
    S["x3"] = t.to_dict("records")
    styles = list(STYLE_NAMES)
    for metric, ylabel, name, title in (("median_err", "Median error of the next-shot landing guess (m)", "x3_habits_where.png", "where the next shot lands"),
                                        ("type_acc", "Shot-type accuracy", "x3_habits_type.png", "which shot comes next (clear / drop / smash)")):
        fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
        for ax, p in zip(axs.ravel(), styles):
            g = t[t.personality == p]
            for m in ("tabpfn", "gbr", "knn", "linear", "prior"):
                a = g[g.model == m].sort_values("n")
                ax.plot(a.n, a[metric], "-o" if m != "prior" else "--", ms=4, color=COLORS[m], label=NAMES[m], lw=2.4 if m == "tabpfn" else 1.4)
            ax.set_xscale("log"); ax.set_xticks([5, 10, 20, 40, 80]); ax.set_xticklabels([5, 10, 20, 40, 80]); ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_title(STYLE_NAMES[p]); ax.grid(alpha=0.3)
            if metric == "type_acc":
                ax.set_ylim(0, 1.02)
        for ax in axs[1]:
            ax.set_xlabel("Shots of this opponent seen")
        fig.supylabel(ylabel, fontsize=11)
        h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=5, fontsize=9, frameon=False)
        fig.suptitle(f"X3 · Learning an opponent's habits: {title}", fontsize=13)
        fig.tight_layout(rect=(0, 0.05, 1, 0.96)); fig.savefig(os.path.join(FIG, name), dpi=150); plt.close(fig)

json.dump(S, open(os.path.join(RES, "adapt_summary.json"), "w"), indent=1, default=float)
print("wrote adapt_summary.json and", sorted(f for f in os.listdir(FIG) if f.startswith("x")))
