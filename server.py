"""Local server for the badminton game + TabPFN inference.

    pip install -r requirements.txt
    python server.py                         # local TabPFN-3.5, device chosen automatically
    python server.py --model v3.5-fast       # another checkpoint from Prior-Labs/tabpfn_3_5
    python server.py --backend api           # Prior Labs hosted API, pinned to TabPFN-3.5
    # then open http://localhost:8000

The BotController in web/badminton.html posts its in-context table to /api/predict here. Each
request is fully self-contained (training rows + query rows), so this server keeps no game state.

Two inference backends, both the real TabPFN model (outputs are tagged backend "tabpfn"):
  local  The `tabpfn` package runs the open weights from huggingface.co/Prior-Labs/tabpfn_3_5 on
         this machine. Device "auto" picks CUDA, then Apple MPS, then CPU. The first run downloads
         the checkpoint (~0.9 GB, ~0.3 GB for Fast) after the one-time licence acceptance.
  api    The `tabpfn-client` package calls the Prior Labs cloud API, pinned to TabPFN-3.5
         (model_path "v3.5_default"). Uses quota: ~10k tokens per call, 60 calls/minute.

Environment (.env next to this file; command-line flags override it):
    TABPFN_TOKEN=...            # Prior Labs token: licence check (local) and API access (api)
    TABPFN_BACKEND=local        # local | api
    TABPFN_MODEL=v3.5           # v3.5 | v3.5-fast | v3.5-multiclass (local only; api is always v3.5)
    TABPFN_DEVICE=auto          # auto | mps | cuda | cpu (local only)
    TABPFN_N_ESTIMATORS=4       # optional, fewer = faster
    TABPFN_GRID_ESTIMATORS=2    # local: ensemble size for the large background movement-grid call (0 = default)
    TABPFN_FALLBACK=local       # local (default): if the API is unavailable, use TabPFN-3.5 from Hugging Face
                                # on this machine instead; none: pause until the API is back
    TABPFN_MODEL_CACHE_DIR=...  # optional, where local weights are stored
    TABPFN_MOCK=1               # optional, use a local k-NN stand-in (refused by play mode)
    PORT=8000                   # optional
"""
import argparse
import hashlib
import importlib.util
import json
import os
import socket
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(ROOT, "web")          # the game page and the developer module
QUANTILES = [0.1, 0.5, 0.9]
MAX_TRAIN_ROWS = 500
MAX_TEST_ROWS = 1024

# The three checkpoints in huggingface.co/Prior-Labs/tabpfn_3_5. Each one serves both
# classification and regression, so all of them work with TabPFNRegressor.
MODELS = OrderedDict([
    ("v3.5", {"name": "TabPFN-3.5", "short": "TabPFN-3.5", "file": "tabpfn-v3.5-20260909.safetensors"}),
    ("v3.5-fast", {"name": "TabPFN-3.5-Fast", "short": "TabPFN-3.5-Fast", "file": "tabpfn-v3.5-fast-20260909.safetensors"}),
    ("v3.5-multiclass", {"name": "TabPFN-3.5 (multiclass variant)", "short": "TabPFN-3.5 multiclass",
                         "file": "tabpfn-v3.5-20260909_multiclass.safetensors"}),
])
API_MODEL = "v3.5"          # the hosted API is always pinned to TabPFN-3.5
INFERENCES = ("local", "api")
DEVICES = ("auto", "mps", "cuda", "cpu")


def load_env(path):
    """Minimal .env parser: KEY=VALUE lines, # comments, optional quotes."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            val = val.strip().strip('"').strip("'")
            os.environ.setdefault(key.strip(), val)


load_env(os.path.join(ROOT, ".env"))
# Keep up to 3 built models (one per checkpoint) on the device, so switching models is fast.
os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", "3")
TOKEN = os.environ.get("TABPFN_TOKEN", "").strip()
MOCK = os.environ.get("TABPFN_MOCK", "0").strip() == "1"
PORT = int(os.environ.get("PORT", "8000"))
N_ESTIMATORS = os.environ.get("TABPFN_N_ESTIMATORS", "").strip()
# The movement grid (one large, low-priority call: ~300 rows -> 625 queries) uses a smaller TabPFN-3.5 ensemble
# locally, so it holds the GPU for ~2 s instead of ~9 s and urgent landing / shot calls are not stuck behind it.
GRID_ESTIMATORS = int(os.environ.get("TABPFN_GRID_ESTIMATORS", "2") or 0)
GRID_MIN_QUERIES = 200

# Availability is checked without importing: `tabpfn` pulls in torch and `tabpfn_client` takes
# ~10 s, so each is imported only when its backend is first used.
HAVE_LOCAL = importlib.util.find_spec("tabpfn") is not None
HAVE_CLIENT = importlib.util.find_spec("tabpfn_client") is not None

# Active configuration. Switched at runtime by the developer-only POST /api/config.
CFG = {
    "inference": os.environ.get("TABPFN_BACKEND", "local").strip().lower(),
    "model": os.environ.get("TABPFN_MODEL", "v3.5").strip().lower(),
    "device": os.environ.get("TABPFN_DEVICE", "auto").strip().lower(),
}
_cfg_lock = threading.Lock()


def validate_cfg(cfg):
    if cfg["inference"] not in INFERENCES:
        raise ValueError(f"backend must be one of {', '.join(INFERENCES)}")
    if cfg["model"] not in MODELS:
        raise ValueError(f"model must be one of {', '.join(MODELS)}")
    if cfg["device"] not in DEVICES:
        raise ValueError(f"device must be one of {', '.join(DEVICES)}")


def current_cfg():
    """Snapshot of the active configuration. The API ignores the model setting (always TabPFN-3.5)."""
    with _cfg_lock:
        cfg = dict(CFG)
    if cfg["inference"] == "api":
        cfg["model"] = API_MODEL
    return cfg


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
_client = None
_client_lock = threading.Lock()


def _api_client():
    """Import tabpfn-client once and set the token."""
    global _client
    with _client_lock:
        if _client is None:
            import tabpfn_client
            tabpfn_client.set_access_token(TOKEN)
            _client = tabpfn_client
    return _client


class PriorityLock:
    """One local forward pass at a time; urgent requests (landing, shot choice: "high") go before background
    ones (movement grid, next-shot guess: "low"). A running call is never interrupted."""
    def __init__(self):
        self._cond = threading.Condition()
        self._busy = False
        self._high_waiting = 0

    def acquire(self, priority="high"):
        with self._cond:
            high = priority != "low"
            self._high_waiting += high
            while self._busy or (not high and self._high_waiting):
                self._cond.wait()
            self._high_waiting -= high
            self._busy = True

    def release(self):
        with self._cond:
            self._busy = False
            self._cond.notify_all()

    def __call__(self, priority="high"):
        lock = self

        class _Ctx:
            def __enter__(self):
                lock.acquire(priority)

            def __exit__(self, *exc):
                lock.release()
        return _Ctx()


# torch on one GPU should not run several forward passes from different threads at once.
_local_lock = PriorityLock()
# Warm-up state per (model, device): {"state": "loading" | "ready" | "error", "msg", "device"}.
_warm = {}
_warm_lock = threading.Lock()


def _local_regressor(cfg):
    from tabpfn import TabPFNRegressor
    from tabpfn.model_loading import prepend_cache_path
    kwargs = {"model_path": prepend_cache_path(MODELS[cfg["model"]]["file"]),
              "device": cfg["device"], "random_state": 0}
    if cfg.get("n_estimators") or N_ESTIMATORS:
        kwargs["n_estimators"] = int(cfg.get("n_estimators") or N_ESTIMATORS)
    return TabPFNRegressor(**kwargs)


def _api_regressor():
    kwargs = {}
    if N_ESTIMATORS:
        kwargs["n_estimators"] = int(N_ESTIMATORS)
    return _api_client().TabPFNRegressor.create_default_for_version(API_MODEL, **kwargs)


def _resolved_device(model):
    devs = getattr(model, "devices_", None)
    return devs[0].type if devs else None


def _friendly_error(e):
    """Turn the usual local-setup failures into one actionable sentence."""
    msg = f"{type(e).__name__}: {e}"
    low = msg.lower()
    if "licen" in low or "gated" in low or "401" in low or "403" in low:
        return ("TabPFN licence not accepted: put your Prior Labs token in .env as TABPFN_TOKEN "
                "(or accept the licence in the browser window tabpfn opens) and restart. " + msg)
    if "huggingface" in low or "connection" in low or "resolve" in low:
        return "Could not download the TabPFN weights (internet needed on the first run). " + msg
    return msg


def _warm_up(cfg):
    """Load (and on the first run download) a local checkpoint, then run one tiny prediction so the
    device kernels are compiled before the first real request. Requests wait on _local_lock."""
    key = (cfg["model"], cfg["device"])
    name = MODELS[cfg["model"]]["name"]
    with _warm_lock:
        if key in _warm and _warm[key]["state"] in ("loading", "ready"):
            return
        _warm[key] = {"state": "loading", "msg": f"Loading {name} (the first run downloads the weights)…",
                      "device": None}
    print(f"[local] loading {name} ({MODELS[cfg['model']]['file']}), device={cfg['device']} …", file=sys.stderr)
    t0 = time.time()
    try:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(24, 4))
        y = X @ np.array([1.0, -0.5, 0.25, 0.0])
        with _local_lock():
            model = _local_regressor(cfg).fit(X, y)
            model.predict(X[:4], output_type="quantiles", quantiles=QUANTILES)
        device = _resolved_device(model) or cfg["device"]
        with _warm_lock:
            _warm[key] = {"state": "ready", "msg": f"{name} running locally on {device.upper()}.", "device": device}
        print(f"[local] {name} ready on {device} in {time.time() - t0:.1f} s", file=sys.stderr)
    except Exception as e:  # reported via /api/health; the bot waits and offers Retry
        msg = _friendly_error(e)
        with _warm_lock:
            _warm[key] = {"state": "error", "msg": msg, "device": None}
        print(f"[local] {name} failed: {msg}", file=sys.stderr)


def start_warm_up(cfg):
    if MOCK:
        return
    if cfg["inference"] == "local" and HAVE_LOCAL:
        threading.Thread(target=_warm_up, args=(cfg,), daemon=True).start()
    elif cfg["inference"] == "api" and HAVE_CLIENT and TOKEN:
        threading.Thread(target=_api_client, daemon=True).start()


# ---------------------------------------------------------------------------
# API -> local fallback: if the Prior Labs API is chosen but can't be used (no token, client missing, quota or
# rate limit, auth or network errors), TabPFN-3.5 is downloaded from Hugging Face and runs on this machine.
# TABPFN_FALLBACK=none turns this off (the game then pauses until the API is back).
# ---------------------------------------------------------------------------
FALLBACK_ENABLED = os.environ.get("TABPFN_FALLBACK", "local").strip().lower() != "none"
FALLBACK = {"reason": None}
_API_DOWN = ("429", "rate limit", "usage limit", "quota", "401", "403", "unauthorized", "forbidden", "access token",
             "invalid token", "connection", "timed out", "timeout", "max retries", "name resolution", "unreachable",
             "temporarily unavailable", "service unavailable", "502", "503", "504", "no module named 'tabpfn_client'")


def api_unavailable(e):
    """True for errors that mean "the API can't be used right now" (not for bad requests)."""
    msg = f"{type(e).__name__}: {e}".lower()
    return isinstance(e, (ConnectionError, TimeoutError, ImportError)) or any(k in msg for k in _API_DOWN)


def fall_back_to_local(reason):
    """Switch from the API to TabPFN-3.5 from Hugging Face on this machine. Returns True if local inference is now active."""
    if not FALLBACK_ENABLED or not HAVE_LOCAL or MOCK:
        return False
    with _cfg_lock:
        if CFG["inference"] != "api":
            return True                       # another request already switched
        CFG["inference"], CFG["model"] = "local", API_MODEL     # the same model, TabPFN-3.5
    FALLBACK["reason"] = str(reason)[:200]
    print(f"[fallback] Prior Labs API unavailable ({FALLBACK['reason']}). "
          f"Switching to TabPFN-3.5 from Hugging Face, running locally.", file=sys.stderr)
    start_warm_up(current_cfg())
    return True


def status(cfg=None):
    """(provenance tag, ready, human-readable status) for a configuration."""
    cfg = cfg or current_cfg()
    name = MODELS[cfg["model"]]["name"]
    if MOCK:
        return "mock", True, "Server is in TABPFN_MOCK mode: play mode refuses non-TabPFN outputs."
    if cfg["inference"] == "api":
        if not HAVE_CLIENT:
            return "none", False, "tabpfn-client is not installed on the server (pip install -r requirements.txt)."
        if not TOKEN:
            return "none", False, "TABPFN_TOKEN is empty in .env (needed for the Prior Labs API)."
        return "tabpfn", True, f"{name} via the Prior Labs API (pinned v3.5)."
    if not HAVE_LOCAL:
        return "none", False, "The tabpfn package is not installed on the server (pip install -r requirements.txt)."
    with _warm_lock:
        w = _warm.get((cfg["model"], cfg["device"]))
    if w is None:
        return "tabpfn", False, f"{name} not loaded yet."
    if w["state"] == "error":
        return "none", False, w["msg"]
    return "tabpfn", w["state"] == "ready", w["msg"]


def describe(cfg=None):
    """Fields that say which model answered and where it ran (added to health and predict responses)."""
    cfg = cfg or current_cfg()
    m = MODELS[cfg["model"]]
    with _warm_lock:
        w = _warm.get((cfg["model"], cfg["device"])) or {}
    local = cfg["inference"] == "local"
    return {
        "inference": "mock" if MOCK else cfg["inference"],
        "model": m["name"],
        "modelShort": m["short"],      # the opponent name players see
        "modelKey": cfg["model"],
        "checkpoint": m["file"] if local else "v3.5_default (Prior Labs API)",
        "device": (w.get("device") or cfg["device"]) if local else "cloud",
        "deviceSetting": cfg["device"],
        "rateLimited": cfg["inference"] == "api" and not MOCK,
    }


# ---------------------------------------------------------------------------
# Fitting and prediction
# ---------------------------------------------------------------------------
# Fitted models keyed by (configuration, hash of the training set). For the API, fitting
# (uploading the context) is ~4 s of the ~5.5 s round trip; the context table only changes when
# a shot lands, so the browser pre-fits via /api/fit right after a landing and /api/predict then
# finds the model here (or waits for the fit already in progress).
_cache = OrderedDict()          # key -> Future[model]
_cache_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=8)
_fit_pool = ThreadPoolExecutor(max_workers=8)


def _key(cfg, X, y):
    tag = f"{cfg['inference']}|{cfg['model']}|{cfg['device']}|{cfg.get('n_estimators') or N_ESTIMATORS}|".encode()
    return hashlib.sha1(tag + X.tobytes() + b"|" + y.tobytes() + str(X.shape).encode()).hexdigest()


def _do_fit(cfg, X, y):
    if cfg["inference"] == "api":
        return _api_regressor().fit(X, y)
    with _local_lock():
        return _local_regressor(cfg).fit(X, y)


def _fit_future(cfg, X, y):
    key = _key(cfg, X, y)
    with _cache_lock:
        fut = _cache.get(key)
        if fut is not None and not (fut.done() and fut.exception() is not None):
            _cache.move_to_end(key)
            return fut
        fut = _fit_pool.submit(_do_fit, cfg, X, y)
        _cache[key] = fut
        while len(_cache) > 32:
            _cache.popitem(last=False)
    return fut


def _quantiles(cfg, X, y, Xt, priority="high"):
    model = _fit_future(cfg, X, y).result()
    if cfg["inference"] == "api":
        qs = model.predict(Xt, output_type="quantiles", quantiles=QUANTILES)
    else:
        with _local_lock(priority):
            qs = model.predict(Xt, output_type="quantiles", quantiles=QUANTILES)
    return dict(zip(("q10", "q50", "q90"), (np.asarray(q, dtype=float).reshape(-1) for q in qs)))


def _mock_predict(X, y, Xt):
    """k-NN stand-in with the same output shape, for testing without a token."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    A, B = (X - mu) / sd, (Xt - mu) / sd
    k = min(7, len(X))
    out = {"q10": [], "q50": [], "q90": []}
    for row in B:
        d = np.sqrt(((A - row) ** 2).sum(1))
        idx = np.argsort(d)[:k]
        w = 1.0 / (d[idx] + 0.3)
        m = float((y[idx] * w).sum() / w.sum())
        s = float(np.sqrt(((y[idx] - m) ** 2 * w).sum() / w.sum()) + 0.05)
        out["q10"].append(m - 1.2816 * s)
        out["q50"].append(m)
        out["q90"].append(m + 1.2816 * s)
    return out


def predict_target(cfg, X, y, Xt, priority="high"):
    if MOCK:
        return _mock_predict(X, y, Xt)
    return {q: v.tolist() for q, v in _quantiles(cfg, X, y, Xt, priority).items()}


def _stack(X, ys, Xt=None):
    """Stack k targets into ONE TabPFN regression: each target is standardized and the rows get a
    one-hot target indicator. The API prices per call (~10k tokens, flat), so this cuts cost k-fold."""
    names = list(ys)
    k = len(names)
    stats = {n: (float(ys[n].mean()), float(ys[n].std()) + 1e-9) for n in names}
    eye = np.eye(k)
    Xs = np.vstack([np.hstack([X, np.tile(eye[j], (len(X), 1))]) for j in range(k)])
    ystack = np.concatenate([(ys[n] - stats[n][0]) / stats[n][1] for n in names])
    Xts = None if Xt is None else np.vstack([np.hstack([Xt, np.tile(eye[j], (len(Xt), 1))]) for j in range(k)])
    return names, stats, Xs, ystack, Xts


def predict_stacked(cfg, X, ys, Xt, priority="high"):
    names, stats, Xs, ystack, Xts = _stack(X, ys, Xt)
    raw = _mock_predict(Xs, ystack, Xts) if MOCK else _quantiles(cfg, Xs, ystack, Xts, priority)
    out, n = {}, len(Xt)
    for j, name in enumerate(names):
        mu, sd = stats[name]
        out[name] = {q: (np.asarray(raw[q][j * n:(j + 1) * n]) * sd + mu).tolist() for q in ("q10", "q50", "q90")}
    return out


def _parse_train(body):
    X = np.array(body["X_train"], dtype=float)
    if X.ndim != 2 or len(X) < 2 or len(X) > MAX_TRAIN_ROWS:
        raise ValueError(f"X_train must be 2-D with 2..{MAX_TRAIN_ROWS} rows")
    ys = {}
    for name, vals in body["y_train"].items():
        y = np.array(vals, dtype=float)
        if y.shape != (len(X),):
            raise ValueError(f"y_train[{name}] must have {len(X)} values")
        ys[name] = y
    return X, ys


def handle_fit(body):
    """Start fitting (no prediction) so a later /api/predict with the same table is fast.
    With "wait": true, respond only once every target's context is fitted and cached."""
    cfg = current_cfg()
    if MOCK:
        return {"started": 0, "backend": "mock"}
    X, ys = _parse_train(body)
    if body.get("stack"):
        _, _, Xs, ystack, _ = _stack(X, ys)
        futs = [_fit_future(cfg, Xs, ystack)]
    else:
        futs = [_fit_future(cfg, X, y) for y in ys.values()]
    if body.get("wait"):
        for f in futs:
            f.result()
    return {"started": len(ys), "fitted": bool(body.get("wait")), "backend": status(cfg)[0], **describe(cfg)}


def handle_gbr(body):
    """DEVELOPER ONLY: gradient-boosted quantile regressor on the same rows.
    Never used by play mode (its outputs carry backend "gbr", which the play-mode guard refuses)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = np.array(body["X_train"], dtype=float)
    Xt = np.array(body["X_test"], dtype=float)
    t0 = time.time()
    out = {}
    for name, vals in body["y_train"].items():
        y = np.array(vals, dtype=float)
        res = {}
        for q, key in zip(QUANTILES, ("q10", "q50", "q90")):
            m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=int(body.get("max_iter", 100)),
                                              learning_rate=float(body.get("learning_rate", 0.1)),
                                              min_samples_leaf=max(2, min(20, len(X) // 5)), random_state=0)
            res[key] = m.fit(X, y).predict(Xt).tolist()
        lo, hi = np.minimum(res["q10"], res["q90"]), np.maximum(res["q10"], res["q90"])
        res["q10"], res["q90"] = lo.tolist(), hi.tolist()
        out[name] = res
    out["ms"] = round((time.time() - t0) * 1000)
    out["backend"] = "gbr"
    return out


def handle_predict(body):
    cfg = current_cfg()
    X = np.array(body["X_train"], dtype=float)
    Xt = np.array(body["X_test"], dtype=float)
    ys = body["y_train"]
    if X.ndim != 2 or Xt.ndim != 2 or X.shape[1] != Xt.shape[1]:
        raise ValueError(f"bad shapes X_train {X.shape} X_test {Xt.shape}")
    if len(X) < 2 or len(X) > MAX_TRAIN_ROWS or len(Xt) > MAX_TEST_ROWS:
        raise ValueError(f"row limits: train 2..{MAX_TRAIN_ROWS}, test <= {MAX_TEST_ROWS}")
    t0 = time.time()
    yarr = {}
    for name, vals in ys.items():
        y = np.array(vals, dtype=float)
        if y.shape != (len(X),):
            raise ValueError(f"y_train[{name}] must have {len(X)} values")
        yarr[name] = y
    priority = "low" if body.get("priority") == "low" else "high"   # local: urgent calls go first
    if priority == "low" and cfg["inference"] == "local" and GRID_ESTIMATORS and len(Xt) >= GRID_MIN_QUERIES:
        cfg = {**cfg, "n_estimators": GRID_ESTIMATORS}            # the movement grid: smaller ensemble, same model
    if body.get("stack") and len(yarr) > 1:
        out = predict_stacked(cfg, X, yarr, Xt, priority)
        calls = 1
    else:
        futures = {name: _pool.submit(predict_target, cfg, X, y, Xt, priority) for name, y in yarr.items()}
        out = {name: f.result() for name, f in futures.items()}
        calls = len(yarr)
    # api_calls counts quota-priced requests (the browser's token estimate); local inference is free.
    out["api_calls"] = calls if cfg["inference"] == "api" and not MOCK else 0
    out["ms"] = round((time.time() - t0) * 1000)
    if out["ms"] > 8000:
        print(f"[predict] slow TabPFN response: {out['ms']} ms ({priority} priority, {len(X)} rows -> {len(Xt)} queries)",
              file=sys.stderr)
    out["backend"] = status(cfg)[0]
    out.update(describe(cfg))
    return out


def handle_config(body):
    """DEVELOPER ONLY: switch backend / model / device at runtime. Fits are cached per configuration,
    and up to 3 built local models stay on the device, so switching back is fast."""
    if MOCK:
        raise ValueError("the server runs in TABPFN_MOCK mode")
    with _cfg_lock:
        new = dict(CFG)
        for k, field in (("inference", "backend"), ("model", "model"), ("device", "device")):
            if body.get(field) is not None:
                new[k] = str(body[field]).strip().lower()
        validate_cfg(new)
        CFG.update(new)
    cfg = current_cfg()
    if cfg["inference"] == "api":
        FALLBACK["reason"] = None             # chosen again explicitly
    if cfg["inference"] == "local":
        with _warm_lock:  # a failed load (e.g. licence) is retried when it is selected again
            if (_warm.get((cfg["model"], cfg["device"])) or {}).get("state") == "error":
                del _warm[(cfg["model"], cfg["device"])]
    start_warm_up(cfg)
    print(f"[config] {describe(cfg)['inference']} · {describe(cfg)['model']} · device={cfg['device']}", file=sys.stderr)
    return health()


def health():
    cfg = current_cfg()
    backend, ready, msg = status(cfg)
    if FALLBACK["reason"] and cfg["inference"] == "local":
        msg += " (switched from the Prior Labs API, which was unavailable)"
    return {
        "ok": True,
        "backend": backend,
        "ready": ready,
        "status": msg,
        **describe(cfg),
        "models": [{"key": k, "name": m["name"], "file": m["file"]} for k, m in MODELS.items()],
        "apiModel": API_MODEL,
        "devices": list(DEVICES),
        "tokenConfigured": bool(TOKEN),
        "tabpfnClientInstalled": HAVE_CLIENT,
        "tabpfnInstalled": HAVE_LOCAL,
        "fallback": FALLBACK["reason"] if cfg["inference"] == "local" else None,
    }


def serve_tabpfn(route, body):
    """/api/predict or /api/fit. If the API is in use and turns out to be unavailable, the same request is
    answered by local TabPFN-3.5 instead (the first local answer waits for the model to load)."""
    run = handle_predict if route == "/api/predict" else handle_fit
    cfg = current_cfg()
    try:
        return run(body)
    except Exception as e:
        if cfg["inference"] == "api" and api_unavailable(e) and fall_back_to_local(f"{type(e).__name__}: {e}"):
            return run(body)
        raise


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload, ctype="application/json"):
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionError):
            pass  # the browser gave up (request timeout) and is already using its fallback

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/badminton.html"):
            # Only the game page and the lazily loaded developer module are served: never .env or other files.
            with open(os.path.join(WEB, "badminton.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/dev-tools.js":
            print("[dev] developer tools module requested", file=sys.stderr)
            with open(os.path.join(WEB, "dev-tools.js"), "rb") as f:
                self._send(200, f.read(), "application/javascript; charset=utf-8")
        elif path == "/api/health":
            self._send(200, health())
        else:
            self._send(404, {"error": "not found"})

    def _body(self):
        n = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(n))

    def do_POST(self):
        route = self.path.split("?")[0]
        if route not in ("/api/predict", "/api/fit", "/api/gbr", "/api/config"):
            return self._send(404, {"error": "not found"})
        if route in ("/api/gbr", "/api/config"):
            try:
                return self._send(200, (handle_gbr if route == "/api/gbr" else handle_config)(self._body()))
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            except Exception as e:
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        backend, _, msg = status()
        if backend == "none" and current_cfg()["inference"] == "api" and fall_back_to_local(msg):
            backend, _, msg = status()
        if backend == "none":
            return self._send(503, {"error": msg})
        try:
            body = self._body()
            self._send(200, serve_tabpfn(route, body))
        except Exception as e:  # report every failure to the browser (the bot then waits / pauses)
            print(f"[predict] error: {e!r}", file=sys.stderr)
            msg = f"{type(e).__name__}: {e}"
            self._send(429 if "429" in msg or "Rate limit" in msg or "usage limit" in msg else 500, {"error": msg})

    def log_message(self, fmt, *args):
        if "/api/" in (args[0] if args else "") and "health" not in (args[0] if args else ""):
            return  # one line per prediction would flood the console
        super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser(description="Badminton vs TabPFN-3.5: game server + TabPFN inference.")
    ap.add_argument("--backend", choices=INFERENCES, help="local (default) or api (Prior Labs API, TabPFN-3.5)")
    ap.add_argument("--model", choices=list(MODELS), help="local checkpoint (default v3.5 = TabPFN-3.5)")
    ap.add_argument("--device", choices=DEVICES, help="local device (default auto: CUDA, then MPS, then CPU)")
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    for k, v in (("inference", args.backend), ("model", args.model), ("device", args.device)):
        if v:
            CFG[k] = v
    try:
        validate_cfg(CFG)
    except ValueError as e:
        sys.exit(f"Bad TabPFN configuration in .env: {e}")

    cfg = current_cfg()
    if cfg["inference"] == "api" and status(cfg)[0] == "none":
        fall_back_to_local(status(cfg)[2])
        cfg = current_cfg()
    d = describe(cfg)
    where = (f"local, device={cfg['device']}, checkpoint {d['checkpoint']}" if cfg["inference"] == "local"
             else "Prior Labs API, pinned to TabPFN-3.5 (v3.5), uses quota")
    print(f"Badminton + TabPFN server on http://localhost:{args.port}")
    print(f"  TabPFN: {'MOCK (k-NN stand-in)' if MOCK else d['model'] + ' (' + where + ')'}")
    backend, _, msg = status(cfg)
    if backend == "none":
        print(f"  ! {msg} Play mode needs TabPFN: the bot waits and the rally pauses until it is available.")
    start_warm_up(cfg)
    # Loopback only (never the network). Browsers may resolve "localhost" to IPv4 or IPv6.
    servers = [ThreadingHTTPServer(("127.0.0.1", args.port), Handler)]
    try:
        class V6(ThreadingHTTPServer):
            address_family = socket.AF_INET6
        servers.append(V6(("::1", args.port), Handler))
    except OSError:
        pass  # no IPv6 loopback on this machine
    for srv in servers[1:]:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    servers[0].serve_forever()


if __name__ == "__main__":
    main()
