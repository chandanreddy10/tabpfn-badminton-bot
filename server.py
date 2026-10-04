"""Local server for the badminton game + TabPFN proxy (Phase 2).

    pip install -r requirements.txt
    python server.py            # then open http://localhost:8000

The browser cannot call the TabPFN cloud API directly (the token must stay
secret, and the official client is Python), so the BotController in
badminton.html posts its in-context table to /api/predict here. Each request is
fully self-contained: the browser sends the latest <=100 context rows as the
training set every time, so this server keeps no game state.

Environment (.env next to this file):
    TABPFN_TOKEN=...            # required for the real model
    TABPFN_N_ESTIMATORS=4       # optional, fewer = faster
    TABPFN_MOCK=1               # optional, use a local k-NN stand-in (no API calls)
    PORT=8000                   # optional
"""
import hashlib
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
QUANTILES = [0.1, 0.5, 0.9]
MAX_TRAIN_ROWS = 500
MAX_TEST_ROWS = 1024


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
TOKEN = os.environ.get("TABPFN_TOKEN", "").strip()
MOCK = os.environ.get("TABPFN_MOCK", "0").strip() == "1"
PORT = int(os.environ.get("PORT", "8000"))
N_ESTIMATORS = os.environ.get("TABPFN_N_ESTIMATORS", "").strip()

try:
    import tabpfn_client
    from tabpfn_client import TabPFNRegressor

    HAVE_CLIENT = True
except ImportError:  # pragma: no cover - reported via /api/health
    HAVE_CLIENT = False

if TOKEN and HAVE_CLIENT and not MOCK:
    tabpfn_client.set_access_token(TOKEN)

if MOCK:
    BACKEND = "mock"
elif TOKEN and HAVE_CLIENT:
    BACKEND = "tabpfn"
else:
    BACKEND = "none"

# Fitted models keyed by a hash of the training set. Fitting (uploading the context) is
# ~4 s of the ~5.5 s round trip, and the context table only changes when a shot lands,
# so the browser pre-fits via /api/fit right after a landing and /api/predict then
# finds the model here (or waits for the fit already in progress).
_cache = OrderedDict()          # key -> Future[model]
_cache_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=8)
_fit_pool = ThreadPoolExecutor(max_workers=8)


def _key(X, y):
    return hashlib.sha1(X.tobytes() + b"|" + y.tobytes() + str(X.shape).encode()).hexdigest()


def _do_fit(X, y):
    kwargs = {}
    if N_ESTIMATORS:
        kwargs["n_estimators"] = int(N_ESTIMATORS)
    return TabPFNRegressor(**kwargs).fit(X, y)


def _fit_future(X, y):
    key = _key(X, y)
    with _cache_lock:
        fut = _cache.get(key)
        if fut is not None and not (fut.done() and fut.exception() is not None):
            _cache.move_to_end(key)
            return fut
        fut = _fit_pool.submit(_do_fit, X, y)
        _cache[key] = fut
        while len(_cache) > 32:
            _cache.popitem(last=False)
    return fut


def _fit(X, y):
    return _fit_future(X, y).result()


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


def predict_target(X, y, Xt):
    if MOCK:
        return _mock_predict(X, y, Xt)
    model = _fit(X, y)
    qs = model.predict(Xt, output_type="quantiles", quantiles=QUANTILES)
    q10, q50, q90 = (np.asarray(q, dtype=float).reshape(-1) for q in qs)
    return {"q10": q10.tolist(), "q50": q50.tolist(), "q90": q90.tolist()}


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


def predict_stacked(X, ys, Xt):
    names, stats, Xs, ystack, Xts = _stack(X, ys, Xt)
    if MOCK:
        raw = _mock_predict(Xs, ystack, Xts)
    else:
        qs = _fit(Xs, ystack).predict(Xts, output_type="quantiles", quantiles=QUANTILES)
        raw = dict(zip(("q10", "q50", "q90"), (np.asarray(q, dtype=float).reshape(-1) for q in qs)))
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
    if MOCK:
        return {"started": 0, "backend": BACKEND}
    X, ys = _parse_train(body)
    if body.get("stack"):
        _, _, Xs, ystack, _ = _stack(X, ys)
        futs = [_fit_future(Xs, ystack)]
    else:
        futs = [_fit_future(X, y) for y in ys.values()]
    if body.get("wait"):
        for f in futs:
            f.result()
    return {"started": len(ys), "fitted": bool(body.get("wait")), "backend": BACKEND}


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
    if body.get("stack") and len(yarr) > 1:
        out = predict_stacked(X, yarr, Xt)
        out["api_calls"] = 1
    else:
        futures = {name: _pool.submit(predict_target, X, y, Xt) for name, y in yarr.items()}
        out = {name: f.result() for name, f in futures.items()}
        out["api_calls"] = len(yarr)
    out["ms"] = round((time.time() - t0) * 1000)
    if out["ms"] > 8000:
        print(f"[predict] slow TabPFN response: {out['ms']} ms", file=sys.stderr)
    out["backend"] = BACKEND
    return out


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
            with open(os.path.join(ROOT, "badminton.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/dev-tools.js":
            print("[dev] developer tools module requested", file=sys.stderr)
            with open(os.path.join(ROOT, "dev-tools.js"), "rb") as f:
                self._send(200, f.read(), "application/javascript; charset=utf-8")
        elif path == "/api/health":
            self._send(200, {
                "ok": True,
                "backend": BACKEND,
                "tokenConfigured": bool(TOKEN),
                "tabpfnClientInstalled": HAVE_CLIENT,
            })
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        route = self.path.split("?")[0]
        if route not in ("/api/predict", "/api/fit", "/api/gbr"):
            return self._send(404, {"error": "not found"})
        if route == "/api/gbr":
            try:
                n = int(self.headers.get("Content-Length", "0"))
                return self._send(200, handle_gbr(json.loads(self.rfile.read(n))))
            except Exception as e:
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        if BACKEND == "none":
            msg = ("TABPFN_TOKEN is empty in .env" if HAVE_CLIENT
                   else "tabpfn-client is not installed (pip install -r requirements.txt)")
            return self._send(503, {"error": msg})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n))
            self._send(200, handle_predict(body) if route == "/api/predict" else handle_fit(body))
        except Exception as e:  # report every failure to the browser (the bot then waits / pauses)
            print(f"[predict] error: {e!r}", file=sys.stderr)
            msg = f"{type(e).__name__}: {e}"
            self._send(429 if "429" in msg or "Rate limit" in msg or "usage limit" in msg else 500, {"error": msg})

    def log_message(self, fmt, *args):
        if "/api/" in (args[0] if args else "") and "health" not in (args[0] if args else ""):
            return  # one line per prediction would flood the console
        super().log_message(fmt, *args)


def main():
    print(f"Badminton + TabPFN server on http://localhost:{PORT}  (backend: {BACKEND})")
    if BACKEND == "none":
        print("  ! No TabPFN backend: put TABPFN_TOKEN=... in .env. "
              "Play mode needs TabPFN: the bot waits and the rally pauses until it is available.")
    # Loopback only (never the network). Browsers may resolve "localhost" to IPv4 or IPv6.
    servers = [ThreadingHTTPServer(("127.0.0.1", PORT), Handler)]
    try:
        class V6(ThreadingHTTPServer):
            address_family = socket.AF_INET6
        servers.append(V6(("::1", PORT), Handler))
    except OSError:
        pass  # no IPv6 loopback on this machine
    for srv in servers[1:]:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    servers[0].serve_forever()


if __name__ == "__main__":
    main()
