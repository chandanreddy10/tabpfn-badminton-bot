"""API -> local fallback in server.py, with the API failures faked (no network, no tokens spent).

    .venv/bin/python tests/test_server_fallback.py

Needs the tabpfn package and the TabPFN-3.5 weights (downloaded on the first run of server.py).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import server as S  # noqa: E402

results = []


def check(name, ok):
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + name, flush=True)


def use_api(token="test-token"):
    S.CFG.update(inference="api", model="v3.5")
    S.TOKEN = token
    S.FALLBACK["reason"] = None


def fail(msg):
    def regressor():
        raise RuntimeError(msg)
    return regressor


body = {"X_train": [[float(i), float(i % 3)] for i in range(12)], "y_train": {"a": [float(i) for i in range(12)]},
        "X_test": [[5.5, 1.0]]}

# 1. API chosen without a token: nothing to call, so the server switches to local before trying.
use_api(token="")
check("no token: API reported unavailable", S.status()[0] == "none")
check("no token: switches to local TabPFN-3.5", S.fall_back_to_local(S.status()[2]) and S.current_cfg()["inference"] == "local")

# 2. A bad request is not an outage: it must fail without switching.
use_api()
S._api_regressor = fail("HTTP 429: Rate limit exceeded")
try:
    S.serve_tabpfn("/api/predict", {**body, "X_test": [[1.0, 2.0, 3.0]]})
    check("bad request raises", False)
except ValueError:
    check("bad request raises and stays on the API", S.current_cfg()["inference"] == "api" and S.FALLBACK["reason"] is None)

# 3. The API hits its rate limit mid-match: the same request is answered locally.
out = S.serve_tabpfn("/api/predict", body)
h = S.health()
check("rate limit: request answered by local TabPFN-3.5",
      out["inference"] == "local" and out["model"] == "TabPFN-3.5" and out["backend"] == "tabpfn" and out["api_calls"] == 0)
check("rate limit: quantiles are ordered", out["a"]["q10"][0] <= out["a"]["q50"][0] <= out["a"]["q90"][0])
check("health reports the fallback", h["inference"] == "local" and bool(h["fallback"]) and "switched from the Prior Labs API" in h["status"])

# 4. Network trouble counts as unavailable too.
use_api()
S._api_regressor = fail("ConnectionError: Max retries exceeded with url: /fit")
out = S.serve_tabpfn("/api/fit", {**body, "wait": True})
check("connection error: switches to local", out["inference"] == "local" and S.current_cfg()["inference"] == "local")

# 5. TABPFN_FALLBACK=none keeps the API and reports the error.
use_api()
S.FALLBACK_ENABLED = False
S._api_regressor = fail("HTTP 429: Rate limit exceeded")
try:
    S.serve_tabpfn("/api/predict", body)
    check("fallback off: error is reported", False)
except RuntimeError:
    check("fallback off: error is reported and the API stays selected", S.current_cfg()["inference"] == "api")

print(f"{sum(results)}/{len(results)}")
sys.exit(0 if all(results) else 1)
