"""Physics baselines for the experiments (no learning except where stated).

Landing
  whitebox_landing  Bayesian least-squares fit of the simulator's EXACT trajectory equation
                    (Physics.shuttleAt: drag-eased path to target+noise, plus a drift term growing with u^2)
                    to the noisy samples seen so far. It knows the shot timing law, the drag constant and
                    the observation noise. It does not know the wind, so the drift gets a zero-mean prior.
                    This is the strongest physics model for landing we can build: it has the true structure.

Movement (dv per 0.1 s control step)
  step_dv           Vectorised port of Physics.stepPlayer over the 12 sub-steps of one control step:
                    acceleration-limited approach to command*vmax, a separate (friction) rate with no
                    input, command lag (the command of the previous control step applies for `lag` ticks)
                    and the court walls. With the true parameters it reproduces the simulator exactly.
  calibrate         Least-squares fit of (accel_eff, friction_eff, vmax[, lag]) on a window of transitions:
                    "white-box physics + system identification", the strongest physics baseline.
  SimplifiedPhysics Textbook first-order model dv = a*u - b*v (no saturation, no lag), fitted by OLS.
"""
import numpy as np
from scipy.optimize import least_squares

DT = 1 / 120
SUB = 12


# ---------------------------------------------------------------------------- landing
def _ease(u, k):
    return (1 - np.exp(-k * u)) / (1 - np.exp(-k)) if k > 1e-6 else u


def whitebox_landing(samples, p0, sp, c, sigma_obs=0.08, sigma_drift=0.5, iters=4):
    """samples: list of {t,x,y} up to checkpoint c. sp: {baseTime, perMeter, dragK}.
    Returns (mx, my, sx, sy, T): landing mean, per-axis std, flight time."""
    S = [s for s in samples if s["t"] <= c + 1e-6]
    t = np.array([s["t"] for s in S]); X = np.array([s["x"] for s in S]); Y = np.array([s["y"] for s in S])
    x0, y0 = p0["x"], p0["y"]
    # first guess of the flight time from the public timing law and the direction travelled so far
    T = sp["baseTime"] + sp["perMeter"] * 6.0
    for _ in range(iters):
        u = np.clip(t / T, 0, 1)
        s = _ease(u, sp["dragK"])
        A = np.stack([s, u * u], 1)                      # unknowns per axis: [b, d] in  pos = p0 + (b - p0) s + d u^2
        prior = np.diag([1e-6, 1 / sigma_drift ** 2])    # flat prior on the end point, N(0, sigma_drift^2) on drift
        H = A.T @ A / sigma_obs ** 2 + prior
        cov = np.linalg.inv(H)
        bx, dx = cov @ (A.T @ (X - x0) / sigma_obs ** 2)  # solves for (b - x0, d)
        by, dy = cov @ (A.T @ (Y - y0) / sigma_obs ** 2)
        bx += x0; by += y0
        T = sp["baseTime"] + sp["perMeter"] * np.hypot(bx - x0, by - y0)
    # landing = end point + full drift (u = 1); variance of (b + d)
    w = np.array([1.0, 1.0])
    var = float(w @ cov @ w)
    return bx + dx, by + dy, np.sqrt(var), np.sqrt(var), T


# ---------------------------------------------------------------------------- movement
def step_dv(F, theta, bounds):
    """F: (n, 8) rows [x, y, vx, vy, ux, uy, upx, upy]. theta = (accel_eff, friction_eff, vmax, lag_ticks).
    Returns dv (n, 2) after one 0.1 s control step (12 sub-steps)."""
    a_eff, f_eff, vmax, lag = theta
    lag = int(round(lag))
    x, y, vx, vy = (F[:, i].astype(float).copy() for i in range(4))
    vx0, vy0 = vx.copy(), vy.copy()
    xm, ym, nm = bounds["xm"], bounds["ym"], bounds["nm"]
    for j in range(SUB):
        cx, cy = (F[:, 6], F[:, 7]) if j < lag else (F[:, 4], F[:, 5])
        L = np.hypot(cx, cy)
        sc = np.where(L > 1, 1 / np.maximum(L, 1e-9), 1.0)
        mx, my = cx * sc, cy * sc
        rate = np.where(L < 0.05, f_eff, a_eff)
        dvx, dvy = mx * vmax - vx, my * vmax - vy
        dv = np.hypot(dvx, dvy); lim = rate * DT
        k = np.where(dv > lim, lim / np.maximum(dv, 1e-12), 1.0)
        vx = vx + dvx * k; vy = vy + dvy * k
        x = x + vx * DT; y = y + vy * DT
        # walls of the bot's half (side -1): x in [-xm, xm], y in [-ym, -nm]
        hitx = (x < -xm) | (x > xm); x = np.clip(x, -xm, xm); vx = np.where(hitx, 0, vx)
        hity = (y < -ym) | (y > -nm); y = np.clip(y, -ym, -nm); vy = np.where(hity, 0, vy)
    return np.stack([vx - vx0, vy - vy0], 1)


NOMINAL = (22.0, 22.0, 4.5, 0)          # DEFAULT_CONFIG: accel, friction, playerSpeed, lag 0


def calibrate(F, dv, bounds, with_lag=True, init=NOMINAL):
    """Least-squares system identification on a window of transitions. Returns (theta, residual std per axis)."""
    lags = range(0, SUB + 1, 2) if with_lag else [0]
    best = None
    for lag in lags:
        def res(p):
            return (step_dv(F, (p[0], p[1], p[2], lag), bounds) - dv).ravel()
        r = least_squares(res, x0=np.array(init[:3], float), bounds=([0.5, 0.5, 0.5], [80, 80, 10]),
                          diff_step=1e-3, max_nfev=60)
        if best is None or r.cost < best[0]:
            best = (r.cost, (*r.x, lag))
    theta = best[1]
    resid = step_dv(F, theta, bounds) - dv
    return theta, resid.std(0) + 1e-3


class SimplifiedPhysics:
    """dv = a*u - b*v per axis (first-order lag toward the command; no saturation, no command lag)."""
    def fit(self, F, dv):
        A = np.concatenate([np.stack([F[:, 4], -F[:, 2]], 1), np.stack([F[:, 5], -F[:, 3]], 1)])
        b = np.concatenate([dv[:, 0], dv[:, 1]])
        self.coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        self.sd = np.std(b - A @ self.coef) + 1e-3
        return self

    def predict(self, F):
        a, b = self.coef
        return np.stack([a * F[:, 4] - b * F[:, 2], a * F[:, 5] - b * F[:, 3]], 1)
