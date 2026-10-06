// =====================================================================
// dev-tools.js: DEVELOPER MODULE (testing only)
//
// Loaded on demand when the "Developer tools" panel is opened (or with ?dev=1).
// It is never loaded, run or referenced by play mode. Nothing here calls TabPFN until a
// button is pressed.
//
// Contents
//   A. Comparison providers (same interfaces as play mode, honest provenance tags):
//      oracle, linear extrapolation, parametric physics fit (+ constant-wind mismatch), EKF,
//      particle filter, kNN / kernel regression on identical rows, RLS, RLS + CUSUM change
//      detection, frozen model, inverse-variance ensemble, gradient boosting (server /api/gbr).
//   B. Planners: random shooting, MPPI (same cost and options as CEM).
//   C. Developer panel: session-only provider switching, provenance log, calibration,
//      "compare estimators on this shot".
// =====================================================================
(function (G) {
  'use strict';
  const nowMs = () => Clock.now();
  const sum = a => a.reduce((s, v) => s + v, 0);
  const avg = a => (a.length ? sum(a) / a.length : NaN);
  const fmt = (v, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : '–');
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  // =====================================================================
  // A1. Tabular baselines: same fit(X, Y) / predict(Xq) interface as TabPFNRegressorAdapter
  // =====================================================================
  function standardizer(X) {
    const d = X[0].length, mu = Array(d).fill(0), s = Array(d).fill(0);
    for (const r of X) r.forEach((v, j) => { mu[j] += v / X.length; });
    for (const r of X) r.forEach((v, j) => { s[j] += (v - mu[j]) ** 2 / X.length; });
    const sc = s.map(v => Math.sqrt(v) + 1e-6);
    return r => r.map((v, j) => (v - mu[j]) / sc[j]);
  }
  const out1 = (mean, std) => ({ mean, std: Math.max(1e-3, std), q10: mean - 1.2816 * std, q50: mean, q90: mean + 1.2816 * std });

  /** k-nearest-neighbour (or Gaussian-kernel) regression on the identical rows TabPFN gets. */
  class KNNRegressor {
    constructor({ k = 7, kernel = false, bandwidth = 1 } = {}) { this.k = k; this.kernel = kernel; this.bw = bandwidth; this.ctx = null; this.version = 0; this.provenance = kernel ? 'kernel' : 'knn'; }
    get rows() { return this.ctx ? this.ctx.X.length : 0; }
    fit(X, Y) { const z = standardizer(X); this.ctx = { X, Y, Z: X.map(z), z, version: ++this.version }; return Promise.resolve(); }
    predictSync(Xq) {
      const { Z, Y, z } = this.ctx, names = Object.keys(Y), targets = {};
      names.forEach(n => { targets[n] = []; });
      for (const q of Xq) {
        const zq = z(q), d = Z.map((r, i) => [Math.sqrt(sum(r.map((v, j) => (v - zq[j]) ** 2))), i]).sort((a, b) => a[0] - b[0]);
        const nb = this.kernel ? d.slice(0, Math.min(d.length, 40)) : d.slice(0, Math.min(this.k, d.length));
        const w = nb.map(([dist]) => (this.kernel ? Math.exp(-0.5 * (dist / this.bw) ** 2) + 1e-9 : 1 / (dist + 0.05)));
        const W = sum(w);
        for (const n of names) {
          const ys = nb.map(([, i]) => Y[n][i]), m = sum(ys.map((y, i) => y * w[i])) / W;
          const v = sum(ys.map((y, i) => w[i] * (y - m) ** 2)) / W;
          targets[n].push(out1(m, Math.sqrt(v * (1 + 1 / nb.length)) + 0.01));
        }
      }
      return { targets, provenance: this.provenance, ms: 0 };
    }
    async predict(Xq) { const t0 = nowMs(), r = this.predictSync(Xq); r.ms = nowMs() - t0; return r; }
  }

  /** Recursive least squares (linear in the row features + bias) with exponential forgetting.
   *  With `cusum`, a CUSUM test on standardized residuals resets the covariance on a detected change. */
  class RLSRegressor {
    constructor({ lambda = 0.98, cusum = false, threshold = 5, drift = 0.5 } = {}) {
      this.lambda = lambda; this.cusum = cusum; this.h = threshold; this.k = drift; this.ctx = null; this.version = 0;
      this.provenance = cusum ? 'rls-cusum' : 'rls'; this.detections = [];
    }
    get rows() { return this.ctx ? this.ctx.n : 0; }
    fit(X, Y) {
      const d = X[0].length + 1, names = Object.keys(Y), models = {};
      this.detections = [];
      for (const n of names) {
        let w = Array(d).fill(0), P = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => (i === j ? 100 : 0)));
        let s2 = 0.05, gp = 0, gn = 0;
        X.forEach((row, t) => {
          const x = row.concat([1]), Px = P.map(r => sum(r.map((v, j) => v * x[j]))), den = this.lambda + sum(x.map((v, j) => v * Px[j]));
          const K = Px.map(v => v / den), e = Y[n][t] - sum(w.map((v, j) => v * x[j]));
          if (this.cusum && t > 10) {               // CUSUM on standardized residuals
            const z = Math.abs(e) / Math.sqrt(s2 + 1e-6);
            gp = Math.max(0, gp + z - 1 - this.k);
            if (gp > this.h) { P = P.map((r, i) => r.map((_, j) => (i === j ? 100 : 0))); gp = 0; gn++; this.detections.push(t); }
          }
          w = w.map((v, j) => v + K[j] * e);
          P = P.map((r, i) => r.map((v, j) => (v - K[i] * Px[j]) / this.lambda));
          s2 = 0.97 * s2 + 0.03 * e * e;
        });
        models[n] = { w, s: Math.sqrt(s2) };
      }
      this.ctx = { models, n: X.length, X, Y, version: ++this.version };
      return Promise.resolve();
    }
    predictSync(Xq) {
      const targets = {};
      for (const [n, m] of Object.entries(this.ctx.models)) targets[n] = Xq.map(r => out1(sum(r.concat([1]).map((v, j) => v * m.w[j])), m.s));
      return { targets, provenance: this.provenance, ms: 0 };
    }
    async predict(Xq) { const t0 = nowMs(), r = this.predictSync(Xq); r.ms = nowMs() - t0; return r; }
  }

  /** Frozen model: the wrapped regressor is fitted on the first context it sees and never updated. */
  class FrozenRegressor {
    constructor(inner) { this.inner = inner; this.frozen = false; this.provenance = 'frozen-' + inner.provenance; }
    get ctx() { return this.inner.ctx; }
    get rows() { return this.inner.rows; }
    fit(X, Y) { if (this.frozen) return Promise.resolve(); this.frozen = true; return this.inner.fit(X, Y); }
    async predict(Xq) { const r = await this.inner.predict(Xq); return { ...r, provenance: this.provenance }; }
  }

  /** Gradient-boosted quantile regressor on the same rows (server /api/gbr, sklearn). */
  class GBRRegressor {
    constructor({ maxIter = 100, learningRate = 0.1 } = {}) { this.maxIter = maxIter; this.lr = learningRate; this.ctx = null; this.version = 0; this.provenance = 'gbr'; }
    get rows() { return this.ctx ? this.ctx.X.length : 0; }
    fit(X, Y) { this.ctx = { X, Y, version: ++this.version }; return Promise.resolve(); }
    async predict(Xq) {
      const t0 = nowMs();
      const r = await fetch((TabPFNClient.base() || '') + '/api/gbr', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ X_train: this.ctx.X, y_train: this.ctx.Y, X_test: Xq, max_iter: this.maxIter, learning_rate: this.lr }) });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || 'gbr failed');
      const targets = {};
      for (const n of Object.keys(this.ctx.Y)) targets[n] = j[n].q50.map((m, i) => ({ mean: m, std: Math.max(1e-3, (j[n].q90[i] - j[n].q10[i]) / 2.563), q10: j[n].q10[i], q50: m, q90: j[n].q90[i] }));
      return { targets, provenance: 'gbr', ms: nowMs() - t0 };
    }
  }

  /** Inverse-variance ensemble of regressors (per target and row). */
  class EnsembleRegressor {
    constructor(members) { this.members = members; this.provenance = 'ensemble(' + members.map(m => m.provenance).join('+') + ')'; }
    get ctx() { return this.members[0].ctx; }
    get rows() { return this.members[0].rows; }
    fit(X, Y) { return Promise.all(this.members.map(m => m.fit(X, Y))); }
    async predict(Xq) {
      const t0 = nowMs(), outs = await Promise.all(this.members.map(m => m.predict(Xq))), targets = {};
      for (const n of Object.keys(outs[0].targets)) targets[n] = Xq.map((_, i) => {
        const w = outs.map(o => 1 / o.targets[n][i].std ** 2), W = sum(w);
        return out1(sum(outs.map((o, k) => w[k] * o.targets[n][i].mean)) / W, Math.sqrt(1 / W));
      });
      return { targets, provenance: this.provenance, ms: nowMs() - t0 };
    }
  }

  /** Oracle dynamics: exact velocity change from the true game physics (cheating upper bound). */
  class OracleDynRegressor {
    constructor(getTruth) { this.getTruth = getTruth; this.ctx = { X: [[0]], Y: { dvx: [0], dvy: [0] }, version: 1 }; this.provenance = 'oracle'; }
    get rows() { return 0; }
    fit() { return Promise.resolve(); }
    predictSync(Xq) {
      const { cfg, env, side } = this.getTruth(), tx = [], ty = [];
      for (const r of Xq) {
        let q = { x: r[0], y: r[1], vx: r[2], vy: r[3], side, recover: 0 };
        for (let i = 0; i < BOTCFG.mpc.ctrlTicks; i++) q = Physics.stepPlayer(q, { x: r[4], y: r[5] }, DT, { ...cfg, lag: 0 }, false, env);
        tx.push(out1(q.vx - r[2], 0.01)); ty.push(out1(q.vy - r[3], 0.01));
      }
      return { targets: { dvx: tx, dvy: ty }, provenance: 'oracle', ms: 0 };
    }
    async predict(Xq) { return this.predictSync(Xq); }
  }

  // =====================================================================
  // A2. Landing baselines that do not use the tabular rows
  //     Each is a function (entry observed up to checkpoint k, context entries, params) -> estimate.
  // =====================================================================
  const CPS = () => BOTCFG.landing.checkpoints;
  const upto = (e, k) => e.samples.filter(s => s.t <= CPS()[k] + 1e-6);
  function lsq(samples, key, deg) {      // least squares polynomial in t (deg 1 or 2) -> coefficients
    const n = samples.length;
    if (deg === 1 || n < 3) {
      const st = sum(samples.map(s => s.t)), sv = sum(samples.map(s => s[key])), stt = sum(samples.map(s => s.t * s.t)), stv = sum(samples.map(s => s.t * s[key]));
      const den = n * stt - st * st, b = Math.abs(den) > 1e-9 ? (n * stv - st * sv) / den : 0;
      return [(sv - b * st) / n, b, 0];
    }
    const S = [0, 0, 0, 0, 0], T = [0, 0, 0];
    for (const s of samples) { let p = 1; for (let i = 0; i < 5; i++) { S[i] += p; if (i < 3) T[i] += p * s[key]; p *= s.t; } }
    const A = [[S[0], S[1], S[2]], [S[1], S[2], S[3]], [S[2], S[3], S[4]]];
    const det = m => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    const D = det(A);
    if (Math.abs(D) < 1e-12) return lsq(samples, key, 1);
    return [0, 1, 2].map(c => det(A.map((r, i) => r.map((v, j) => (j === c ? T[i] : v)))) / D);
  }
  const est = (mx, my, sx, sy, T, sT, provenance) => ({ mean: [r4(mx), r4(my)], cov: [[sx * sx, 0], [0, sy * sy]], tArrival: r4(T), tArrivalSigma: sT, provenance });
  /** Flight time from a quadratic fit of the observed height (falls back to the public timing law). */
  function flightTime(e, k, timing) {
    const S = upto(e, k), c = CPS()[k], [a, b, q] = lsq(S, 'z', 2);
    if (q < -1e-3) { const disc = b * b - 4 * q * a; if (disc >= 0) { const r = (-b - Math.sqrt(disc)) / (2 * q); if (r > c) return r; } }
    const last = S[S.length - 1], d = hyp(last.x - e.p0.x, last.y - e.p0.y);
    return Math.max(c + 0.1, (timing[e.type].baseTime + timing[e.type].perMeter * d * 1.6));
  }
  const LandingMethods = {
    oracle: { name: 'oracle (upper bound)', fn: (e, k) => est(e.landing.x, e.landing.y, 0.01, 0.01, e.tArrival, 0.01, 'oracle'), hp: [{}] },
    linear: {
      name: 'linear extrapolation', hp: [{ scale: 0.5 }, { scale: 1 }, { scale: 2 }],
      fn(e, k, ctx, P, hp) {
        const S = upto(e, k), [ax, bx] = lsq(S, 'x', 1), [ay, by] = lsq(S, 'y', 1), T = flightTime(e, k, P.timing);
        const res = Math.sqrt(avg(S.map(s => (s.x - ax - bx * s.t) ** 2 + (s.y - ay - by * s.t) ** 2)) / 2) + 0.05;
        const sp = (res + 0.4 * (T - CPS()[k])) * hp.scale;
        return est(ax + bx * T, ay + by * T, sp, sp, T, 0.3, 'linear-extrapolation');
      },
    },
    physics: {
      name: 'parametric physics fit', hp: [{ window: 0 }, { window: 3 }, { window: 10 }],
      fn(e, k, ctx, P, hp) {     // public timing + drag law; window > 0 adds a CONSTANT-wind correction (mean recent residual)
        const g = physicsGuess(e, k, P.timing);
        let cx = 0, cy = 0, s = 0.45;
        if (hp.window > 0) {
          const recent = ctx.slice(-hp.window).map(r => { const q = physicsGuess(r, k, P.timing); return q && r.landing ? [r.landing.x - q.x, r.landing.y - q.y] : null; }).filter(Boolean);
          if (recent.length) { cx = avg(recent.map(v => v[0])); cy = avg(recent.map(v => v[1])); s = Math.sqrt(avg(recent.map(v => (v[0] - cx) ** 2 + (v[1] - cy) ** 2)) / 2) + 0.05; }
        }
        return est(g.x + cx, g.y + cy, s, s, g.T, 0.15, hp.window ? 'physics-fit+const-wind' : 'physics-fit');
      },
    },
    ekf: {
      name: 'extended Kalman filter', hp: [{ q: 0.5 }, { q: 2 }, { q: 5 }],
      fn(e, k, ctx, P, hp) {     // state [x, y, vx, vy, lambda]: velocity decays at rate lambda (drag); xy measurements
        const S = upto(e, k), R = 0.08 * 0.08;
        let x = [S[0].x, S[0].y, 0, 0, 1.5], Pm = [[R, 0, 0, 0, 0], [0, R, 0, 0, 0], [0, 0, 25, 0, 0], [0, 0, 0, 25, 0], [0, 0, 0, 0, 1]];
        const mm = (A, B) => A.map(r => B[0].map((_, j) => sum(r.map((v, i) => v * B[i][j]))));
        const tr = A => A[0].map((_, j) => A.map(r => r[j]));
        for (let i = 1; i < S.length; i++) {
          const dt = S[i].t - S[i - 1].t, l = x[4], f = Math.exp(-l * dt), g = (1 - f) / Math.max(l, 1e-3);
          x = [x[0] + x[2] * g, x[1] + x[3] * g, x[2] * f, x[3] * f, l];
          const dg = (dt * f * l - (1 - f)) / Math.max(l * l, 1e-6);
          const F = [[1, 0, g, 0, x[2] / Math.max(f, 1e-6) * dg], [0, 1, 0, g, x[3] / Math.max(f, 1e-6) * dg], [0, 0, f, 0, -dt * x[2]], [0, 0, 0, f, -dt * x[3]], [0, 0, 0, 0, 1]];
          const Q = [0, 1, 2, 3, 4].map(a => [0, 1, 2, 3, 4].map(b => (a === b ? [0.001, 0.001, hp.q * dt, hp.q * dt, 0.05 * dt][a] : 0)));
          Pm = mm(mm(F, Pm), tr(F)).map((r, a) => r.map((v, b) => v + Q[a][b]));
          const zr = [S[i].x - x[0], S[i].y - x[1]], Sx = [[Pm[0][0] + R, Pm[0][1]], [Pm[1][0], Pm[1][1] + R]];
          const det = Sx[0][0] * Sx[1][1] - Sx[0][1] * Sx[1][0], Si = [[Sx[1][1] / det, -Sx[0][1] / det], [-Sx[1][0] / det, Sx[0][0] / det]];
          const K = Pm.map(r => [r[0] * Si[0][0] + r[1] * Si[1][0], r[0] * Si[0][1] + r[1] * Si[1][1]]);
          x = x.map((v, a) => v + K[a][0] * zr[0] + K[a][1] * zr[1]);
          x[4] = clamp(x[4], 0.05, 8);
          Pm = Pm.map((r, a) => r.map((v, b) => v - K[a][0] * Pm[0][b] - K[a][1] * Pm[1][b]));
        }
        const T = flightTime(e, k, P.timing), tau = Math.max(0, T - S[S.length - 1].t), l = x[4], g = (1 - Math.exp(-l * tau)) / l;
        const sx = Math.sqrt(Math.max(1e-4, Pm[0][0] + g * g * Pm[2][2])) + 0.1, sy = Math.sqrt(Math.max(1e-4, Pm[1][1] + g * g * Pm[3][3])) + 0.1;
        return est(x[0] + x[2] * g, x[1] + x[3] * g, sx, sy, T, 0.3, 'ekf');
      },
    },
    pf: {
      name: 'particle filter', hp: [{ slack: 0.1 }, { slack: 0.25 }],
      fn(e, k, ctx, P, hp) {     // particles over (landing x, y, flight time T) under the public drag-eased path
        const S = upto(e, k), sp = P.timing[e.type], kd = sp.dragK, ease = u => (kd > 1e-6 ? (1 - Math.exp(-kd * u)) / (1 - Math.exp(-kd)) : u);
        const rng = { s: (e.shotId * 2654435761) >>> 0 || 1 }, side = Math.sign(S[S.length - 1].y - e.p0.y) || -1, n = 1500, sig = 0.08 + hp.slack;
        const parts = [];
        for (let i = 0; i < n; i++) {
          const px = -3.2 + 6.4 * RNG.next(rng), py = side * (0.2 + 7.2 * RNG.next(rng));
          const T = Math.max(S[S.length - 1].t + 0.05, sp.baseTime + sp.perMeter * hyp(px - e.p0.x, py - e.p0.y) + 0.5 * (RNG.next(rng) - 0.5));
          let ll = 0;
          for (const s of S) { const u = ease(Math.min(1, s.t / T)); ll += ((e.p0.x + (px - e.p0.x) * u - s.x) ** 2 + (e.p0.y + (py - e.p0.y) * u - s.y) ** 2) / (2 * sig * sig); }
          parts.push([px, py, T, ll]);
        }
        const mn = Math.min(...parts.map(p => p[3])), w = parts.map(p => Math.exp(mn - p[3])), W = sum(w);
        const m = [0, 1, 2].map(j => sum(parts.map((p, i) => w[i] * p[j])) / W), v = [0, 1, 2].map(j => sum(parts.map((p, i) => w[i] * (p[j] - m[j]) ** 2)) / W);
        return est(m[0], m[1], Math.sqrt(v[0]) + 0.05, Math.sqrt(v[1]) + 0.05, m[2], Math.sqrt(v[2]) + 0.05, 'particle-filter');
      },
    },
  };
  /** The M2 parametric physics guess (public timing + drag law, ignores wind and noise). */
  function physicsGuess(e, k, timing) {
    const S = upto(e, k), pc = S[S.length - 1], c = CPS()[k];
    if (!pc) return null;
    const sp = timing[e.type], dx = pc.x - e.p0.x, dy = pc.y - e.p0.y, trav = hyp(dx, dy);
    if (trav < 1e-3) return { x: pc.x, y: pc.y, T: sp.baseTime };
    const kd = sp.dragK, ease = u => (kd > 1e-6 ? (1 - Math.exp(-kd * u)) / (1 - Math.exp(-kd)) : u);
    const f = D => D * ease(Math.min(1, c / (sp.baseTime + sp.perMeter * D))) - trav;
    let lo = 0.05, hi = 20;
    if (f(hi) < 0) lo = hi;
    for (let i = 0; i < 40 && hi - lo > 1e-3; i++) { const m = (lo + hi) / 2; if (f(m) < 0) lo = m; else hi = m; }
    const D = (lo + hi) / 2;
    return { x: e.p0.x + dx / trav * D, y: e.p0.y + dy / trav * D, T: sp.baseTime + sp.perMeter * D };
  }

  /** LandingEstimator wrapper for the non-tabular methods (live switching / closed loop). */
  class FilterLandingEstimator extends TabularLandingEstimator {
    constructor(methodKey, hp, timing) {
      super(() => ({ fit() {}, ctx: null }));
      this.method = LandingMethods[methodKey]; this.hp = hp || this.method.hp[0]; this.timing = timing; this.key = methodKey;
    }
    query(k) {
      const cur = this.cur, t0 = nowMs();
      if (this.key === 'oracle' && !cur.landing) { if (this.truth) Object.assign(cur, this.truth(cur.shotId)); if (!cur.landing) return; }
      const e = this.method.fn(cur, k, this.entries, { timing: this.timing }, this.hp);
      if (!e) return;
      const out = { shotId: cur.shotId, k, c: CPS()[k], ...e, ms: nowMs() - t0 };
      CallLog.add({ task: 'landing', rows: this.entries.length, batch: 1, ms: out.ms, outcome: 'ok', provenance: e.provenance });
      if (!Guard.check(out, 'landing')) return;
      cur.estimates.push(out);
      if (!this.est || this.est.k <= k) this.est = out;
    }
  }

  // =====================================================================
  // B. Planners sharing MPC.evaluate (cost) and the options of CEM
  // =====================================================================
  class RandomShootingPlanner {
    constructor(opts, rng) { this.o = opts; this.rng = rng; this.name = 'random-shooting'; }
    plan(C, warm) {
      const g = () => RNG.gauss(this.rng), n = this.o.N * this.o.iters, H = C.H;
      const base = (warm || []).slice(0, H); while (base.length < H) base.push(base.length ? base[base.length - 1] : [0, 0]);
      let best = null, all = [];
      for (let i = 0; i < n; i++) {
        const U = i === 0 ? base : base.map(u => clipNorm([u[0] + this.o.sigma0 * g(), u[1] + this.o.sigma0 * g()]));
        const r = { U, ...MPC.evaluate(U, C, i < 16) }; all.push(r);
        if (!best || r.cost < best.cost) best = r;
      }
      if (!best.path) best = { ...best, ...MPC.evaluate(best.U, C, true) };
      return { U: best.U, cost: best.cost, pHit: best.pHit, path: best.path, shown: all.slice(0, 16).map(r => r.path).filter(Boolean) };
    }
  }
  class MPPIPlanner {
    constructor(opts, rng, temperature = 1) { this.o = opts; this.rng = rng; this.lam = temperature; this.name = 'mppi'; }
    plan(C, warm) {
      const g = () => RNG.gauss(this.rng), H = C.H;
      let mu = (warm || []).slice(0, H); while (mu.length < H) mu.push(mu.length ? mu[mu.length - 1] : [0, 0]);
      let shown = [];
      for (let it = 0; it < this.o.iters; it++) {
        const cands = Array.from({ length: this.o.N }, (_, i) => (i === 0 ? mu : mu.map(u => clipNorm([u[0] + this.o.sigma0 * g(), u[1] + this.o.sigma0 * g()]))));
        const sc = cands.map(U => ({ U, ...MPC.evaluate(U, C, it === this.o.iters - 1) })), mn = Math.min(...sc.map(s => s.cost));
        const w = sc.map(s => Math.exp(-(s.cost - mn) / this.lam)), W = sum(w);
        mu = mu.map((_, k) => clipNorm([sum(sc.map((s, i) => w[i] * s.U[k][0])) / W, sum(sc.map((s, i) => w[i] * s.U[k][1])) / W]));
        if (it === this.o.iters - 1) shown = sc.slice(0, 16).map(s => s.path);
      }
      const r = MPC.evaluate(mu, C, true);
      return { U: mu, cost: r.cost, pHit: r.pHit, path: r.path, shown };
    }
  }

  // =====================================================================
  // Provider factories (used by live switching, closed-loop runs and tests)
  // =====================================================================
  const TABULAR_LANDING = ['tabpfn', 'knn', 'kernel', 'gbr', 'ensemble'];
  function makeRegressor(kind, task, opts = {}) {
    switch (kind) {
      case 'tabpfn': return opts.tabpfn ? opts.tabpfn(task) : new TabPFNRegressorAdapter(TabPFNClient, task);
      case 'knn': return new KNNRegressor({ k: opts.k || 7 });
      case 'kernel': return new KNNRegressor({ kernel: true, bandwidth: opts.bw || 1 });
      case 'gbr': return new GBRRegressor();
      case 'rls': return new RLSRegressor({ lambda: opts.lambda || 0.98 });
      case 'rls-cusum': return new RLSRegressor({ lambda: opts.lambda || 0.98, cusum: true, threshold: opts.threshold || 5 });
      case 'frozen': return new FrozenRegressor(new KNNRegressor({ k: opts.k || 7 }));
      case 'ensemble': return new EnsembleRegressor([new KNNRegressor({ k: 7 }), new RLSRegressor({ lambda: 0.98 })]);
      case 'oracle': return new OracleDynRegressor(opts.truth);
      default: throw new Error('unknown regressor ' + kind);
    }
  }
  function makeLanding(kind, opts = {}) {
    if (TABULAR_LANDING.includes(kind)) return new TabularLandingEstimator(t => makeRegressor(kind, t, opts));
    const f = new FilterLandingEstimator(kind, opts.hp, opts.timing || defaultTiming());
    f.truth = opts.landingTruth;
    return f;
  }
  function makeDynamics(kind, opts = {}) { return new TabularDynamicsModel(t => makeRegressor(kind, t, opts)); }
  function defaultTiming() { return Observation.build(Sim.create('t', 1, DEFAULT_CONFIG, 0), 1).params.shotTiming; }

  /** Test data for the developer self-tests: machine practice shots at an exploring receiver (P2). */
  function practiceData(seed, nShots, nTrans) {
    const S = Sim.createPractice(`dev-${seed}`, clone(DEFAULT_CONFIG), 900 + seed, nShots, 0);
    const col = new BotController({ makeRegressor: () => ({ fit() { return Promise.resolve(); }, ctx: null, rows: 0 }), storageKey: null });
    col.landing.opts = { ...col.landing.opts, window: 1e9 };
    col.dynamics.opts = { ...col.dynamics.opts, window: 1e9, windowSec: 1e9, dedupEps: 0 };
    col.setMode('warmup');
    const L = new ShotLogger(), idle = { observe() {}, act: () => ({ move: { x: 0, y: 0 } }) };
    for (let k = 0; k < 2e6 && S.phase !== 'practiceDone'; k++) Sim.step(S, [idle, col], L);
    return { entries: col.landing.entries.filter(e => !e.byBot), trans: col.dynamics.trans.slice(0, nTrans), timing: Observation.build(S, 1).params.shotTiming };
  }

  // =====================================================================
  // C. Developer panel
  // =====================================================================
  const DevTools = {
    app: null, root: null,
    landingKinds: ['tabpfn', 'knn', 'kernel', 'gbr', 'ensemble', 'linear', 'physics', 'ekf', 'pf', 'oracle'],
    dynKinds: ['tabpfn', 'knn', 'rls', 'rls-cusum', 'frozen', 'gbr', 'ensemble', 'oracle'],
    plannerKinds: ['cem', 'random-shooting', 'mppi'],
    init(app, root) {
      this.app = app; this.root = root;
      root.innerHTML = `
        <p class="muted">Testing tools. Switching providers affects this session only and is never saved; a reload always returns to TabPFN-only play.</p>
        <h3>Providers (session only)</h3>
        <div class="dev-grid">
          <label>Landing <select id="dv-land">${this.landingKinds.map(k => `<option>${k}</option>`).join('')}</select></label>
          <label>Dynamics <select id="dv-dyn">${this.dynKinds.map(k => `<option>${k}</option>`).join('')}</select></label>
          <label>Planner <select id="dv-plan">${this.plannerKinds.map(k => `<option>${k}</option>`).join('')}</select></label>
        </div>
        <label class="check"><input type="checkbox" id="dv-strict"> Strict per-step TabPFN rollout (slow, many API calls) instead of the TabPFN surrogate</label>
        <label class="check"><input type="checkbox" id="dv-stack" checked> Stack targets into one TabPFN call (cheaper; off = one regressor per coordinate)</label>
        <div class="row"><button id="dv-apply">Apply to this session</button><button id="dv-reset" class="primary">Restore TabPFN-only play</button></div>
        <p id="dv-mode" class="muted"></p>
        <h3>Status and calibration</h3><div id="dv-status"></div>
        <h3>Call log (provenance)</h3><div id="dv-log" class="dev-scroll"></div>
        <h3>Compare estimators on this shot</h3>
        <p class="muted">Uses the last opponent shot the bot saw and the TabPFN estimates recorded during play (no new calls).</p>
        <button id="dv-compare">Compare on last shot</button><div id="dv-cmp"></div>`;
      const $ = id => root.querySelector('#' + id);
      $('dv-apply').onclick = () => this.apply($('dv-land').value, $('dv-dyn').value, $('dv-plan').value, $('dv-strict').checked, $('dv-stack').checked);
      $('dv-reset').onclick = () => { $('dv-land').value = 'tabpfn'; $('dv-dyn').value = 'tabpfn'; $('dv-plan').value = 'cem'; $('dv-strict').checked = false; $('dv-stack').checked = true; this.restore(); };
      $('dv-compare').onclick = () => this.compare();
      this.timer = setInterval(() => this.refresh(), 500);
      this.refresh();
    },
    /** Session-only provider switch. Never persisted. */
    apply(land, dyn, plan, strict, stack) {
      const bot = this.app.bot, allTab = land === 'tabpfn' && dyn === 'tabpfn';
      Guard.mode = allTab ? 'play' : 'dev';
      BOTCFG.mpc.linearize = !strict; BOTCFG.stackTargets = stack;
      const entries = bot.landing.entries, trans = bot.dynamics.trans;
      bot.landing = makeLanding(land, { timing: defaultTiming(), landingTruth: id => { const r = this.app.logger.records.find(x => x.shotId === id && x.matchId === this.app.S.matchId); return r ? { landing: { x: r.trueLandingX, y: r.trueLandingY }, tArrival: r.flightTime } : {}; } });
      bot.landing.entries = entries; bot.landing.version++;
      bot.dynamics = makeDynamics(dyn, { truth: () => ({ cfg: this.app.S.cfg, env: this.app.S.env, side: this.app.S.players[1].side }) });
      bot.dynamics.trans = trans; bot.dynamics.maybeRefit(true);
      bot.planner = plan === 'cem' ? new CEMPlanner(BOTCFG.mpc, { s: 0x2545f491 }) : plan === 'mppi' ? new MPPIPlanner(BOTCFG.mpc, { s: 7 }) : new RandomShootingPlanner(BOTCFG.mpc, { s: 7 });
      bot.landing.onFail = bot.dynamics.onFail = () => { bot.stats.failed++; };
      bot.landing.onOutput = (t, o) => bot.logOutput(t, o);
      bot.setMode(bot.mode);
      this.root.querySelector('#dv-mode').textContent = allTab && plan === 'cem' && !strict ? 'Play configuration: TabPFN only.' : `Session override: landing=${land}, dynamics=${dyn}, planner=${plan}${strict ? ', strict rollout' : ''}. Not saved.`;
    },
    restore() { this.apply('tabpfn', 'tabpfn', 'cem', false, true); },
    refresh() {
      const b = this.app.bot, pc = c => (c && c.rate !== null ? Math.round(c.rate * 100) + '% (n=' + c.n + ')' : '–'), st = this.root.querySelector('#dv-status');
      if (!st) return;
      const errs = a => (a && a.length ? fmt(avg(a), 3) : '–');
      st.innerHTML = `<table class="kv">
        <tr><td>Guard mode</td><td>${Guard.mode} · accepted ${Guard.accepted} · refused ${Guard.rejected}</td></tr>
        <tr><td>Landing provider</td><td>${esc(b.landing.constructor.name)} · 90% coverage ${pc(b.landing.cov90)} · mean err ${errs(b.landing.err)} m</td></tr>
        <tr><td>Dynamics provider</td><td>${esc(b.dynamics.reg ? b.dynamics.reg.provenance || 'tabpfn' : '–')} · 90% coverage ${pc(b.dynamics.cov90)} · mean err ${errs(b.dynamics.err)} m/s</td></tr>
        <tr><td>Own-shot model</td><td>90% coverage ${pc(b.shots.cov90)}</td></tr>
        <tr><td>Planner</td><td>${esc(b.planner.name || 'cem')} · status ${b.status}</td></tr>
        <tr><td>TabPFN model</td><td>${esc(TabPFNClient.describe())}</td></tr>
        <tr><td>API calls this session</td><td>${CallLog.apiCalls} (≈${Math.round(CallLog.apiCalls * BOTCFG.tokensPerCall / 1000)}k tokens) · rate budget used ${RateBudget.used()}/${BOTCFG.rate.perMinute} per min · deferred ${RateBudget.deferred}</td></tr>
      </table>`;
      const log = this.root.querySelector('#dv-log');
      log.innerHTML = '<table><tr><th>task</th><th>prov.</th><th>rows</th><th>batch</th><th>ms</th><th>outcome</th></tr>' +
        CallLog.entries.slice(-25).reverse().map(e => `<tr><td>${esc(e.task)}</td><td>${esc(e.provenance)}</td><td>${e.rows}</td><td>${e.batch}</td><td>${Math.round(e.ms)}</td><td>${esc(e.outcome)}</td></tr>`).join('') + '</table>';
    },
    /** Compare all landing methods on the last opponent shot (TabPFN from the recorded play outputs). */
    compare() {
      const b = this.app.bot, ent = b.landing.entries.filter(e => !e.byBot), e = ent[ent.length - 1], out = this.root.querySelector('#dv-cmp');
      if (!e) { out.innerHTML = '<p class="muted">No opponent shot observed yet.</p>'; return; }
      const ctx = ent.slice(0, -1), timing = defaultTiming(), recorded = b.providerLog.filter(p => p.task === 'landing' && p.out.shotId === e.shotId);
      let html = '<table><tr><th>method</th>' + CPS().map(c => `<th>${c}s</th>`).join('') + '</tr>';
      const cell = r => { if (!r) return '<td>–</td>'; const dx = e.landing.x - r.mean[0], dy = e.landing.y - r.mean[1], inside = dx * dx / r.cov[0][0] + dy * dy / r.cov[1][1] <= 4.605; return `<td>${fmt(hyp(dx, dy), 2)} m ${inside ? '✓' : '✗'}</td>`; };
      html += '<tr><td><b>tabpfn (recorded)</b></td>' + CPS().map((_, k) => cell((recorded.find(p => p.out.k === k) || {}).out)).join('') + '</tr>';
      for (const m of ['knn', 'physics', 'ekf', 'pf', 'linear', 'oracle']) {
        html += `<tr><td>${m}</td>` + CPS().map((_, k) => {
          if (!LandingRows.row(e, k, CPS(), true)) return '<td>–</td>';
          if (m === 'knn') {
            const it = LandingRows.table(ctx, k, CPS(), true);
            if (it.length < 3) return '<td>–</td>';
            const reg = new KNNRegressor({ k: 7 }); reg.fit(it.map(z => z.x), { dx: it.map(z => z.y.dx), dy: it.map(z => z.y.dy), tr: it.map(z => z.y.tr) });
            const q = LandingRows.row(e, k, CPS(), true), o = reg.predictSync([q.x]).targets;
            return cell({ mean: [q.last.x + o.dx[0].mean, q.last.y + o.dy[0].mean], cov: [[o.dx[0].std ** 2, 0], [0, o.dy[0].std ** 2]] });
          }
          return cell(LandingMethods[m].fn(e, k, ctx, { timing }, LandingMethods[m].hp[0]));
        }).join('') + '</tr>';
      }
      out.innerHTML = html + '</table><p class="muted">Error to the true landing; ✓ = inside the method\'s 90% ellipse.</p>';
    },
    // ---- developer self-tests (fixtures live here, never in play mode) ----------------------
    async selfTest() {
      const res = [], check = (name, ok) => res.push({ name: '[dev] ' + name, ok: !!ok });
      const savedMode = Guard.mode, savedClock = Clock.now;
      // regressors
      const X = Array.from({ length: 60 }, (_, i) => [Math.sin(i), Math.cos(i * 0.7), i / 60]), Y = { a: X.map(r => 2 * r[0] - r[1]), b: X.map(r => r[2] * 3) };
      const knn = new KNNRegressor({ k: 5 }); await knn.fit(X, Y); const kq = knn.predictSync([[0.5, 0.2, 0.5]]);
      check('kNN regressor predicts on identical rows', Math.abs(kq.targets.a[0].mean - 0.8) < 0.5 && kq.provenance === 'knn');
      const rls = new RLSRegressor({ lambda: 0.99 }); await rls.fit(X, Y); const rq = rls.predictSync([[0.5, 0.2, 0.5]]);
      check('RLS recovers a linear target', Math.abs(rq.targets.a[0].mean - 0.8) < 0.05 && Math.abs(rq.targets.b[0].mean - 1.5) < 0.05);
      const Xc = X.concat(X.slice(0, 40)), Yc = { a: Y.a.concat(Y.a.slice(0, 40).map(v => v + 3)), b: Y.b.concat(Y.b.slice(0, 40)) };
      const cus = new RLSRegressor({ lambda: 0.995, cusum: true, threshold: 4 }); await cus.fit(Xc, Yc);
      check('RLS + CUSUM detects an injected change', cus.detections.some(t => t >= 60 && t < 80));

      // landing baselines on generated data
      const data = practiceData(3, 24, 120);
      const e = data.entries[20], errs = {};
      for (const m of ['linear', 'physics', 'ekf', 'pf']) { const r = LandingMethods[m].fn(e, 2, data.entries.slice(0, 20), { timing: data.timing }, LandingMethods[m].hp[0]); errs[m] = hyp(r.mean[0] - e.landing.x, r.mean[1] - e.landing.y); }
      check(`landing baselines produce estimates (${Object.entries(errs).map(([k, v]) => k + ' ' + v.toFixed(2)).join(', ')} m; linear ignores drag)`,
        Object.values(errs).every(Number.isFinite) && errs.physics < 1.5 && errs.ekf < 1.5 && errs.pf < 1.5);

      // oracle providers are refused in play mode
      Guard.mode = 'play';
      const S = Sim.createPractice('guard', DEFAULT_CONFIG, 1, 8, 0);
      const bot = new BotController({ landing: makeLanding('oracle', { landingTruth: () => ({}) }), dynamics: makeDynamics('oracle', { truth: () => ({ cfg: S.cfg, env: S.env, side: -1 }) }), makeRegressor: t => makeRegressor('knn', t), storageKey: null });
      bot.setMode('play'); for (const t of practiceData(4, 8, 80).trans) bot.dynamics.addTransition(t);
      const rej0 = Guard.rejected; let moved = 0; const L = new ShotLogger(), idle = { observe() {}, act: () => ({ move: { x: 0, y: 0 } }) };
      for (let k = 0; k < 6000 && S.phase !== 'practiceDone'; k++) { Sim.step(S, [idle, bot], L); await Promise.resolve(); moved = Math.max(moved, hyp(S.players[1].vx, S.players[1].vy)); }
      check('play mode refuses injected oracle providers (bot never moves on them)', Guard.rejected > rej0 && moved === 0 && !bot.dynamics.sur);

      // degraded sequence: fresh plan -> late (follows shifted plan) -> brake -> unavailable (fake clock)
      Guard.mode = 'dev';
      let fake = 0; Clock.now = () => fake;
      const S2 = Sim.createPractice('degraded', DEFAULT_CONFIG, 1, 6, 0);
      const ok = { fail: false };
      const flaky = t => { const r = makeRegressor('oracle', t, { truth: () => ({ cfg: S2.cfg, env: S2.env, side: -1 }) }); const p = r.predict.bind(r); r.predict = async q => { if (ok.fail) throw new Error('down'); return p(q); }; return r; };
      const bot2 = new BotController({ landing: makeLanding('oracle', { landingTruth: () => ({}) }), dynamics: new TabularDynamicsModel(flaky), makeRegressor: t => makeRegressor('knn', t), storageKey: null });
      bot2.setMode('play'); for (const t of practiceData(5, 8, 80).trans) bot2.dynamics.addTransition(t);
      const seen = new Set(), L2 = new ShotLogger();
      for (let k = 0; k < 1500; k++) {
        if (k === 300) { ok.fail = true; fake += BOTCFG.dynamics.ttlMs + 1; bot2.dynamics.reg.ctx.version++; TabPFNHealth.reset(); }
        Sim.step(S2, [idle, bot2], L2); await Promise.resolve(); await Promise.resolve();
        if (k > 300 && k % 12 === 0) { fake += 400; bot2.dynamics.maybeRefit(true); }
        seen.add(bot2.status);
      }
      if (ok.fail) { for (let i = 0; i < 3; i++) TabPFNHealth.fail(new Error('down')); fake += BOTCFG.unavailableMs + 1; }
      check(`degraded: plan -> late -> waiting (statuses ${[...seen].join(', ')}) -> unavailable`, seen.has('ok') && seen.has('late') && seen.has('waiting') && TabPFNHealth.unavailable());
      Clock.now = savedClock; TabPFNHealth.reset(); Guard.mode = savedMode;

      // provider switching is never persisted
      const store = {}, prevLS = G.localStorage;
      try { G.localStorage = { setItem: (k, v) => { store[k] = v; }, getItem: k => store[k] || null, removeItem: k => { delete store[k]; } }; } catch (err) { /* read-only */ }
      const fakeApp = { bot: new BotController({ makeRegressor: t => makeRegressor('knn', t), storageKey: null }), logger: new ShotLogger(), S: Sim.create('x', 1, DEFAULT_CONFIG, 1),
                        download() {} };
      const fakeRoot = { innerHTML: '', querySelector: () => ({ textContent: '', set onclick(v) {}, value: '', checked: false }), querySelectorAll: () => [] };
      const dt = Object.create(DevTools); dt.app = fakeApp; dt.root = fakeRoot;
      dt.apply('knn', 'rls', 'mppi', false, true);
      check('developer provider switch is session-only (nothing written to storage)', Object.keys(store).every(k => !/provider|planner|dev/i.test(k)) && Guard.mode === 'dev');
      dt.restore();
      check('restore returns to the TabPFN-only play configuration', Guard.mode === 'play');
      try { G.localStorage = prevLS; } catch (err) { /* ignore */ }
      Guard.mode = savedMode;

      return res;
    },
  };

  G.DevTools = DevTools;
  G.DevProviders = { LandingMethods, KNNRegressor, RLSRegressor, makeLanding, makeDynamics, makeRegressor };
})(typeof window !== 'undefined' ? window : globalThis);
