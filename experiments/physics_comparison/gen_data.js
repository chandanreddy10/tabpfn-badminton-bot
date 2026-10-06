// Experiment data generator: node experiments/physics_comparison/gen_data.js [--seeds 3]
// Uses the game's own simulator (Physics, Wind, observation noise, Explorer) and the bot's real
// feature builders (LandingRows, DynRows), so every row is exactly what the bot would feed TabPFN.
// It also precomputes the existing physics baselines from dev-tools.js (DevProviders.LandingMethods)
// on the same shots. Output: experiments/data/{landing,dynamics}_<scenario>_s<seed>.json
const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..', '..');
const html = fs.readFileSync(path.join(ROOT, 'web', 'badminton.html'), 'utf8');
const src = html.match(/<script>([\s\S]*)<\/script>/)[1];
const dev = fs.readFileSync(path.join(ROOT, 'web', 'dev-tools.js'), 'utf8');
const G = new Function(src + '\n' + dev + `
;return { Physics, Wind, RNG, Explorer, LandingRows, DynRows, DEFAULT_CONFIG, COURT, BOTCFG, SHOT_KINDS, clone, r4,
          DevProviders: globalThis.DevProviders };`)();
const { Physics, Wind, RNG, Explorer, LandingRows, DynRows, DEFAULT_CONFIG, COURT, BOTCFG, SHOT_KINDS, clone, r4 } = G;
const LM = G.DevProviders.LandingMethods;

const args = process.argv.slice(2);
const SEEDS = +(args[args.indexOf('--seeds') + 1] || 3) || 3;
const OUT = path.join(__dirname, '..', 'data');
fs.mkdirSync(OUT, { recursive: true });
const DT = 1 / 120;

// ---------------------------------------------------------------- landing
// The human (side +1) hits rally shots to the bot (side -1). Shots come every ~3.5-5.5 s, so the
// slowly drifting wind evolves across the sequence as in a real match.
const CPS = BOTCFG.landing.checkpoints;
const PHYS = { physics: { m: LM.physics, hp: { window: 0 } }, physics_wind: { m: LM.physics, hp: { window: 10 } },
               ekf: { m: LM.ekf, hp: { q: 2 } }, pf: { m: LM.pf, hp: { slack: 0.1 } }, linear: { m: LM.linear, hp: { scale: 1 } } };

function landingSeq({ seed, wind, noise, n = 400, shiftAt = null }) {
  const cfg = clone(DEFAULT_CONFIG); cfg.windStrength = wind; cfg.noiseMult = noise;
  const seedInt = RNG.hash(`landing-${seed}`);
  let windP = Wind.makeParams(seedInt);
  const rShot = RNG.stream(seedInt, 'shot'), rObs = RNG.stream(seedInt, 'obs'), rPos = RNG.stream(seedInt, 'pos');
  const u = () => RNG.next(rPos), W = COURT.singlesHalfW;
  const timing = {}; for (const k of SHOT_KINDS) { const s = cfg.shots[k]; timing[k] = { baseTime: s.baseTime, perMeter: s.perMeter, dragK: s.dragK }; }
  const P = { timing };
  const shots = [], ctx = [];
  let t0 = 5, bot = { x: 0, y: -3.3 };
  for (let i = 0; i < n; i++) {
    if (shiftAt !== null && i === shiftAt) {                 // regime change: the base wind reverses direction
      windP = { ...windP, bx: -windP.bx, by: -windP.by };
    }
    const type = ['clear', 'drop', 'smash'][Math.floor(u() * 3)];
    const from = { x: r4(-2.3 + 4.6 * u()), y: r4(1.5 + 4.5 * u()) };
    const far = type === 'drop' ? cfg.dropMaxDepth : COURT.halfLen;
    const target = { x: r4(-W + 2 * W * u()), y: r4(-(cfg.netMargin + (far - cfg.netMargin) * u())) };
    const plan = Physics.plan({ from, target, type, t0 }, windP, cfg, rShot);
    const samples = [];
    for (let t = 0; t <= Math.min(plan.T, CPS[CPS.length - 1]) + 1e-9; t += cfg.obsInterval) {
      const pos = Physics.shuttleAt(plan, t);
      samples.push({ t: r4(t), x: r4(pos.x + RNG.gauss(rObs) * cfg.obsNoiseXY), y: r4(pos.y + RNG.gauss(rObs) * cfg.obsNoiseXY),
                     z: r4(Math.max(0, pos.z + RNG.gauss(rObs) * cfg.obsNoiseZ)) });
    }
    // The bot's position is a feature only (a random walk near its base, independent of the landing).
    bot = { x: r4(Math.max(-2.4, Math.min(2.4, bot.x * 0.7 + (u() - 0.5) * 1.6))), y: r4(Math.max(-6, Math.min(-1.2, -3.3 + (bot.y + 3.3) * 0.7 + (u() - 0.5) * 1.6))) };
    const botAt = CPS.map((c, k) => ({ x: r4(bot.x + 0.15 * k * (u() - 0.5)), y: r4(bot.y + 0.15 * k * (u() - 0.5)) }));
    const e = { shotId: i + 1, type, isServe: false, p0: from, t: t0, samples, botAt,
                landing: { x: r4(plan.landing.x), y: r4(plan.landing.y) }, tArrival: r4(plan.T) };
    const feats = CPS.map((c, k) => { const q = LandingRows.row(e, k, CPS, true); return q ? { x: q.x, last: { x: q.last.x, y: q.last.y } } : null; });
    const phys = {};
    for (const [name, { m, hp }] of Object.entries(PHYS)) {
      phys[name] = CPS.map((c, k) => {
        if (!feats[k]) return null;
        const ee = { ...e, landing: null };                 // the method must not see the answer
        const r = m.fn(ee, k, ctx.slice(-120), P, hp);
        return { mx: r.mean[0], my: r.mean[1], sx: Math.sqrt(r.cov[0][0]), sy: Math.sqrt(r.cov[1][1]), T: r.tArrival, sT: r.tArrivalSigma };
      });
    }
    shots.push({ i, t0: r4(t0), type, p0: from, target, landing: e.landing, T: e.tArrival, drift: plan.drift, noise: plan.noise,
                 meanWind: plan.meanWind, samples, feats, phys });
    ctx.push(e);
    t0 += 3.5 + 2 * u();
  }
  return { kind: 'landing', seed, wind, noise, shiftAt, checkpoints: CPS, featureNames: CPS.map((c, k) => LandingRows.names(k, CPS, true)),
           shotParams: timing, obsNoiseXY: cfg.obsNoiseXY, shots };
}

// ---------------------------------------------------------------- dynamics
// The bot's own movement under the warm-up Explorer's random commands, recorded every 0.1 s exactly
// like BotController.controlTick: state at the decision, command u, previous command up, and dv.
function dynamicsSeq({ seed, scenario, seconds = 180 }) {
  const cfg = clone(DEFAULT_CONFIG);
  if (scenario === 'lag') cfg.lag = 0.1;
  if (scenario === 'slippery') cfg.friction = 6;
  const envAt = (t, p) => {
    if (scenario === 'court' && t >= 60) return { grip: 0.35, speedMult: 0.75 };            // the game's COURT_CHANGE
    if (scenario === 'wetpatch' && p.x < -0.8) return { grip: 0.35, speedMult: 1 };       // unmodelled: a wet patch on the left
    return null;
  };
  const ex = new Explorer(RNG.hash(`dyn-${scenario}-${seed}`));
  let p = { x: 0, y: -3.5, vx: 0, vy: 0, side: -1, recover: 0 }, lastU = [0, 0];
  const trans = [], ticks = Math.round(seconds / DT);
  for (let tick = 0; tick < ticks; tick += 12) {
    const t = tick * DT, s = { x: p.x, y: p.y, vx: p.vx, vy: p.vy };
    const u = ex.command({ t, side: -1, self: { x: p.x, y: p.y } }), up = lastU;
    const env0 = envAt(t, p);
    for (let j = 0; j < 12; j++) p = Physics.stepPlayer(p, { x: u[0], y: u[1] }, DT, cfg, false, envAt((tick + j) * DT, p));
    const dv = { x: r4(p.vx - s.vx), y: r4(p.vy - s.vy) };
    trans.push({ t: r4(t), x: DynRows.row({ s, u, up }), dv: [dv.x, dv.y], grip: env0 ? env0.grip : 1, speedMult: env0 ? env0.speedMult : 1 });
    lastU = u;
  }
  return { kind: 'dynamics', seed, scenario, featureNames: DynRows.names, dt: 0.1, substeps: 12,
           truth: { accel: cfg.accel, friction: cfg.friction, playerSpeed: cfg.playerSpeed, lag: cfg.lag, grip: cfg.grip },
           bounds: { xm: COURT.doublesHalfW + 0.9, ym: COURT.halfLen + 1.0, nm: 0.25 }, trans };
}

const write = (name, obj) => fs.writeFileSync(path.join(OUT, name + '.json'), JSON.stringify(obj));
const LANDING = [
  { name: 'wind0', wind: 0, noise: 1 }, { name: 'wind1', wind: 1, noise: 1 }, { name: 'wind2', wind: 2, noise: 1 }, { name: 'wind3', wind: 3, noise: 1 },
  { name: 'noise0', wind: 1, noise: 0 }, { name: 'noise2', wind: 1, noise: 2 }, { name: 'windshift', wind: 2, noise: 1, shiftAt: 200 },
];
const DYN = ['default', 'court', 'lag', 'slippery', 'wetpatch'];
for (let seed = 1; seed <= SEEDS; seed++) {
  for (const c of LANDING) { write(`landing_${c.name}_s${seed}`, { scenario: c.name, ...landingSeq({ seed, ...c }) }); process.stdout.write('.'); }
  for (const sc of DYN) { write(`dynamics_${sc}_s${seed}`, dynamicsSeq({ seed, scenario: sc })); process.stdout.write('.'); }
}
console.log(`\nwrote ${SEEDS * (LANDING.length + DYN.length)} files to ${path.relative(ROOT, OUT)}`);
