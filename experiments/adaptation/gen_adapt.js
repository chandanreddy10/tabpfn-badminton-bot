// Adaptation experiment data: node experiments/adaptation/gen_adapt.js [--seeds 2]
// X1 shuttle worlds (200 shots each), X2 a sequence that switches world every 50 shots, and X3 opponent
// personalities (where the human hits next). Uses the game's own Worlds, Physics, Wind, observation noise
// and LandingRows, so every landing row is exactly what the bot would feed TabPFN.
const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..', '..');
const html = fs.readFileSync(path.join(ROOT, 'web', 'badminton.html'), 'utf8');
const src = html.match(/<script>([\s\S]*)<\/script>/)[1];
const dev = fs.readFileSync(path.join(ROOT, 'web', 'dev-tools.js'), 'utf8');
const G = new Function(src + '\n' + dev + `
;return { Physics, Wind, RNG, Worlds, WORLDS, LandingRows, DEFAULT_CONFIG, COURT, BOTCFG, SHOT_KINDS, clone, r4,
          DevProviders: globalThis.DevProviders };`)();
const { Physics, Wind, RNG, Worlds, LandingRows, DEFAULT_CONFIG, COURT, BOTCFG, SHOT_KINDS, clone, r4 } = G;
const LM = G.DevProviders.LandingMethods;
const args = process.argv.slice(2);
const SEEDS = +(args[args.indexOf('--seeds') + 1] || 2) || 2;
const OUT = path.join(__dirname, '..', 'data_adapt');
fs.mkdirSync(OUT, { recursive: true });
const CPS = BOTCFG.landing.checkpoints, W = COURT.singlesHalfW;
const NOMINAL = {};                       // the default timing law: what the bot (and the reference physics) knows
for (const k of SHOT_KINDS) { const s = DEFAULT_CONFIG.shots[k]; NOMINAL[k] = { baseTime: s.baseTime, perMeter: s.perMeter, dragK: s.dragK }; }

/** Shots from the human (side +1) to the bot. worldAt(i) gives the world of shot i. */
function landingSeq({ seed, n, worldAt, tag }) {
  const seedInt = RNG.hash(`adapt-${tag}-${seed}`);
  const windP = Wind.makeParams(seedInt);
  const rShot = RNG.stream(seedInt, 'shot'), rObs = RNG.stream(seedInt, 'obs'), rPos = RNG.stream(seedInt, 'pos');
  const u = () => RNG.next(rPos);
  const shots = [];
  let t0 = 5, bot = { x: 0, y: -3.3 };
  for (let i = 0; i < n; i++) {
    const world = worldAt(i), cfg = Worlds.apply(DEFAULT_CONFIG, world);
    const type = ['clear', 'drop', 'smash'][Math.floor(u() * 3)];
    const from = { x: r4(-2.3 + 4.6 * u()), y: r4(1.5 + 4.5 * u()) };
    const far = type === 'drop' ? cfg.dropMaxDepth : COURT.halfLen;
    const target = { x: r4(-W + 2 * W * u()), y: r4(-(cfg.netMargin + (far - cfg.netMargin) * u())) };
    const plan = Physics.plan({ from, target, type, t0 }, windP, cfg, rShot);
    const samples = [];
    for (let t = 0; t <= Math.min(plan.T, CPS[CPS.length - 1]) + 1e-9; t += cfg.obsInterval) {
      const p = Physics.shuttleAt(plan, t);
      samples.push({ t: r4(t), x: r4(p.x + RNG.gauss(rObs) * cfg.obsNoiseXY), y: r4(p.y + RNG.gauss(rObs) * cfg.obsNoiseXY),
                     z: r4(Math.max(0, p.z + RNG.gauss(rObs) * cfg.obsNoiseZ)) });
    }
    bot = { x: r4(Math.max(-2.4, Math.min(2.4, bot.x * 0.7 + (u() - 0.5) * 1.6))), y: r4(Math.max(-6, Math.min(-1.2, -3.3 + (bot.y + 3.3) * 0.7 + (u() - 0.5) * 1.6))) };
    const botAt = CPS.map((c, k) => ({ x: r4(bot.x + 0.15 * k * (u() - 0.5)), y: r4(bot.y + 0.15 * k * (u() - 0.5)) }));
    const e = { shotId: i + 1, type, isServe: false, p0: from, t: t0, samples, botAt, landing: null, tArrival: null };
    const feats = CPS.map((c, k) => { const q = LandingRows.row(e, k, CPS, true); return q ? { x: q.x, last: { x: q.last.x, y: q.last.y } } : null; });
    // reference: the game's timing+drag physics, tuned for the NORMAL world (it is not told the world either)
    const phys = CPS.map((c, k) => { if (!feats[k]) return null; const r = LM.physics.fn(e, k, [], { timing: NOMINAL }, { window: 0 });
                                     return { mx: r.mean[0], my: r.mean[1], sx: Math.sqrt(r.cov[0][0]), sy: Math.sqrt(r.cov[1][1]), T: r.tArrival }; });
    shots.push({ i, world, type, t0: r4(t0), landing: { x: r4(plan.landing.x), y: r4(plan.landing.y) }, T: r4(plan.T), feats, phys });
    t0 += 3.5 + 2 * u();
  }
  return { checkpoints: CPS, featureNames: CPS.map((c, k) => LandingRows.names(k, CPS, true)), shots };
}

// ------------------------------------------------------------------ opponent personalities (X3)
// The human (side +1) hits to the bot's half (y < 0). Before each human hit the bot knows: where the human
// hits from (its own shot's landing), its own position, the human's previous shot and the rally shot index.
const PERSONALITIES = ['crosscourt', 'dropper', 'smasher', 'away', 'pattern', 'random'];
const TYPES = ['clear', 'drop', 'smash'];
function intent(p, st, u) {
  const sgn = v => (v >= 0 ? 1 : -1), deep = () => -(4.8 + 1.6 * u()), front = () => -(0.5 + 1.5 * u());
  switch (p) {
    case 'crosscourt': return { type: u() < 0.8 ? 'clear' : 'smash', x: -sgn(st.hx + (u() - 0.5) * 0.3) * (1.2 + 1.2 * u()), y: deep() };
    case 'dropper': return u() < 0.65 ? { type: 'drop', x: -sgn(st.bx) * (1.0 + 1.4 * u()), y: front() } : { type: 'clear', x: (u() - 0.5) * 2, y: deep() };
    case 'smasher': return st.hy < 3.0 ? { type: 'smash', x: st.bx + (u() - 0.5) * 1.2, y: -(2.5 + 2.5 * u()) } : { type: 'clear', x: (u() - 0.5) * 4, y: deep() };
    case 'away': return st.by > -3.5 ? { type: 'clear', x: -sgn(st.bx) * 2.2, y: deep() } : { type: 'drop', x: -sgn(st.bx) * 2.0, y: front() };
    case 'pattern': return [{ type: 'clear', x: -2, y: -6 }, { type: 'drop', x: 2, y: -1.2 }, { type: 'clear', x: 2, y: -6 }, { type: 'drop', x: -2, y: -1.2 }][st.k % 4];
    default: { const type = TYPES[Math.floor(u() * 3)]; return { type, x: (u() - 0.5) * 2 * W, y: type === 'drop' ? front() : -(0.5 + 5.9 * u()) }; }
  }
}
function habitsSeq({ seed, personality, n = 200 }) {
  const r = RNG.stream(RNG.hash(`habits-${personality}-${seed}`), 'h'), u = () => RNG.next(r), g = () => RNG.gauss(r);
  const rows = [];
  let prev = null, k = 0, rallyLeft = 0;
  for (let i = 0; i < n; i++) {
    if (rallyLeft <= 0) { rallyLeft = 2 + Math.floor(u() * 7); prev = null; k = 0; }
    // the bot's shot decides where the human hits from; the bot then recovers part-way to its base
    const hx = r4((u() - 0.5) * 4.6), hy = r4(0.8 + 5.4 * u());
    const hitFrom = prev ? { x: prev.lx, y: prev.ly } : { x: (u() - 0.5) * 2, y: -4 };
    const bx = r4(hitFrom.x * 0.5 + (u() - 0.5) * 0.8), by = r4(-3.3 + (hitFrom.y + 3.3) * 0.5 + (u() - 0.5) * 0.8);
    const st = { hx, hy, bx, by, k };
    const it = intent(personality, st, u);
    const tx = Math.max(-W, Math.min(W, it.x + g() * 0.25)), ty = Math.max(-COURT.halfLen, Math.min(-0.3, it.y + g() * 0.25));
    const sig = DEFAULT_CONFIG.shots[it.type].noise;           // the shot's own landing noise
    const lx = r4(tx + g() * sig), ly = r4(ty + g() * sig);
    const pt = prev ? prev.type : 'none';
    rows.push({ i, x: [hx, hy, bx, by, prev ? prev.lx : 0, prev ? prev.ly : 0, +(pt === 'none'), +(pt === 'clear'), +(pt === 'drop'), +(pt === 'smash'), k],
                landing: { x: lx, y: ly }, type: it.type });
    prev = { lx, ly, type: it.type }; k++; rallyLeft--;
  }
  return { featureNames: ['hitFromX', 'hitFromY', 'botX', 'botY', 'prevX', 'prevY', 'prevNone', 'prevClear', 'prevDrop', 'prevSmash', 'rallyShot'], rows };
}

const write = (name, obj) => fs.writeFileSync(path.join(OUT, name + '.json'), JSON.stringify(obj));
const keys = Worlds.keys;
for (let seed = 1; seed <= SEEDS; seed++) {
  for (const w of keys) write(`x1_${w}_s${seed}`, { exp: 'x1', world: w, seed, ...landingSeq({ seed, n: 200, worldAt: () => w, tag: 'x1-' + w }) });
  // X2: 6 worlds x 50 shots, in a seed-dependent order starting from normal
  const order = ['normal', ...keys.filter(k => k !== 'normal').sort((a, b) => RNG.hash(a + seed) - RNG.hash(b + seed))];
  write(`x2_switch_s${seed}`, { exp: 'x2', seed, order, block: 50, ...landingSeq({ seed, n: 300, worldAt: i => order[Math.floor(i / 50)], tag: 'x2' }) });
  for (const p of PERSONALITIES) write(`x3_${p}_s${seed}`, { exp: 'x3', personality: p, seed, ...habitsSeq({ seed, personality: p }) });
  process.stdout.write('.');
}
console.log(`\nwrote ${fs.readdirSync(OUT).length} files to ${path.relative(ROOT, OUT)}`);
