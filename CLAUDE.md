# CLAUDE.md: Badminton vs TabPFN-3.5

Handover notes so a new session (now on macOS) can continue seamlessly. Read this first.

## What this is
A browser singles-badminton game (top-down canvas, one HTML file) where the opponent, shown to players
as **"TabPFN-3.5"**, is driven only by TabPFN predictions plus a model predictive controller (CEM).
It is being prepared as a submission to the **Prior Labs "TabPFN-3.5 Hackathon"** (open-ended; judged by a
Prior Labs panel; categories include "build an agent", "build an app", "formalize a new problem").

## Files
| File | Purpose |
|---|---|
| `badminton.html` | The whole game + play-mode bot + self-tests (~2.7k lines, one inline `<script>`) |
| `dev-tools.js` | Developer module, loaded lazily only when the "Developer tools" panel opens (comparison providers kNN/RLS/EKF/particle filter/physics fit/GBR/oracle, planners, provider switching, call log, "compare estimators on this shot") |
| `server.py` | Local server + TabPFN proxy (stdlib http.server + `tabpfn-client`). Serves only `/`, `/badminton.html`, `/dev-tools.js`; endpoints `/api/health`, `/api/predict`, `/api/fit`, `/api/gbr` (developer only) |
| `run_selftest.js` | Headless tests: `node run_selftest.js` (expects 44/44 PASS) |
| `.env` | `TABPFN_TOKEN=...` (secret; in `.gitignore`; copy manually, never commit or print it) |
| `.env.example`, `requirements.txt`, `README.md`, `.gitignore` | Setup and docs |
| `prompts/`, `tabpfn_context_table.csv` | Created by the user, not by Claude. Ask before including them in a submission |

Not a git repository yet.

## Run (macOS)
```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # tabpfn-client, numpy (sklearn comes with it; used by /api/gbr)
python3 server.py                         # ~10 s to import tabpfn-client, then http://localhost:8000
node run_selftest.js                      # tests, no network
```
URLs: `/` = developer mode (default), `/?dev=0` = clean player view, `/?dev=1` also opens Developer tools,
`?start=bot` skips the start screen, `#selftest` runs tests in the browser.
On Windows, `python` resolved to `C:\Users\chand\Documents\mvp-learn\chatMS\.venv`; on the Mac use the project venv above.

## Architecture (badminton.html)
- Fixed-step deterministic sim (`DT = 1/120`), seeded RNG streams, the whole state `S` is plain JSON. Replay = snapshot + recorded actions (+ recorded court-change env events).
- Modules: `CONFIG/DEFAULT_CONFIG`, `RNG`, `Wind`, `Physics` (plan, shuttleAt, stepPlayer with friction/grip/lag/env), `Rules`, `Observation.build` (the only thing controllers see), controllers (`HumanController`, `ReplayController`, `ScriptController` for tests, `BotController`), `ShotLogger`, `Sim` (incl. `createPractice` warm-up), `Input`, `Renderer`, `App`, `SelfTest`.
- Controller interface: `observe(obs)` + synchronous `act() -> {move, hit?, aim?}`. Network calls are async and never block.
- Observations never contain wind, landing noise, the true trajectory or the landing point before it lands.

### The TabPFN bot (play mode)
Three TabPFN tasks, all via `TabPFNRegressorAdapter` -> `TabPFNClient` -> `server.py` -> hosted API:
1. **Landing** (`TabularLandingEstimator`): at 0.2/0.4/0.6 s after the opponent's hit; features = time since hit, observed x/y/z at each checkpoint so far, finite-difference velocity, shot-type one-hot, hitter x/y, bot x/y; targets = landing minus last observed position (x, y) and remaining flight time. Context = last 120 shots that came TO the bot (own shots excluded because they fly slowed). Min 10 rows.
2. **Movement dynamics** (`TabularDynamicsModel`): rows [x, y, vx, vy, ux, uy, uPrevX, uPrevY] -> (dvx, dvy) per 0.1 s; window 300 rows / 90 s, de-duplicated; refit after >=25% new rows. Planner uses a **global TabPFN surrogate**: one 625-row grid query (5x5 velocities x 5x5 commands), multilinear interpolation, provenance `tabpfn-linearized`, rebuilt only after a refit and at most every 6 s. A "strict per-step" mode exists (developer setting).
3. **Own shot choice** (`TabularShotModel`): 24 random candidates (12 when serving), features = shot one-hot, hitter, target, distance, game time, mean miss of last 3 own shots; target = landing minus target; score = P(in) x P(opponent can't reach). This is a one-step model-based choice, NOT MPC.
- **MPC = movement only:** CEM (N=48, elites 8, 4 iters, H=12 x 0.1 s), warm start from the shifted plan, 8 Monte Carlo rollouts with TabPFN's std and landing samples, chance constraint P(reach within r) >= 0.9, terminal + effort + recovery costs. ~9 ms per replan.
- **TabPFN-only rule:** every output carries a provenance tag; `Guard.check` accepts only `tabpfn` / `tabpfn-linearized` in play mode. Non-TabPFN providers live only in `dev-tools.js` (never referenced by the play script).
- **Degraded behaviour:** late -> follow shifted previous plan; plan runs out -> brake (zero command); unavailable > 5 s -> pause with Retry. Never substitutes another method.
- **Warm-up:** practice phase (20 machine shots + 10 bot practice hits), no TabPFN calls; skipped when a stored context (localStorage) is big enough.
- **Cost controls:** targets of one task are stacked into ONE TabPFN call (standardized targets + target-indicator columns, server `stack: true`); client rate budget (60 req/min, priority to landing/shot); server caches fitted contexts by hash and supports pre-fit.
- **Slow motion** (default 5% speed) while a critical TabPFN call is pending.

## TabPFN facts learned (important)
- Hosted API via `tabpfn-client` 0.6.1. Default model is **TabPFN v3.5** (verified via `estimate_cost(...)` -> `model_version=v3.5`, `n_estimators=8`). **The version is NOT pinned** in `server.py` (`TabPFNRegressor(**kwargs)` at the fit helper): pinning is a pending task.
- Pricing: **~10,000 tokens per predict request, flat** regardless of size (tested up to 3000x3000x10). Limits: **60 predict requests/minute**, **5M tokens/day**, **20M tokens/month** (was 8.66M used on 2026-10-02; resets 2026-11-01). A full match ≈ 1–3M tokens.
- Latency: ~1.1–1.5 s with a cached fit, 4–6 s with a fresh fit (fit dominates), occasionally > 20 s.
- Quantiles are supported (`output_type="quantiles"`, 0.1/0.5/0.9); `random_state` default 0.
- Measured: direct TabPFN dynamics error 0.047 m/s (98–99% coverage); landing beat a physics-only guess by ~10–30% (more in strong wind) once "recent residual" features were used. Last full live game (older surrogate): bot reached 6/9 shots, won 5/8 rallies, 100% TabPFN provenance; landing 90% coverage 100% (n=7); **movement 90% coverage only 56% (too narrow)**.
- The final changes (stacked targets, global surrogate, 3x slower bot shots, UI changes) have **not been validated live** yet (quota ran out).

## Product / UI decisions made with the user
- Opponent display name: `OPPONENT_NAME = 'TabPFN-3.5'` (badminton.html) + static text in the markup. Use "predicting", not "thinking".
- Player view: "You" vs "TabPFN-3.5"; hotseat: "Player 1"/"Player 2". No milestone/metadata text for players.
- **Developer mode is the default**; `?dev=0` = clean player view. Technical panels (call counters, ≈ tokens, latency, provenance/calibration badges, context sizes, "what it sees", Developer tools, seed, debug view `G`) are `.dev-only`.
- How to play screen before the first game; **H** toggles it (pauses the game), Esc closes.
- Bot shots towards the human fly **3x slower** (`slowShotsTo`, `botShotSlowdown`, slider 1–4x); same landing point, only time is stretched; the bot knows this rule (`params.oppReceiveTimeScale`).
- Colour themes: Classic (default, original), Midnight, Clay, Arena; swatches under the title; light+dark each; stored in localStorage `badminton.palette`.
- The benchmark was **removed at the user's request** (also `/api/estimate`, `/api/usage`). Session token estimate exists in developer panels only; account usage display is gone (user was offered to restore it).

## Improvement backlog
**See `IMPROVEMENTS.md`** for every suggestion made so far, prioritized (P0 credibility → P3 technical limits), with status and quota cost.

## Hackathon plan (agreed direction, not yet done)
1. Pin TabPFN v3.5 in `server.py` and show the version in the app.
2. Fix movement-uncertainty calibration (e.g. add grid-interpolation spread from neighbouring TabPFN predictions; stay TabPFN-only).
3. Evidence: small separate evaluation script with cached outputs: court-change adaptation (TabPFN vs frozen vs RLS, time to recover) as the headline; landing accuracy vs physics/kNN/linear; learning curve; a few closed-loop runs. Report honestly, incl. losses.
4. In-app "TabPFN-3.5 is learning" live chart; a "Show technical details" toggle button; optional 2-shot planning for shot choice.
5. Token-free demo/replay mode with recorded TabPFN outputs; 90-second video (user records); README as pitch (GIF, diagram, results, 3-command setup).
6. git repo + license; decide on `prompts/` and the CSV.
The user wants to give it their absolute best (aiming for top 3). Ask for the submission deadline if unknown.

## Next step in progress: local inference
The user wants local TabPFN inference to avoid quota. The `tabpfn` package (PyPI, v9.x) supports v2/v2.5/v3/**v3.5 (default)**/v3.5-Fast;
weights download from Hugging Face after a one-time licence acceptance (browser, or `TABPFN_TOKEN` for headless);
TabPFN-3.5 is non-commercial licensed (fine for a hackathon). Select with `create_default_for_version(ModelVersion.V3_5)`.
The Windows laptop (i5-1335U, Iris Xe, no CUDA) would be CPU-only; **the Mac (Apple Silicon -> MPS GPU) is better**.
Planned approach: measure first (our 3 workloads: landing ~40 rows -> 1 query; dynamics 300x8 -> 625 queries; shot ~20x12 -> 24 queries;
v3.5 and v3.5-Fast; n_estimators 1/2/4/8; fit caching), then add `TABPFN_BACKEND=local|api` to `server.py` with the same endpoints,
so the game, the provenance rule and the tags stay unchanged. Local outputs should still be tagged `tabpfn` (it is the TabPFN model).

## Working conventions
- Never print, log or export the token. `.env` is never served.
- Keep the TabPFN-only rule intact in play mode; any non-TabPFN method goes in `dev-tools.js`.
- After edits: `node run_selftest.js` must stay all-PASS. M1 behaviour (default movement, determinism, replay) is covered by tests.
- Be careful with the API quota: estimate calls before live runs; avoid burning tokens on exploratory tests.
