# Badminton: play singles against TabPFN-3.5

`badminton.html` is a top-down singles badminton game. You play against **TabPFN-3.5**, an opponent that learns the court as you play, or against a friend on one keyboard.
- **Every learned or predictive part of the bot comes from TabPFN:** where the shuttle will land, how the bot itself moves, and the uncertainty used by the planner.
- **Planner:** CEM model predictive control with a Monte Carlo chance constraint.
- **Other methods** (filters, kNN, RLS, gradient boosting, and so on) exist only in `dev-tools.js`, a developer module that is loaded on demand.

## Run

```
pip install -r requirements.txt
# .env: TABPFN_TOKEN=...   (https://platform.priorlabs.ai/account/api-keys)
python server.py           # imports tabpfn-client (~10 s), then serves http://localhost:8000
```

- Open `http://localhost:8000` and click **Play vs TabPFN-3.5**. **Two players, one keyboard** is the other option.
- The page shows only what a player needs: score, whose serve, plain-language hints, TabPFN-3.5's status and confidence, settings and the shot log. If the opponent can't be reached (no server, no token, usage limit), the start screen and the pause screen say so in plain words.
- **Colour themes:** Classic (the original), Midnight, Clay and Arena, switched with the swatches under the title. Each has a light and a dark version that follows the system setting; the choice is remembered in the browser.
- **Developer mode is on by default:** technical panels (TabPFN call counters, ≈ tokens, latency, provenance and calibration badges, context sizes, what the model sees), the Developer tools panel (collapsed; it loads when opened), the seed field, the debug view (`G`) and visible provenance-guard errors.
- `?dev=0` gives the clean player view without them; `?dev=1` also opens the Developer tools panel straight away.
- `#selftest` runs all checks in the browser.
- Before the first game, a **How to play** screen explains the controls and rules; press **H** at any time to bring it back (the game pauses; H, Esc or the button resumes).
- `?start=bot` skips the start screen and the instructions.

## How the bot works (play mode)

**Warm-up.** Before the first rally, a machine fires about 20 practice shots while the bot moves with random exploratory commands and hits 10 practice shots of its own. This fills TabPFN's context with real landings and movements.
- No TabPFN calls are made during warm-up.
- It can be skipped. It's skipped automatically when a stored context is large enough (saved in localStorage as rows only).
- A "low context" notice shows while the context is still small.

| Part | TabPFN rows | Target | When it's called |
|---|---|---|---|
| Landing | time since hit; observed x/y/z at each checkpoint so far; finite-difference velocity; shot-type cue; hitter x/y; bot x/y | landing minus the last observed position (x, y), and the remaining flight time, as 10/50/90 % quantiles | 0.2 / 0.4 / 0.6 s after the opponent's hit; the estimate is carried forward between checkpoints |
| Movement dynamics | x, y, vx, vy, ux, uy, previous ux, previous uy | Δvx, Δvy per 0.1 s | the context is refit as new movement arrives; then one batched grid query (see below) |
| Own shot choice | shot type, hitter x/y, target x/y, distance, game time, last-3 miss | landing minus target | once per bot hit, for 24 random candidates in one batch |

- **Planner:** CEM (48 candidates × 4 iterations over 1.2 s of commands).
  - It is warm-started by shifting the previous plan.
  - Each candidate gets 8 Monte Carlo rollouts, which include TabPFN's movement uncertainty and a landing point and arrival time sampled from TabPFN's landing quantiles.
  - The cost penalizes any shortfall of P(distance to landing ≤ r at arrival) below 0.9, plus terminal distance, control effort, and recovery to the base position.
  - The horizon is tied to TabPFN's predicted arrival time.
  - One replan takes about 9 ms.
- **Shot choice:** score = P(lands in) × P(opponent can't reach), both computed from TabPFN's predicted landing spread and the opponent's distance and speed limit.

### Why the movement model is a "TabPFN grid surrogate"

The spec offered two options: a strict per-step TabPFN rollout, or a local linearization built purely from TabPFN queries. Measurements decided between them:
- **Latency:** each predict takes 1.1–1.5 s with a cached fit and 4–6 s with a fresh fit. A strict rollout needs 12 sequential calls per plan, so 15 s or more.
- **Rate and quota:** the API allows **60 predict requests per minute**, and every request costs about **10,000 tokens** regardless of size (quoted by the API's cost estimator). The account has a daily cap of 5M and a monthly cap of 20M tokens.
- **Accuracy:**
  - Direct TabPFN predictions of Δv are very accurate: 0.047 m/s error, versus 0.75 m/s for predicting no change, with 98–99 % interval coverage.
  - A local Jacobian at one command point does not extrapolate, because acceleration saturates (1.06 m/s error).
  - A polar command lattice was 0.24 m/s off even with exact values.
  - A Cartesian grid interpolates well.

The default is therefore **one batched TabPFN query on a 5×5 velocity × 5×5 command grid (625 rows)**, interpolated multilinearly and tagged `tabpfn-linearized`.
- Every value in it is TabPFN's own prediction, and TabPFN's quantile spread is carried along.
- It is rebuilt only after the movement context is refit, at most every 6 s, so the planner never waits on the network.
- The **strict per-step rollout** is still available as a clearly labelled developer setting.
- **Limitation:** the grid queries assume the previous command equals the current one. Command-lag effects are learned from the context rows but are only exactly represented in strict mode.

### Cost control
- **Stacked targets:** by default, the targets of one task are stacked into **one** TabPFN regression: each target is standardized and a target-indicator column is added. This costs one request instead of 2–3. The spec asked for one regressor per coordinate; that layout is one switch away, in the developer tools or with `BOTCFG.stackTargets = false`.
- **Rate budget:** a client-side budget respects the per-minute limit. Landing and shot calls take priority, and movement refreshes keep a reserve. A landing checkpoint that is over budget is skipped and the previous estimate carried forward; another method is never substituted.
- **Usage display:** the panel shows the estimated tokens used this session.

### Degraded behaviour (never another predictor)
1. **A TabPFN call is late:** the bot keeps replanning on the current TabPFN surrogate and the carried-forward landing estimate. If the surrogate is stale (more than 20 s old with failing refreshes), it follows the shifted previous plan.
2. **The plan runs out:** the bot brakes (zero command) and shows *"TabPFN slow: bot is waiting"*.
3. **TabPFN unavailable for more than 5 s** (errors, rate limit or quota): the rally pauses with *"TabPFN unavailable"* and a **Retry** button.
4. **Slow motion:** while a call that matters is pending (an opponent shot in flight, or the bot must hit), the game runs at 5 % speed. This is adjustable and can be turned off. It never blocks rendering.

### Enforcement of the TabPFN-only rule
- **Provenance tags:** every provider output carries one. `ProvenanceGuard` admits only `tabpfn` and `tabpfn-linearized` in play mode.
  - In production it refuses anything else, so the bot brakes.
  - In developer mode (the default) it also throws a visible error.
  - Outputs from the server's mock backend (`backend:mock`) are refused too.
- **Separate code:** play-mode code contains no non-TabPFN provider. A static check confirms that none of the developer class names appear in `badminton.html`'s script. `dev-tools.js` is requested only when the developer panel is opened or `#selftest` runs; the server log shows this.
- **Tests:** they inject `knn`, `oracle`, filter and mock outputs and assert that each is refused and that the bot never moves on them.
- **On screen:** a **"100% TabPFN"** badge with per-task call counters, a calibration badge (90 % coverage for landing and movement), a confidence meter (P reach), the status badge (connected / slow / rate-limited / unavailable) and the latencies.

### Court change, settings, replay
- **"Slippery court / tired legs"** sets grip × 0.35 and speed × 0.75 mid-rally for both players. It's recorded as an event in the action log, so replays stay exact. The bot isn't told; it adapts as the recency window of movement rows refills.
- **Reaction time:** against the bot, its shots towards you fly **3x slower** (setting "Bot's shots fly slower", 1–4x). They land on exactly the same spot; only the flight is stretched in time, so wind and noise are unchanged. Shots towards the bot keep normal speed.
- **Settings:** wind, noise, r, speed, friction, grip, command lag, the bot-shot slow-down and slow motion, in a collapsible section. Defaults reproduce Milestone 1's movement exactly; this is tested over 3000 ticks.
- **Replay:** replays use the recorded actions and make **no TabPFN calls**. Provider outputs are logged per rally.
- **Determinism:** the simulation is deterministic. Which TabPFN answer arrives at which tick depends on network timing, so a live bot game can't be reproduced from the seed alone. Recorded actions and outputs make it replayable.

## Developer tools (`dev-tools.js`, hidden by default)
- **Session-only provider switching**, never saved; a reload always returns to TabPFN-only play:
  - landing: TabPFN, kNN, kernel, gradient boosting, ensemble, linear extrapolation, physics fit (with a constant-wind mismatch option), EKF, particle filter, oracle;
  - dynamics: TabPFN, kNN, RLS, RLS + CUSUM, frozen, gradient boosting, ensemble, oracle;
  - planner: CEM, random shooting, MPPI.
- **Monitoring:** status, calibration per provider, and the provenance call log.
- **Compare estimators on this shot:** every method on the last opponent shot, using the TabPFN outputs recorded during play, so no new calls.

## Measured so far (real TabPFN, before the daily quota ran out)
- **Live play-mode run** against a scripted human-paced opponent (Milestone 3 surrogate):
  - the bot reached 6 of 9 opponent shots and won 5 of 8 rallies;
  - 100 % of the outputs it used had provenance `tabpfn` / `tabpfn-linearized`, with 0 refused;
  - landing error median 0.30 m; landing 90 % coverage 100 % (n = 7); own-shot coverage 100 % (n = 10);
  - movement 90 % coverage 56 %, so its uncertainty is somewhat underestimated;
  - median call 1.4 s.
- **Offline accuracy:** TabPFN predicts movement Δv with 0.047 m/s error and 98–99 % coverage on held-out transitions.

## Files
| File | Purpose |
|---|---|
| `badminton.html` | game, play-mode TabPFN bot, self-tests |
| `dev-tools.js` | developer module (all non-TabPFN providers), lazily loaded |
| `server.py` | local server and TabPFN proxy: `/api/predict` (stacking), `/api/fit`, `/api/gbr` (developer only) |
| `.env` | `TABPFN_TOKEN` (never sent to the browser, never served, never logged) |
