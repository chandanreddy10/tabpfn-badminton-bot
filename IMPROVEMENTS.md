# Improvement backlog

Every suggestion made during development, grouped and prioritized for the **TabPFN-3.5 Hackathon** submission.
Status: **open** unless marked otherwise. Quota cost: none unless noted (TabPFN API ≈ 10k tokens per request).

## P0: credibility (do first)
1. **Pin TabPFN v3.5** in `server.py` (`create_default_for_version(ModelVersion.V3_5)` or `model_path`), and have the server report the version it uses so the app shows the real one ("Model: TabPFN v3.5"). Today v3.5 is only the API default, so the "TabPFN-3.5" label is linked to the model by coincidence.
2. **Fix movement-uncertainty calibration.** The 90 % interval holds only ~56 % of outcomes. Likely cause: interpolating the 625-point TabPFN grid adds error that TabPFN's own spread doesn't include. Fix while staying TabPFN-only, e.g. widen σ by the spread between neighbouring TabPFN grid predictions, or add a finer grid near the current velocity. Target ≈ 90 %.
3. **Validate the latest changes live** (needs quota or local inference): target stacking, the global movement surrogate, 3× slower bot shots, the new UI. Play several full matches; check provenance stays 100 % TabPFN, interception rate, calibration, latency and tokens per match.

## P1: evidence (judges reward a performance showcase)
4. **Small, separate evaluation script** (outside the app; the in-app benchmark was removed on request), with every TabPFN answer cached so re-runs are free, and results saved to JSON/CSV files automatically:
   - **Headline: court-change adaptation.** Movement-prediction error over time when the court turns slippery: TabPFN-3.5 (recency window) vs a frozen model vs RLS (± change detection). Report time to recover.
   - **Landing accuracy** at 0.2 / 0.4 / 0.6 s: TabPFN-3.5 vs physics-only vs kNN vs linear extrapolation, calm and windy.
   - **Learning curve:** error vs number of context rows (5, 10, 20, 40).
   - **Closed loop:** interception rate of the full bot, a few short runs.
   - Honest reporting: confidence intervals, and say where a baseline wins.
   - Rough budget: ≈ 0.6–1M tokens (much less if local inference works).
5. **Have a clear answer to "why not just use physics?"** TabPFN never sees the hidden wind or the physics and still adapts to a court change. Back it with the headline numbers.
6. **Update the README's "Measured" section** with the new results (the current numbers are from an older build).

## P1: local inference (removes the quota problem)
7. **Local TabPFN-3.5 backend.** Use the `tabpfn` package (v9.x; supports v3.5 and v3.5-Fast; weights from Hugging Face after a one-time licence acceptance; non-commercial licence is fine for a hackathon). The Mac's Apple Silicon GPU (MPS) is preferred.
   - **Step 1: measure.** Run the three real workloads (landing ~40 rows → 1 query; dynamics 300×8 → 625 queries; shot ~20×12 → 24 queries) with v3.5 and v3.5-Fast, `n_estimators` 1/2/4/8, with and without fit caching.
   - **Step 2: integrate.** Add `TABPFN_BACKEND=local|api` to `server.py` with the same endpoints, so the game and provenance tags are unchanged (local outputs are still tagged `tabpfn`). Keep the API as a fallback.
   - If local is fast enough, the slow-motion while predicting could be reduced or dropped.

## P2: wow factor in the app
8. **Live "TabPFN-3.5 is learning" chart** in the panel: landing error per shot over the match, with a marker at the court change. Judges see learning happen live. Highest impact per hour.
9. **"Show technical details" toggle button** instead of the `?dev=` URL flag. Default to the clean player view for judges, with one click to reveal the engineering. (Developer mode is currently the default by the user's choice; revisit before submission.)
10. **Optional: true MPC for shot choice.** Plan two shots ahead (e.g. a drop to pull the opponent forward, then a clear). Today shot choice is a one-step model-based choice, not MPC. Strengthens the "agent" story.
11. **Token / quota display in the normal view:** a session estimate ("≈ 200k tokens used this session") and/or account usage from the server ("8.9M of 20M tokens, resets 1 Nov"), refreshed every minute; fetching usage costs no quota. The `/api/usage` endpoint was removed with the benchmark and would need to come back.

## P2: make judging effortless
12. **Token-free demo / replay mode:** ship a recorded match (actions + recorded TabPFN outputs) that plays back in the app without a server, token or quota.
13. **90-second demo video** recorded in a real browser (headless screenshots barely advance the game). Show: a rally with the shrinking landing ellipse, "Show TabPFN-3.5's planned path", the court change and the recovery, the learning chart. On macOS: Cmd+Shift+5. Claude can write the shot list and narration.
14. **README as the pitch:** one-line hook, GIF, a "three prediction tables" diagram, results table, 3-command setup, limitations.
15. **Short write-up:** the problem (rallies as three tabular tasks), why TabPFN-3.5 (in-context learning, no training, calibrated uncertainty), results, limitations (latency, cost) and how they were handled.

## P2: repository quality
16. **git repo + licence**; keep `.env` out (`.gitignore` already does); pin versions in `requirements.txt`; one command for tests (`node run_selftest.js`, already added).
17. **Decide on `prompts/` and `tabpfn_context_table.csv`** (created by the user): include or exclude from the submission.
18. **Optional: split the ~2.7k-line `badminton.html`** into modules (sim, bot, renderer, app) for readability, keeping the single-file build if wanted.

## P3: known technical limitations worth improving
19. **Smashes are hard for the bot:** they fly ~0.5 s and the first landing prediction comes at 0.2 s plus latency. Consider an earlier checkpoint (e.g. 0.1 s) or faster local inference.
20. **Command lag** is exact only in the slow strict mode: the movement grid assumes the previous command equals the current one. Could add a previous-command dimension to the grid (more rows) if lag matters.
21. **Latency dependence:** the game slows down while TabPFN predicts. Local inference (item 7) or fewer ensemble members may remove most of it.
22. **Live bot games aren't reproducible from the seed alone** (network timing). Inherent; recorded actions and outputs make them replayable, which item 12 builds on.
23. **Movement surrogate refresh:** rebuilt at most every 6 s, after a refit. With local inference it could refresh more often for faster adaptation after a court change.

## Done (for reference)
- How to play screen + `H` shortcut; product-grade player view; TabPFN-3.5 naming; "predicting" wording; colour themes; 3× slower bot shots; developer mode default; benchmark removed (on request); `CLAUDE.md` handover; `run_selftest.js`.
