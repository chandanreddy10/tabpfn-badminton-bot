# TabPFN-3.5 for live play: a badminton bot that adapts while it plays


I built a badminton bot whose every prediction comes from TabPFN-3.5, in real time. Where will this shot land? How will my body move? Which shot should I play? Where will my opponent hit next? The bot answers each of these from a table of what has happened so far in the match. A planner then acts on those answers and on how unsure they are.

There's no physics engine and no training on the game. To test how well it adapts, I change the shuttle mid-match (feather, heavy, low gravity, a storm) without telling the bot. It recovers within 14–22 shots, and its 90% uncertainty ranges stay right 91–94% of the time in every world. That reliability is what lets it plan safely.

- **Demo (no install):** https://chandanreddy10.github.io/tabpfn-badminton-bot/
- **Video (80 s):** https://chandanreddy10.github.io/tabpfn-badminton-bot/demo.mp4
- **Code and setup:** https://github.com/chandanreddy10/tabpfn-badminton-bot
- **Experiments:** https://github.com/chandanreddy10/tabpfn-badminton-bot/blob/master/experiments/REPORT.md
