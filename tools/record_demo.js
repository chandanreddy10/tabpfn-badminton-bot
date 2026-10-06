// Record a watch-mode session for the no-install demo.
//   python server.py --port 8765            (local TabPFN-3.5, in another terminal)
//   node tools/record_demo.js --warm 15 --rec 8 --style pattern
// The warm-up part lets the bot build its contexts (normal world, no recording); the recorded part switches
// the shuttle world every 5 rallies. Writes demo/recording.json. Every prediction in it is TabPFN-3.5's own.
const fs = require('fs');
const os = require('os');
const path = require('path');
const { launch, sleep } = require('./cdp');
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const PORT = +arg('port', 8765), WARM = +arg('warm', 15), REC = +arg('rec', 8), STYLE = arg('style', 'pattern');
const SPEED = arg('speed', '4'), EVERY = arg('world-every', '5');
const OUT = path.join(__dirname, '..', 'demo', 'recording.json');

(async () => {
  const b = await launch({ port: 9341, profile: path.join(os.tmpdir(), 'badminton-demo-profile'), width: 1400, height: 1000 });
  const status = () => b.evaluate(`JSON.stringify({ mode: App.mode, world: App.S.cfg.world || 'normal', score: App.S.rules.scores,
      landings: LearningChart.n, antRows: App.bot.anticip.rows(), antAcc: App.bot.anticip.accuracy(), paused: App.paused,
      ticks: Recorder.rec ? Recorder.rec.ticks : 0, ms: TabPFNHealth.lastMs })`);
  await b.send('Page.navigate', { url: `http://localhost:${PORT}/?dev=0` });
  await sleep(4000);
  await b.evaluate(`(async () => { for (let i = 0; i < 120 && !TabPFNClient.ready; i++) { await TabPFNClient.health(); await new Promise(r => setTimeout(r, 2000)); } return TabPFNClient.ready; })()`);
  await b.evaluate(`document.getElementById('style-sel').value = '${STYLE}'; document.getElementById('world-every').value = '0';
                    document.getElementById('style-every').value = '0'; document.getElementById('watch-speed').value = '${SPEED}';
                    App.startBot(true); 'ok'`);
  console.log(`warming up for ${WARM} min (normal world, style ${STYLE}, not recorded)`);
  for (let t = 0; t < WARM * 60; t += 30) { await sleep(30000); console.log(`  warm t+${t + 30}s`, await status()); }
  // Start recording at the beginning of a fresh match, then switch worlds every few rallies.
  await b.evaluate(`App.newMatch(); App.ralliesInWorld = 0; document.getElementById('world-every').value = '${EVERY}'; App.startRecording()`);
  console.log(`recording for ${REC} min (world switch every ${EVERY} rallies)`);
  for (let t = 0; t < REC * 60; t += 30) { await sleep(30000); console.log(`  rec t+${t + 30}s`, await status()); }
  const json = await b.evaluate('JSON.stringify(App.stopRecording())');
  fs.writeFileSync(OUT, json);
  console.log(`wrote ${OUT} (${(json.length / 1e6).toFixed(2)} MB)`);
  b.close();
})().catch(e => { console.error(e); process.exit(1); });
