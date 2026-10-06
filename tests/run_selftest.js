// Headless self-tests: node tests/run_selftest.js
// Runs the play-mode checks from badminton.html plus the developer checks from dev-tools.js
// (no browser, no server, no TabPFN calls). Prints PASS/FAIL per check and a total.
const fs = require('fs');
const path = require('path');
const html = fs.readFileSync(path.join(__dirname, '..', 'web', 'badminton.html'), 'utf8');
const src = html.match(/<script>([\s\S]*)<\/script>/)[1];
const dev = fs.readFileSync(path.join(__dirname, '..', 'web', 'dev-tools.js'), 'utf8');
const api = new Function(src + '\n' + dev + '\n;return { SelfTest, DevTools: globalThis.DevTools };')();
(async () => {
  const res = await api.SelfTest.runAll();
  if (api.DevTools) res.push(...await api.DevTools.selfTest());
  for (const r of res) console.log((r.ok ? 'PASS ' : 'FAIL ') + r.name);
  const pass = res.filter(r => r.ok).length;
  console.log(`${pass}/${res.length}`);
  process.exit(pass === res.length ? 0 : 1);
})().catch(e => { console.error('CRASH', e); process.exit(1); });
