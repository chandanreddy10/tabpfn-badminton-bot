// Minimal Chrome DevTools Protocol helper for the demo tools (headless Chrome, no extra packages).
const { spawn } = require('child_process');
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function launch({ port = 9340, profile, width = 1280, height = 720 } = {}) {
  const args = ['--headless=new', '--disable-gpu', `--remote-debugging-port=${port}`, `--window-size=${width},${height}`,
                '--hide-scrollbars', '--mute-audio', '--disable-background-timer-throttling', '--disable-renderer-backgrounding',
                '--disable-backgrounding-occluded-windows'];
  if (profile) args.push(`--user-data-dir=${profile}`);
  const proc = spawn(CHROME, [...args, 'about:blank'], { stdio: 'ignore' });
  let tabs = null;
  for (let i = 0; i < 60 && !tabs; i++) { await sleep(500); try { tabs = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); } catch (e) { /* starting */ } }
  if (!tabs) throw new Error('Chrome did not start');
  const ws = new WebSocket(tabs.find(t => t.type === 'page').webSocketDebuggerUrl);
  await new Promise(r => { ws.onopen = r; });
  let id = 0; const pending = {}; const listeners = [];
  ws.onmessage = m => { const j = JSON.parse(m.data); if (pending[j.id]) { pending[j.id](j); delete pending[j.id]; } else listeners.forEach(f => f(j)); };
  const send = (method, params = {}) => new Promise(r => { pending[++id] = r; ws.send(JSON.stringify({ id, method, params })); });
  const evaluate = async expr => {
    const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
    if (r.result && r.result.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails).slice(0, 500));
    return r.result && r.result.result ? r.result.result.value : undefined;
  };
  await send('Runtime.enable'); await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: false });
  const close = () => { try { ws.close(); } catch (e) { /* closed */ } proc.kill('SIGKILL'); };
  return { send, evaluate, close, onEvent: f => listeners.push(f) };
}
module.exports = { launch, sleep };
