// Render the recorded demo to a video, frame by frame (smooth, constant speed), then encode with ffmpeg.
//   node tools/render_video.js --from 0 --seconds 75 --speed 2.5
// Writes demo/demo.mp4. The GIF is cut from it with tools/make_gif.sh.
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const { launch, sleep } = require('./cdp');
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const FROM = +arg('from', 0), SECONDS = +arg('seconds', 75), SPEED = +arg('speed', 2.5), FPS = +arg('fps', 30);
const W = +arg('width', 1280), H = +arg('height', 720), LABEL = arg('label', '');
const ROOT = path.join(__dirname, '..'), FRAMES = path.join(ROOT, 'demo', '.frames'), OUT = arg('out', path.join(ROOT, 'demo', 'demo.mp4'));

(async () => {
  fs.rmSync(FRAMES, { recursive: true, force: true }); fs.mkdirSync(FRAMES, { recursive: true });
  const b = await launch({ port: 9342, width: W, height: H });
  await b.send('Page.navigate', { url: 'file://' + path.join(ROOT, 'demo', 'index.html') + `?capture=1&captionScale=${SPEED}` + (LABEL ? `&label=${encodeURIComponent(LABEL)}` : '') });
  await sleep(3000);
  const tpf = 120 * SPEED / FPS;                                  // game ticks per video frame
  if (FROM > 0) await b.evaluate(`demoAdvance(${Math.round(FROM * 120)})`);
  let acc = 0;
  const n = Math.round(SECONDS * FPS);
  for (let f = 0; f < n; f++) {
    acc += tpf; const k = Math.floor(acc); acc -= k;
    await b.evaluate(`demoAdvance(${k})`);
    await b.evaluate('new Promise(r => requestAnimationFrame(() => r(1)))');
    const shot = await b.send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync(path.join(FRAMES, String(f).padStart(5, '0') + '.png'), Buffer.from(shot.result.data, 'base64'));
    if (f % 150 === 0) process.stdout.write(`frame ${f}/${n}\n`);
  }
  b.close();
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-framerate', String(FPS), '-i', path.join(FRAMES, '%05d.png'),
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-movflags', '+faststart', OUT]);
  fs.rmSync(FRAMES, { recursive: true, force: true });
  console.log('wrote', OUT);
})().catch(e => { console.error(e); process.exit(1); });
