// Build the no-install demo: one HTML file with the recording embedded (opens with a double-click,
// or from any static host). node tools/build_demo.js [demo/recording.json] -> demo/index.html
const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..');
const recPath = process.argv[2] || path.join(ROOT, 'demo', 'recording.json');
const html = fs.readFileSync(path.join(ROOT, 'web', 'badminton.html'), 'utf8');
const rec = fs.readFileSync(recPath, 'utf8').replace(/<\//g, '<\\/');      // never close the script tag early
const marker = "<script>\n'use strict';";
if (!html.includes(marker)) throw new Error('main script not found');
const out = html.replace(marker, `<script>window.DEMO_RECORDING = ${rec};</script>\n${marker}`)
  .replace(/<title>[^<]*<\/title>/, '<title>Badminton vs TabPFN-3.5: recorded match</title>');
fs.writeFileSync(path.join(ROOT, 'demo', 'index.html'), out);
console.log(`wrote demo/index.html (${(out.length / 1e6).toFixed(2)} MB)`);
