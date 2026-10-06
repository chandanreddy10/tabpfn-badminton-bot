#!/bin/sh
# Cut a short GIF for the README from the 1920x1080 demo video: tools/make_gif.sh START_SECONDS DURATION_SECONDS
# The court and the sidebar (scoreboard, status, learning chart) are cropped and placed side by side,
# so the text stays readable at README size.
set -e
cd "$(dirname "$0")/.."
START=${1:-0}; DUR=${2:-12}
ffmpeg -y -loglevel error -ss "$START" -t "$DUR" -i demo/demo.mp4 -filter_complex \
  "[0:v]crop=820:1080:380:0[court];[0:v]crop=340:1080:1580:0[side];[court][side]hstack,fps=12,scale=760:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4" \
  demo/demo.gif
ls -lh demo/demo.gif
