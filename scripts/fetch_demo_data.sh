#!/usr/bin/env bash
# Download the public demo data into data/raw/ (never committed).
# Licences and attribution: docs/DATA_SOURCES.md
#
# Usage:  bash scripts/fetch_demo_data.sh [tum] [glass] [london]   (default: all)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
UA="xo-glasses-splat/0.1 (research prototype; https://github.com/LaunchPadLabsGH/xo-glasses-splat)"
want() { [ $# -eq 0 ] && return 0; for a in "${ARGS[@]}"; do [ "$a" = "$1" ] && return 0; done; return 1; }
ARGS=("$@")
[ ${#ARGS[@]} -eq 0 ] && ARGS=(tum glass london)

fetch() {  # url dest  (Wikimedia rate-limits busy IPs: back off and retry)
  local url="$1" dest="$2"
  [ -s "$dest" ] && { echo "have $dest"; return 0; }
  for i in 1 2 3 4 5 6; do
    code=$(curl -sSL -A "$UA" -o "$dest.part" -w "%{http_code}" "$url" || echo 000)
    if [ "$code" = 200 ]; then mv "$dest.part" "$dest"; echo "got $dest"; return 0; fi
    echo "HTTP $code for $dest, retrying in $((i * 30)) s"; sleep $((i * 30))
  done
  rm -f "$dest.part"; echo "failed to download $url" >&2; return 1
}

if want tum; then
  # TUM RGB-D fr1/desk: handheld camera with motion-capture ground truth (CC BY 4.0)
  if [ ! -d data/raw/tum/rgbd_dataset_freiburg1_desk ]; then
    mkdir -p data/raw/tum
    fetch https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz data/raw/tum/fr1_desk.tgz
    tar -xzf data/raw/tum/fr1_desk.tgz -C data/raw/tum && rm data/raw/tum/fr1_desk.tgz
  fi
fi
if want glass; then
  # Real Google Glass footage, Washington DC (CC BY 3.0, brownpau)
  fetch "https://upload.wikimedia.org/wikipedia/commons/f/f2/Wintry_DC_through_Google_Glass.webm" \
        data/raw/dc_google_glass.webm
fi
if want london; then
  # 4K walk, Canada Water, London (CC BY-SA 4.0, Acabashi)
  fetch "https://upload.wikimedia.org/wikipedia/commons/8/86/Canada_Water_and_Rafter_Walk_in_Southwark%2C_London_Docklands.webm" \
        data/raw/london_canada_water.webm
fi
echo "Data in data/raw/:"; ls -la data/raw
