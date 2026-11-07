#!/usr/bin/env bash
# Reproduce the demo scenes end to end.
#   CPU stages run anywhere; `xgs train` needs the CUDA GPU (scripts/setup_wsl_gpu.sh).
#   Set GPU=0 to skip training (the viewer then shows the training-free layers).
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
GPU="${GPU:-1}"

scene() {  # name source device [overrides...]
  local name="$1" src="$2" dev="$3"; shift 3
  [ -e "$src" ] || { echo "skip $name: $src missing (bash scripts/fetch_demo_data.sh)"; return 0; }
  local sets=(); for o in "$@"; do sets+=(--set "$o"); done
  echo; echo "=== $name ==="
  xgs run "$src" --run "runs/$name" --device "$dev" "${sets[@]}"
  if [ "$GPU" = 1 ]; then xgs train --run "runs/$name" --device "$dev" "${sets[@]}"; fi
  xgs report --run "runs/$name"
  xgs export --run "runs/$name" --name "$name" --device "$dev" "${sets[@]}"
}

# Benchmark with ground truth: handheld desk sequence (pose accuracy in cm).
scene tum-desk data/raw/tum/rgbd_dataset_freiburg1_desk tum_fr1 ingest.fps=30
# Real smart-glasses footage: Google Glass, Washington DC waterfront (first 110 s).
scene dc-glass data/raw/dc_google_glass.webm google_glass ingest.start_s=0 ingest.duration_s=110 ingest.fps=4
# Urban walk, London Docklands (4K source, downscaled).
scene london-walk data/raw/london_canada_water.webm generic_video ingest.fps=3 ingest.max_width=1920

echo; echo "Done. Preview locally:  (cd docs && python3 -m http.server 8000)  then open http://localhost:8000"
