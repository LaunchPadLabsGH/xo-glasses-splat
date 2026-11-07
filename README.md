# xo-glasses-splat

**Smart-glasses footage to live 3D maps and Gaussian Splatting models, processed on worn and local hardware only.**

X-O Labs proof of concept for the HMGCC Co-Creation challenge *Smart glasses for mapping* (CH-2026-011).
It shows the pipeline from the proposal working end to end today:

- a **live path** that tracks the wearer and builds a sparse map frame by frame within a real-time budget;
- a **post-mission path** that refines poses and trains a photorealistic **Gaussian Splatting** model;
- **accuracy measured against ground truth**, not asserted;
- **privacy built in**, with faces blurred before anything is reconstructed;
- **no cloud services at any stage**.

**[Open the 3D viewer](https://launchpadlabsgh.github.io/xo-glasses-splat/)**: orbit the map, switch layers, walk the camera route.

![Viewer showing the sparse map and camera path of the benchmark scene](docs/img/viewer-tum-desk.png)

## Measured results

Benchmark: **TUM RGB-D `freiburg1_desk`**, a handheld camera with fast, jerky motion (a reasonable
stand-in for head-worn capture), with motion-capture ground truth. 407 frames, 20 s, 640×480.

| Measure | Result |
|---|---|
| Keyframes kept (blur, motion and redundancy filtered) | 220 of 407 |
| Images posed after the mission | 219 of 220 (99.5%) |
| Mean reprojection error | 0.92 px |
| **Post-mission pose error vs ground truth (ATE RMSE)** | **3.5 cm over a 9.4 m path (0.37%)** |
| Live front end, time per frame (p50 / p95, CPU) | 11.1 / 29.0 ms |
| Live frames within a 30 fps budget | 97.1% |
| Live pose drift, longest tracked segment | 11.4 cm over 2.8 m (4.1%) |
| Map points flagged low-confidence | 13.1% |
| Gaussian Splatting PSNR on held-out views | *pending RTX run* |

<p>
<img src="docs/assets/tum-desk/trajectory.png" width="49%" alt="Estimated camera paths against motion-capture ground truth">
<img src="docs/assets/tum-desk/keyframes.png" width="49%" alt="Keyframe selection by sharpness over time">
</p>

## How it works

```mermaid
flowchart LR
    CAP[Glasses capture<br/>video, IMU, audio] --> SEL[Keyframe selection]
    SEL --> LIVE[Live tracking +<br/>sparse map]
    SEL --> PRIV[Face blurring]
    PRIV --> SFM[Pose refinement<br/>COLMAP]
    SFM --> CONF[Confidence layer]
    SFM --> GS[Gaussian Splatting<br/>local GPU]
    GS --> VIEW[Analyst viewer]
    CONF --> VIEW
    LIVE --> VIEW
```

1. **Keyframe selection.** Drops motion-blurred frames and fast head turns (using IMU data when
   available), and keeps frames with enough parallax. Every dropped frame has a recorded reason.
2. **Face blurring.** Runs before reconstruction, so no identifiable face reaches a 3D model.
3. **Live path.** Frames are processed once, in capture order, and each is timed against the frame
   budget: a stand-in for tracking on the glasses' compute puck.
4. **Post-mission refinement.** COLMAP Structure-from-Motion, then Gaussian Splatting training with
   [gsplat](https://github.com/nerfstudio-project/gsplat) on a local NVIDIA GPU.
5. **Confidence.** Each map region is scored by how many viewpoints saw it and from how wide an angle.
   Analysts see what was observed properly and what is a guess.
6. **Audio events.** Loud acoustic events are pinned to where the wearer was. Speech is never transcribed.

Details: [Architecture](docs/ARCHITECTURE.md) · [Data flow and access](docs/DATA_FLOW.md) ·
[Privacy](docs/PRIVACY.md) · [Capture protocol](docs/CAPTURE_PROTOCOL.md) · [Roadmap to TRL 4](docs/ROADMAP.md)

## Honest limitations

- **No smart-glasses hardware yet.** Demo inputs are public footage, including real Google Glass video.
  The proposed XREAL Aura is confirmed at the project's Week 1 device gate.
- **The live path is a CPU proxy.** It is a monocular visual-odometry front end timed on an x86 CPU,
  not on the Aura's Snapdragon puck. It has no loss-of-tracking relocalisation and drifts, as the
  numbers above show. The funded project replaces it with visual-inertial odometry, where the IMU gives
  metric scale and copes with fast head turns.
- **Scale.** Monocular footage has no absolute scale. The benchmark is scaled by alignment to ground
  truth. For own footage, printed ArUco markers at measured positions give metric scale and an error
  figure ([capture protocol](docs/CAPTURE_PROTOCOL.md)).
- **Privacy.** Face blurring is deliberately conservative and also fires on some face-like patterns.
  Number-plate blurring is not yet implemented.

## Run it

CPU stages (any Linux, macOS or WSL machine with Python 3.10+ and ffmpeg):

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
bash scripts/fetch_demo_data.sh tum
xgs run data/raw/tum/rgbd_dataset_freiburg1_desk --run runs/tum-desk --device tum_fr1 --set ingest.fps=30
xgs report --run runs/tum-desk
```

Gaussian Splatting training (NVIDIA RTX GPU; tested setup is Windows 11 + WSL2 Ubuntu with an RTX 5070):

```bash
bash scripts/setup_wsl_gpu.sh
xgs train --run runs/tum-desk --device tum_fr1
xgs export --run runs/tum-desk --name tum-desk --device tum_fr1
```

All demo scenes end to end: `bash scripts/run_demo.sh`. To view locally:
`(cd docs && python3 -m http.server 8000)`, then open http://localhost:8000.

Every stage writes a `manifest.json` with its parameters, code hash, timings, metrics and library
versions. Re-runs skip unchanged stages, which keeps results reproducible.

## Licence

Copyright © 2026 X-O Labs Ltd. All rights reserved. Shared for evaluation; see [LICENSE](LICENSE).
Third-party components are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and demo data
attribution in [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).
