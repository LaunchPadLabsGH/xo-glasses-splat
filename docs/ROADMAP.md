# From proof of concept to TRL 4

How this repository maps onto the proposal's six two-week sprints.

| Sprint | Weeks | Proposal deliverable | Already in this repository | Still to do in the project |
|---|---|---|---|---|
| 0 | Pre-start | Buy glasses, second device, survey kit | Device profiles, capture protocol | Procurement |
| Mobilise | 1–2 | Device gate: API access, terms, offline operation; acceptance metrics | Data-flow starter, metrics defined and computed | Hands-on Aura tests; traffic capture; terms review |
| Foundation | 3–4 | Daytime urban capture; quality pipeline | Ingest, keyframe selection, face blurring, SfM | Aura camera and IMU ingest; own-site captures with markers |
| Reconstruct | 5–6 | First reproducible 3D map and benchmarks | Splat training, confidence layer, ground-truth evaluation | Splat tuning for glasses footage; metric scale from IMU |
| Integrate | 7–8 | Capture-to-model workflow; edge profiling; analyst export | Live-path replay with latency, web viewer, exports | Port the live path to the Aura puck; visual-inertial odometry |
| Validate | 9–10 | Repeatability and robustness; privacy review; desirables | Manifests and reports for repeat runs; audio events | Low-light pass; second manufacturer; number-plate blurring |
| Deliver | 11–12 | TRL 4 evidence package and demonstration | Report generator, viewer | Final demonstration at HMGCC's indoor site, plus outdoor results |

## Phase 2 (indicative)

On-device splat refinement, multi-operator map merging, low-light capture with an external sensor
module if needed (HMGCC clarification Q1.4), and hardened encryption and key management.
