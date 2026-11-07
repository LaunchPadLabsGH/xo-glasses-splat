# Privacy and lawful capture

## In the pipeline

- **Faces are blurred before reconstruction.** The `privacy` stage detects faces with YuNet (OpenCV,
  MIT licence). It pixelates and blurs them in every keyframe *before* SfM or splat training, so no
  identifiable face can be baked into a 3D model.
- **Conservative by design.** The detector is tuned to blur when in doubt, so it also fires on some
  face-like patterns (on the desk benchmark these are false positives on monitors and papers).
  Over-blurring costs a little texture; under-blurring would leave personal data in the model.
- **Audio.** Only the timing and loudness of acoustic events are kept. Speech is never transcribed or
  stored by the pipeline.
- **Raw footage is never committed.** `data/` and `runs/` are git-ignored; only derived, face-blurred
  web assets are published.
- **Not yet handled:** vehicle number plates (planned for the funded project). Public demo scenes are
  reviewed by hand before publishing.

## Capture protocol for project footage

1. Capture on **private premises** with the owner's permission, or with consenting participants only.
   No recording in sensitive locations (HMGCC clarification Q1.1).
2. Brief anyone present; display notices on site.
3. **Never obscure or disable the glasses' recording indicator light.** Doing so breaches manufacturer
   terms and the principle of transparency.
4. Use dedicated project devices and accounts; disable cloud sync, backup and AI assistant features.
5. Transfer footage by cable to an offline workstation; delete it from the device after transfer.
6. Retain raw footage only for the project's duration; keep derived, face-blurred outputs.

## DPIA checklist (to complete before capture)

- [ ] Purpose and lawful basis recorded (legitimate interests or consent)
- [ ] Data minimisation: only the frames and sensors needed; location off
- [ ] Bystander risk assessed for each site; notices in place
- [ ] Storage: encrypted, access-controlled, offline
- [ ] Retention and deletion schedule agreed
- [ ] Processor review: no vendor cloud services in the data path (see `DATA_FLOW.md`)
- [ ] Subject-rights process (access and erasure requests) in place
