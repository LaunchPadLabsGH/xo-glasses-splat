# Data flow and access: starter assessment

This is the starting point for essential requirement 1 of the challenge: *"Assess the data generated
from smart glasses and provide a report on where the data may go and who may have access."*

It covers the proposed primary device, the **XREAL Aura** (Android XR), using public information only.
Every row is marked **To verify** until it has been tested during the Week 1 device gate. Testing will
use network traffic capture, permission and account audits, and a review of each party's terms.

## Data generated

| Data | Source | Personal data? | Notes |
|---|---|---|---|
| RGB video and stills | Glasses camera | Yes (bystanders' faces, number plates, locations) | Core input to mapping |
| Tracking camera images | Glasses tracking cameras | Possibly | Used by the platform for head tracking; third-party access unknown |
| IMU (gyro, accelerometer) | Glasses | Indirectly (gait, behaviour) | Needed for metric scale and head-turn handling |
| Audio | Glasses microphones | Yes (voices, conversations) | Only event timing and loudness are kept by this pipeline |
| Location | Paired device / puck | Yes | Not required by the pipeline; to be disabled |
| Derived 3D model | This pipeline | Yes if faces or plates survive | Faces are blurred before reconstruction |
| Device and app telemetry | Platform services | Yes (device IDs, usage) | Main unknown: what leaves the device by default |

## Where data could go

| Path | Party | Default behaviour | Control in our design | Status |
|---|---|---|---|---|
| On-device storage (glasses/puck) | Wearer's organisation | Stored locally | Encrypted storage; offline operation; download at base only | To verify |
| Platform account and sync | Google (Android XR), XREAL | Account required for setup on most Android devices | Dedicated, non-personal accounts; cloud backup and sync disabled | To verify |
| AI assistant features | Google (Gemini) | May process camera and audio in the cloud when invoked | Disabled; not used by the pipeline | To verify |
| Telemetry and crash reports | Google, XREAL, Qualcomm | Typically on by default | Opt out where possible; capture traffic to confirm | To verify |
| App updates and Play services | Google | Network access required | Updates applied before deployment; network off during missions | To verify |
| Transfer to base | Wearer's organisation | USB or local network | Wired transfer to an offline workstation | Design |
| Processing at base | Wearer's organisation | Local GPU workstation | No cloud processing (HMGCC clarification Q1.54) | Design |

## Jurisdiction and legal access (to assess)

- US-headquartered platform providers may be subject to US legal process (for example the CLOUD Act)
  for any data held on their services. Keeping all data off vendor services removes this route.
- Device and component manufacturers' locations, and any data they collect, will be documented.
- UK GDPR / DPA 2018 apply to all capture of identifiable people; see `PRIVACY.md`.

## Terms of use (to check at Week 1 gate)

- XREAL developer terms and Google Android XR / Play developer terms: check for any clause restricting
  defence, security or surveillance use. HMGCC will not seek exceptions (clarification Q1.49).
- Confirm third-party apps can read the RGB camera and IMU, with timestamps on a shared clock.
- Confirm the device works fully offline after setup.
