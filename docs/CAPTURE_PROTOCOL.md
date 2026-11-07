# Capture protocol for own footage, with ground truth

HMGCC provides no test site or dataset, so we arrange both (clarifications Q1.28, Q1.53). This protocol
gives a repeatable capture with a **metric accuracy check**, without survey equipment.

## Site

- A representative outdoor built environment on private premises (an accepted demonstration setting,
  Q1.38), for example a business park courtyard or building frontage. Aim for 20–60 m of facade and
  some depth: steps, recesses, street furniture.
- Daylight, ideally overcast (soft, even light). Add a separate dusk pass for the low-light assessment.

## Markers (ground truth)

1. Print 6–8 ArUco markers (dictionary `DICT_4X4_50`, IDs 0–7) at A4 size, about 18 cm square.
   Mount them flat on walls or boards around the site at different heights and depths.
2. Measure the distances between marker centres with a tape or laser measure. Record them in
   `markers.yaml` as 3D coordinates, or as pairwise distances.
3. Optional: scan the site with a phone LiDAR app as a dense reference for cloud-to-model distance.

`xgs.sfm.scale.marker_scale` turns these into a metric scale and reports distance errors (cm and % of
distance).

## Walk

- Walk at normal pace along the facade, then back on a parallel line 2–3 m further out.
- Look around naturally, including some fast head turns: the pipeline is meant to cope with them.
- Finish where you started (a loop helps the reconstruction close up).
- 2–4 minutes per pass. Record a second pass for repeatability.

## Metadata to record

Device and firmware, start time, lighting, weather, route sketch, marker table, and anyone present (and
their consent).

## Running it

```bash
xgs run data/raw/site_pass1.mp4 --run runs/site1 --device xreal_aura --imu data/raw/site_pass1_imu.csv
xgs train --run runs/site1 --device xreal_aura
xgs report --run runs/site1
xgs export --run runs/site1 --name site-1 --device xreal_aura
```
