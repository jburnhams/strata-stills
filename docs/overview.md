# strata-stills: design brief

## 1. Goal

Given a folder of raw(ish) stills, produce a finished set with no manual editing: the best available source developed well, levelled, framed, labelled. A browser app (as in strata-360) lets a person review the proposals and override any decision; the default must already be good.

Non-goals for now: video (the drone's MP4/LRF/SRT files are noted by `scan` and left alone), cloud services, destructive edits of originals.

## 2. The first test shoot: `\\Server\Drone\20250402 - Hazelmere`

157 files, 71 shots (measured with `./stratastills scan`):
- DJI Mini 4 Pro (`FC8482`, 24 mm equivalent, f/1.7, 8064x6048 = 48 MP), shot 2025-04-02 and 04-05/06 around lat 51.5388, lon -0.1996.
- 62 stills as DNG (~98 MB) + JPG (~35 MB) pairs; 9 videos (MP4 + LRF proxy + SRT flight log) to skip.
- Some shots are straight down (gimbal pitch -90, relative altitude ~42 m: roofs and gardens), some oblique (pitch -20, ~69 m).
- Derived files already made in Lightroom (`-Pano.jpg/.dng`, `-Enhanced-NR.dng`, `Thumbs.db`): kept with their shot, never treated as sources, never overwritten.
- The DJI XMP block of each JPEG gives GPS, relative/absolute altitude, gimbal and flight roll/pitch/yaw. This is read cheaply (first 256 KB) by `stratastills.meta`.
- The raw DNG decodes with rawpy; the Lightroom panorama DNGs are linear DNGs that LibRaw rejects, which is fine because they are derived files.

## 3. Pipeline (mapped from strata-360)

Each stage keeps its result per shot with a version and the key it was made for (as `analysis/photo_analysis.py` in strata-360), so only what changed is redone.

| Stage | What it does | From strata-360 |
|---|---|---|
| ingest | group files into shots, choose the source (raw else JPEG), read metadata | `photos.py` |
| develop | raw to a working image: demosaic, white balance, exposure, tone, lens correction, noise | `analysis/exposure.py`, `analysis/palette.py` (the grade) |
| level | find the true horizon (oblique shots) or the dominant grid direction (nadir shots: roofs, roads) and rotate to it | `analysis/views.py`, `edit/aim.py`, the 360 gravity levelling |
| detect | people, cars, objects, faces (YOLO11, InsightFace; faces only to avoid/blur or keep, never identify without a profile) | `analysis/objects.py`, `analysis/people_detect.py` |
| label | scene description, tags, place (reverse geocode from GPS), a short caption | `analysis/scenes.py`, `places` |
| frame | choose crops (one or several aspect ratios) that keep the subject, respect thirds and the horizon, and avoid cutting through detections | `edit/framing.py`, `edit/aim.py`, `edit/attention.py` |
| score | plain quality measures (sharpness, exposure, haze) to rank the shots and flag rejects | `analysis/quality`, `edit/view_quality.py` |
| render | write the finished image(s) and a sidecar with every decision | `render/still.py` |

## 4. Principles

- Originals are read-only. Output (and all analysis) goes to a separate project folder.
- Every automatic decision is recorded (what, why, with what confidence) so the GUI can show it and the person can override it; nothing is silent.
- Big files on a network share: read metadata from file heads, copy or cache a decoded working image locally once, then work from the cache.
- Same stack as strata-360: Python 3.13+, numpy/scipy/OpenCV/Pillow, PyTorch + ultralytics/insightface for detectors, FastAPI + React for the app, pytest + vitest, ruff, GitHub Actions.
