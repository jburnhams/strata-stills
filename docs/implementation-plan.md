# Implementation plan

Each milestone ends with something that can be run on the Hazelmere shoot.

- [x] **M0 scaffold**: repo layout, environment, CI, `doctor`, `scan` (grouping, metadata). 
- [x] **M0b projects**: projects directory setting, projects with several source folders, `projects/new/source/set/serve` commands, project list on app start.
- [x] **M0c browser app (first cut)**: React app (web/): projects, source folders, contents grid, item page with variants (pick the primary) and transforms (original, levelled; first levelling estimate for nadir and oblique shots), panoramas as their own items.
- [ ] **M1 working images**: in each project, decode each source once (DNG via rawpy, else JPEG) into a cached working image (long side 2048) and a thumbnail; resumable and versioned like strata-360's analysis stages.
- [ ] **M2 level**: nadir shots: dominant orientation of edges (roof lines, roads) folded to 90 degrees, rotate by the smallest angle with a confidence; oblique shots: horizon from gradient/line voting, plus the DJI flight roll as a prior. Contact sheet of before/after to judge on real shots.
- [x] **M3 develop and export (first cut)**: raw development following the DNG model (opcodes, colour matrices, tone curve, noise profile), automatic exposure, local shadow/highlight recovery, detail measurement and recommended export size, MTF-matched sharpening, JPEG/TIFF export with a decision record. Goal: beat the camera JPEG in every way.
- [ ] **M3b develop quality**: shadow grain (wavelet coring by the noise profile), edge defringe, warmer shadows, a regression check that measures texture energy and noise against the camera JPEG on a set of shoot crops; allow tuning (recover, exposure) per item in the app.
- [ ] **M4 detect and label**: YOLO11 objects and people, scene tags, place from GPS; stored per shot.
- [ ] **M5 frame**: crops per target aspect ratio from detections and composition rules; the person can override.
- [ ] **M6 render**: finished JPEG/TIFF plus sidecar of decisions; batch run over a folder.
- [ ] **M7 web app**: FastAPI server and React UI (review proposals, override, export), reusing strata-360's web scaffolding.
- Later: other cameras and phone photos, HDR/bracket merge, panorama stitching, video stills.
