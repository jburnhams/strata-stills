# Implementation plan

Each milestone ends with something that can be run on the Hazelmere shoot.

- [x] **M0 scaffold**: repo layout, environment, CI, `doctor`, `scan` (grouping, metadata). 
- [ ] **M1 project and working images**: a project folder (`shoots/<name>/`) with `project.json`; decode each source once (DNG via rawpy, else JPEG) into a cached working image (long side 2048) and a thumbnail; resumable and versioned like strata-360's analysis stages.
- [ ] **M2 level**: nadir shots: dominant orientation of edges (roof lines, roads) folded to 90 degrees, rotate by the smallest angle with a confidence; oblique shots: horizon from gradient/line voting, plus the DJI flight roll as a prior. Contact sheet of before/after to judge on real shots.
- [ ] **M3 develop**: raw development to a good starting point (exposure/white balance/tone/highlight recovery), compared against the camera JPEG; port exposure and palette grading from strata-360.
- [ ] **M4 detect and label**: YOLO11 objects and people, scene tags, place from GPS; stored per shot.
- [ ] **M5 frame**: crops per target aspect ratio from detections and composition rules; the person can override.
- [ ] **M6 render**: finished JPEG/TIFF plus sidecar of decisions; batch run over a folder.
- [ ] **M7 web app**: FastAPI server and React UI (review proposals, override, export), reusing strata-360's web scaffolding.
- Later: other cameras and phone photos, HDR/bracket merge, panorama stitching, video stills.
