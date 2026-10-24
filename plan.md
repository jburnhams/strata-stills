# strata-stills: plan

Where the project is going, in three horizons. The detailed milestone list is [docs/implementation-plan.md](docs/implementation-plan.md), the design brief [docs/overview.md](docs/overview.md), the log of what was measured [docs/progress.md](docs/progress.md). Guiding rule: **full auto first**. Defaults pick the best treatment for each picture; every other setting exists for debugging, testing and tuning, and a new behaviour ships with the rule that selects it.

## Short term (days)

- **Land the sharpening PR** (#7): amount 3, wider roll-off, overshoot 0, comparison tooling. Then re-run `sharpen_sweep.py` + `compare_detail.py --report` after any change to develop, denoise or sharpen, at q95, and keep the tables in `docs/sharpening_comparison.md` current.
- **Close what is still behind RawTherapee/the camera JPEG** (see the tables): ringing beside edges (`ring`, +3 to 6 %), `halo` in the A7C ISO 251-800 band, and the DJI sharpness against the camera's own JPEG (52 against 66). Ideas: ring-aware clamp (limit overshoot only on strong edges), per-device or per-ISO sharpening from the measured MTF instead of one amount, a halo/ringing term inside the Wiener gain.
- **Make the GPU route safe on small cards.** On the 2 GB GTX 1050 (shared with the desktop) it runs out of memory or gives wrong pictures; the tile planner must use the memory that is actually free, an OOM must fall back to tiles then the CPU, and a parity check against the CPU reference should run on the real card (not only synthetic data). Until then use `STRATASTILLS_GPU=0` there.
- **Lens gaps:** no lensfun profile for the Samyang AF 50mm F1.4 II (shading from the camera table only); the FE 28-60's camera-table distortion convention is unexplained (lensfun fixes it); the image-circle crop is in, check the crop on all E 50mm candidates by eye.
- **Check the learned denoise on DJI** (the DJI frames never reach the noise threshold, so it has not run there) and on a full-size export, not only the cached 3000 px pictures.
- **Small hygiene:** documentation for `source_roots` in the README (Windows: UNC shares; Linux: mounts); the macOS CI job aborted once in `test_lensfundb` (flake, watch it); a random-order flake in `tests/unit/test_server.py`; `launcher` and `setup_env.sh` exec bits are in git now.

## Medium term (weeks)

- **M1 working images** and **M2 level**: decode once into cached working images; levelling that is robust (a handheld frame was turned by 5 degrees on a confidence of 0.37, barely above the 0.35 threshold: needs a stricter rule, a lens-aware check and a before/after contact sheet).
- **M3b develop quality** as automatic regression checks: a small synthetic suite in CI for the metrics (`sharp`, `halo`, `ring`, `noise`, `detail`), and a documented procedure to refit a device (`NoiseScale` by ISO, look table, shading) from its own camera JPEGs, so a new camera is a script run, not research.
- **M4 detect and label, M5 frame, M6 render**: YOLO11 objects and people, scene tags, place from GPS; crops per target aspect ratio that respect the subject, horizon and detections; finished JPEG/TIFF plus a sidecar of every decision; batch over a folder with `auto` rules.
- **More devices:** a second and third camera (phone, another drone) through the same per-device profile mechanism; non-DNG raws (ARW done, CR3/NEF/RAF next) with the camera's own correction tables where present.
- **Web app (M7):** review proposals, override any decision, compare variants with the loupe at 100 %, edit `source_roots` and the projects folder in the UI, show why a decision was made (level, crop, lens source, denoise route).
- **Quality of the comparison itself:** more candidate sets (night, backlit, foliage, architecture), a standard encode quality (q95) in every comparison script, and a "no worse than RawTherapee/darktable/camera on every metric" gate that runs locally before a tuning PR.

## Long term (months)

- **Whole-shoot output:** pick the best source and the best of near-duplicates, rank and reject, one finished set per shoot with captions and places; works from a folder on the NAS to a delivered folder with no manual step.
- **HDR/bracket merge, panorama stitching, video stills** (the drone's videos are noted and left alone today).
- **Learned parts where measured to be better:** denoise and tone already use the camera JPEG as the reference; consider a small learned model for sharpening/ringing control and for matching a device's look, trained on the user's own raw/JPEG pairs, only if it beats the classical code on the comparison tables.
- **Hardware independence:** the same result on the RTX 3060 (Windows), Apple silicon and a small CPU/GPU box; a documented support matrix (torch build per GPU generation, e.g. Pascal needs cu126) and `doctor` that checks it.
- **Sharing:** one project opening on every machine (done for sources through `source_roots`; next: cache and generated variants in a place both machines can use) and a clean split between code (repo), models (fetched), data (NAS).
- **Packaging:** a single installer or container so the app is not a research environment to set up.

## Ideas parked

- Use the camera's own APS-C mode data to choose the crop; compare with the lens-circle rule.
- Per-region sharpening from the local blur (depth of field, motion) rather than one MTF per picture.
- Gimbal/flight data as priors for levelling and cropping on drone shots.
- A bench page in the app that shows the metric tables for the project's own pictures against the camera JPEGs.
