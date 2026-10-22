# Progress log

## 2026-10-22: scaffold

- Mirrored strata-360's layout, tooling (pytest/ruff/coverage floor, GitHub Actions unit workflow, `stratastills.cmd` launcher) and dependency choices. Only Python 3.14 is installed on this machine; the light dependencies (numpy 2.5, scipy, OpenCV 5.0, Pillow, rawpy 0.27) install and pass on it. strata-360 pins 3.13 because of torch and the model packages: install 3.13 (`winget install Python.Python.3.13`) before adding the detector stages.
- Looked at the Hazelmere shoot (see overview section 2): Mini 4 Pro, 48 MP, DNG+JPG pairs, first photos inspected are straight-down shots of rooftops, so "levelling" there means aligning to the dominant building/road direction, not a horizon.
- Network reads are fine: a 98 MB DNG reads in about 1 s (104 MB/s) from `\\Server\Drone`, and rawpy opens it in 10 ms and develops a half-size image in 1.7 s. (An earlier note here that it took tens of seconds was wrong: that was a cold first access.) Reading the XMP from the JPEG head still keeps `scan --info` quick.

- The Lightroom `-Pano.dng` files are a different kind of DNG from the camera's (checked with tifffile on `DJI_20250402190211_0046_D`). Camera DNG: 8064x6048, one channel, uncompressed 16-bit Bayer mosaic (PhotometricInterpretation CFA), DNG 1.4, plus a 960x720 preview. Pano DNG: 14925x7807 (116 MP), already demosaiced and merged, 3-channel 16-bit LinearRaw, compressed with JPEG XL (TIFF compression 52546), with an 8-bit transparency mask for the ragged edges, a pyramid of reduced sizes, DNG 1.7 written by Lightroom Classic 15.1.1. rawpy/LibRaw rejects it ("Unsupported file format"); reading it needs tifffile with imagecodecs (JPEG XL). It is a derived file (never a source by default), and since it is already demosaiced there is no white-balance or demosaic choice left: only tone and colour.

## 2026-10-22: projects

- A project is separate from the photos: a folder in the projects directory (setting `projects_dir`, default `projects/` in this repo, gitignored; `~/.stratastills/settings.json`, env `STRATASTILLS_PROJECTS`) with `project.json` listing any number of source folders. Projects may be nested in plain sub-folders. `./stratastills serve` opens on the list of projects (interim single-file page until the React app exists).

## 2026-10-22: browser app, first cut

- Items and variants: a shot's files (raw, camera JPEG, `-Enhanced-NR` ...) are its variants and the person picks the primary (stored in `<project>/items.json`; default raw, a panorama's JPEG). `-Pano` files are their own items, named after the shot they came from.
- Previews: decoded once to a 2048 px working image (raw: LibRaw half-size with camera white balance; Lightroom linear DNG: the embedded 2048 level through tifffile + imagecodecs for JPEG XL) and cached under `<project>/cache/`, keyed by path, size and mtime. A transform result and its info are cached beside it.
- The raw as LibRaw develops it by default is darker and cooler than the camera's own JPEG; choosing a good development (baseline exposure, DJI tone curve, white balance) is milestone M3.
- Levelling v1 (transforms.py): long straight segments (Canny + probabilistic Hough), deviation from the nearest axis; straight-down shots use the 4x-angle circular mean so any building direction folds to the nearest 90 degrees; oblique shots only trust the median within 10 degrees. Not applied when under 12 lines, confidence under 0.35 or the turn under 0.15 degrees. Synthetic grids turned 3, 7 and 20 degrees are recovered within 0.5 degrees. Real-shot confidence is modest (0.49 on the first rooftop shot), so judging it on the whole shoot is the next step. A panorama's pitch is not known from its own JPEG, so it is treated as oblique for now.
- OpenCV 5 returns `HoughLinesP` segments as (N,4), OpenCV 4 as (N,1,4).
