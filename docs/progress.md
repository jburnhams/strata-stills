# Progress log

## 2026-10-22: scaffold

- Mirrored strata-360's layout, tooling (pytest/ruff/coverage floor, GitHub Actions unit workflow, `stratastills.cmd` launcher) and dependency choices. Only Python 3.14 is installed on this machine; the light dependencies (numpy 2.5, scipy, OpenCV 5.0, Pillow, rawpy 0.27) install and pass on it. strata-360 pins 3.13 because of torch and the model packages: install 3.13 (`winget install Python.Python.3.13`) before adding the detector stages.
- Looked at the Hazelmere shoot (see overview section 2): Mini 4 Pro, 48 MP, DNG+JPG pairs, first photos inspected are straight-down shots of rooftops, so "levelling" there means aligning to the dominant building/road direction, not a horizon.
- Network reads are fine: a 98 MB DNG reads in about 1 s (104 MB/s) from `\\Server\Drone`, and rawpy opens it in 10 ms and develops a half-size image in 1.7 s. (An earlier note here that it took tens of seconds was wrong: that was a cold first access.) Reading the XMP from the JPEG head still keeps `scan --info` quick.

- The Lightroom `-Pano.dng` files are a different kind of DNG from the camera's (checked with tifffile on `DJI_20250402190211_0046_D`). Camera DNG: 8064x6048, one channel, uncompressed 16-bit Bayer mosaic (PhotometricInterpretation CFA), DNG 1.4, plus a 960x720 preview. Pano DNG: 14925x7807 (116 MP), already demosaiced and merged, 3-channel 16-bit LinearRaw, compressed with JPEG XL (TIFF compression 52546), with an 8-bit transparency mask for the ragged edges, a pyramid of reduced sizes, DNG 1.7 written by Lightroom Classic 15.1.1. rawpy/LibRaw rejects it ("Unsupported file format"); reading it needs tifffile with imagecodecs (JPEG XL). It is a derived file (never a source by default), and since it is already demosaiced there is no white-balance or demosaic choice left: only tone and colour.

## 2026-10-22: projects

- A project is separate from the photos: a folder in the projects directory (setting `projects_dir`, default `projects/` in this repo, gitignored; `~/.stratastills/settings.json`, env `STRATASTILLS_PROJECTS`) with `project.json` listing any number of source folders. Projects may be nested in plain sub-folders. `./stratastills serve` opens on the list of projects (interim single-file page until the React app exists).
