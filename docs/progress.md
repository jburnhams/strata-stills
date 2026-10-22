# Progress log

## 2026-10-22: scaffold

- Mirrored strata-360's layout, tooling (pytest/ruff/coverage floor, GitHub Actions unit workflow, `stratastills.cmd` launcher) and dependency choices. Only Python 3.14 is installed on this machine; the light dependencies (numpy 2.5, scipy, OpenCV 5.0, Pillow, rawpy 0.27) install and pass on it. strata-360 pins 3.13 because of torch and the model packages: install 3.13 (`winget install Python.Python.3.13`) before adding the detector stages.
- Looked at the Hazelmere shoot (see overview section 2): Mini 4 Pro, 48 MP, DNG+JPG pairs, first photos inspected are straight-down shots of rooftops, so "levelling" there means aligning to the dominant building/road direction, not a horizon.
- Reading a 100 MB DNG over the network share takes tens of seconds; the XMP-in-the-JPEG-head trick makes `scan --info` cheap. Plan to cache decoded working images locally.
