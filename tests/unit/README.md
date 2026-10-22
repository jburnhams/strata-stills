# Unit tests

Shared commands, layout, hermetic setup and file-ownership rules are in [`tests/README.md`](../README.md); read that first. This file is owned by unit-test work.

Fast and isolated: synthetic arrays and tiny generated images; no real photos, no models, no network. `pytest tests/unit --cov` takes about a minute (CUDA tests included where there is a card) and is what CI gates on (`fail_under` in `pyproject.toml`).

## Where things are tested

| Module | Tests | Notes |
| --- | --- | --- |
| `develop.py` | `test_develop.py` | the NumPy reference: DNG opcodes, colour (matrices, hue/saturation map, ProPhoto), tone, clean-up, whole development on a synthetic DNG |
| `gpu.py`, `gpu_tiled.py`, `cpu_tiled.py` | `test_gpu.py` | every GPU function against its CPU original on each device; the tiled developers against the whole-picture ones; tile and slot sizing in `test_pipeline.py` |
| `detail.py`, `jpeg_quality.py` | `test_detail.py`, `test_jpeg_quality.py` | MTF measurement, Wiener sharpening, JPEG quality choice |
| `lineardng.py`, `inputs.py`, `match.py` | `test_inputs.py`, `test_match.py` | linear DNG parsing (black levels, polynomial opcodes, the valid area), device profiles, matching a panorama's levels |
| `pipeline.py`, `sizeplan.py`, `resources.py` | `test_pipeline.py`, `test_pipeline_raw.py` | stages, presets, keys, rules, folder sizes, worker and tile planning |
| `transforms.py`, `preview.py`, `projects.py`, `ingest.py`, `meta.py`, `server/app.py` | `test_transforms.py`, `test_preview.py`, `test_projects.py`, `test_ingest.py`, `test_meta.py`, `test_server*.py` | |
| `denoise_ai.py` | `test_denoise_ai.py` | with the model faked |

## Conventions on top of the shared ones

- A fake DNG for `develop.develop` is built by `synthetic_dng(monkeypatch, ...)` in `test_gpu.py` (a LibRaw-like uint16 picture plus the tags of a real Mini 4 Pro file); real files are never read.
- Equivalence tests give a mean and a 99.9th-percentile bound, not exact equality (float32 on different devices); keep the bounds as tight as the test passes with.
- Tests that need torch use `pytest.importorskip('torch')` (CI installs the CPU build; a developer without it runs the NumPy tests only).

## Known issues

- `test_pipeline.py::test_the_device_slot_limits...` and the sizing tests monkeypatch `gpu._slots` (reset to `None` afterwards); a new test touching the slot semaphores must do the same.
- An occasional CUDA error appears when many xdist workers share one card (`-n 4` with a GPU present); run GPU tests serially when chasing it.
