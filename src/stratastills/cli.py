"""stratastills command line: `doctor` checks the environment, `scan` lists the shots in a folder and what is known of each."""

import argparse
import json
import shutil
import sys

from strata_python import cli as strata_cli

from stratastills import __version__, ingest, meta, projects, settings

IMPORTS = (
    "numpy",
    "scipy",
    ("opencv", "cv2"),
    ("pillow", "PIL"),
    ("rawpy (DNG decoding)", "rawpy"),
    ("pyyaml", "yaml"),
    "loguru",
    strata_cli.Module("torch (the GPU path and the detector models; scripts/setup_torch.py)", "torch", optional=True),
)
HINT = "uv sync (scripts/setup_env)"


def check_ffmpeg():
    have = shutil.which("ffmpeg")
    return ("ok", have) if have else ("missing", "only needed for video; winget install Gyan.FFmpeg")


def check_torch():
    """Where the pixel work runs: the GPU (and whether it runs a kernel there), or the CPU. `slow` when cuDNN is off; `broken` (fails the run) when torch sees a card it cannot run on."""
    try:
        import torch
    except ImportError:
        return "ok", "on the CPU (no torch: the NumPy/OpenCV path, same result)"

    from stratastills import gpu

    if not gpu.available():
        return (
            "ok",
            f"on the CPU, torch {torch.__version__}: no CUDA or MPS device (or STRATASTILLS_GPU=0); slower, same result",
        )
    d = gpu.dev()
    name = torch.cuda.get_device_name(0) if d.type == "cuda" else "Apple silicon (MPS)"
    try:
        x = torch.ones(64, 64, device=d)
        float((x @ x).sum())
    except Exception as e:
        return (
            "broken",
            f"torch {torch.__version__} sees {name} but cannot run a kernel on it ({str(e).splitlines()[0][:90]}): install the build that includes this card (a Pascal GTX 10xx needs the cu126 wheels, see README), or set STRATASTILLS_GPU=0",
        )
    if d.type == "cuda" and not torch.backends.cudnn.enabled:
        return (
            "slow",
            f"on the GPU, torch {torch.__version__}, {name}, but cuDNN does not work on this card (turned off for the run): convolutions, the learned denoise above all, take 3 times as long; python scripts/setup_torch.py repairs a clashing cuDNN or picks the newest torch the card supports",
        )
    return "ok", f"on the GPU, torch {torch.__version__}: {name}"


def check_denoise():
    from stratastills import denoise_ai

    try:
        import spandrel  # noqa: F401

        have_net = True
    except ImportError:
        have_net = False
    if denoise_ai.available() and have_net:
        return "ok", "Real-ESRGAN weights, spandrel"
    return (
        "missing",
        f'{"weights" if have_net else "spandrel"}: high-ISO pictures get the classical clean-up only; scripts/setup_env installs it, or pip install spandrel==0.4.2 einops && python -c "from stratastills import denoise_ai; denoise_ai.fetch()"',
    )


def check_external():
    from stratastills import externaltools

    gone = externaltools.missing()
    if not gone:
        return "ok", "for the comparison report and scripts/render_external.py"
    return (
        "missing",
        f"{', '.join(gone)}: needed for the comparison report and scripts/render_external.py (they stop without them); {externaltools.HINT.get(sys.platform if sys.platform in externaltools.HINT else 'linux')}",
    )


# strata-python's doctor rows: the imports, then what an import cannot tell. Only a missing import or a GPU that cannot run a kernel fails the run; the rest is optional.
CHECKS = (
    strata_cli.Check("ffmpeg", check_ffmpeg, optional=True),
    strata_cli.Check("pixel work", check_torch),
    strata_cli.Check("learned denoise", check_denoise, optional=True),
    strata_cli.Check("darktable and RawTherapee", check_external, optional=True),
)


def scan(args):
    shots = ingest.scan(args.folder, recursive=args.recursive)
    out = []
    for s in shots:
        d = s.to_json()
        if args.info and s.jpeg:
            i = meta.photo_info(s.jpeg)
            d["info"] = dict(i, taken=i["taken"].isoformat() if i["taken"] else None, nadir=meta.is_nadir(i["dji"]))
        out.append(d)
    if args.json:
        print(json.dumps(out, indent=1))
        return 0
    for s, d in zip(shots, out):
        extra = ""
        if "info" in d:
            i = d["info"]
            g = i["dji"].get("GimbalPitchDegree")
            extra = f"  {i['width']}x{i['height']}  pitch {g if g is not None else '?'}{' nadir' if i['nadir'] else ''}  {i['taken'] or ''}"
        print(
            f"{s.stem:34} {s.kind:6} {'RAW ' if s.raw else '    '}{'JPG ' if s.jpeg else '    '}{'VID ' if s.video else '    '}{'+' + ','.join(t for t, _ in s.derived) if s.derived else ''}{extra}"
        )
    print(
        f"{len(shots)} shots ({sum(s.kind == 'still' for s in shots)} stills, {sum(s.kind == 'video' for s in shots)} videos)"
    )
    return 0


def cmd_projects(args):
    ps = projects.list_projects()
    print(f"projects folder: {settings.projects_dir()}")
    for p in ps:
        print(f"  {p['id']:30} {p['name']!r:30} {p['n_sources']} source folder(s)")
    if not ps:
        print("  (none yet: ./stratastills new NAME [--source FOLDER ...])")
    return 0


def cmd_new(args):
    try:
        d = projects.create(args.name, args.source or [], group=args.group)
    except projects.ProjectError as e:
        sys.exit(str(e))
    print(f"created {d['id']} in {projects.folder(d['id'])}")
    return 0


def cmd_source(args):
    try:
        d = (
            projects.remove_source(args.project, args.folder)
            if args.remove
            else projects.add_source(args.project, args.folder)
        )
    except projects.ProjectError as e:
        sys.exit(str(e))
    print(f"{d['id']}: {len(d['sources'])} source folder(s)")
    return 0


def serve(host, port):
    from stratastills.server import app  # FastAPI and the pipeline load only for `serve`

    app.main(["--host", host, "--port", str(port)])


def main(argv=None):
    p = argparse.ArgumentParser(prog="stratastills")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("projects", help="list the projects").set_defaults(fn=cmd_projects)
    n = sub.add_parser("new", help="create a project")
    n.add_argument("name")
    n.add_argument("--source", action="append", help="a source folder (repeatable)")
    n.add_argument("--group", help="a sub-folder of the projects folder to put it in")
    n.set_defaults(fn=cmd_new)
    n = sub.add_parser("source", help="add (or --remove) a source folder of a project")
    n.add_argument("project")
    n.add_argument("folder")
    n.add_argument("--remove", action="store_true")
    n.set_defaults(fn=cmd_source)
    strata_cli.add_set(
        sub, settings.SETTINGS
    )  # `set KEY [VALUE]`; "default" resets it; a list (source_roots) is joined by the path separator, ":" on Linux and ";" on Windows
    strata_cli.add_serve(
        sub, serve, default_port=8361
    )  # `serve [--host] [--port]`: the web app, opens on the list of projects
    strata_cli.add_doctor(sub, "stratastills", __version__, IMPORTS, checks=CHECKS, hint=HINT)
    s = sub.add_parser("scan", help="list the shots in a folder")
    s.add_argument("folder")
    s.add_argument("--recursive", action="store_true")
    s.add_argument("--info", action="store_true", help="read EXIF/XMP of each JPEG (slow on a network share)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=scan)
    a = p.parse_args(argv)
    return a.fn(a)
