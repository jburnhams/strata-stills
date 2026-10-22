"""stratastills command line: `doctor` checks the environment, `scan` lists the shots in a folder and what is known of each."""
import argparse, importlib, json, shutil, sys

from stratastills import __version__, ingest, meta

CHECKS = (('numpy', 'numpy'), ('scipy', 'scipy'), ('opencv', 'cv2'), ('pillow', 'PIL'), ('rawpy (DNG decoding)', 'rawpy'), ('pyyaml', 'yaml'), ('loguru', 'loguru'))


def doctor(args):
    ok = True; print(f'stratastills {__version__}, python {sys.version.split()[0]}')
    for label, mod in CHECKS:
        try: m = importlib.import_module(mod); print(f'  ok       {label} {getattr(m, "__version__", "")}'.rstrip())
        except ImportError: ok = False; print(f'  MISSING  {label}: pip install -r requirements.txt')
    have = bool(shutil.which('ffmpeg')); print(f'  {"ok      " if have else "missing "} ffmpeg' + ('' if have else ' (only needed for video; winget install Gyan.FFmpeg)'))
    try:
        import torch; print(f'  ok       torch {torch.__version__}, cuda {torch.cuda.is_available()}')
    except ImportError: print('  missing  torch (only needed for the detector stages: pip install -r requirements-vision.txt)')
    return 0 if ok else 1


def scan(args):
    shots = ingest.scan(args.folder, recursive=args.recursive); out = []
    for s in shots:
        d = s.to_json()
        if args.info and s.jpeg:
            i = meta.photo_info(s.jpeg); d['info'] = dict(i, taken=i['taken'].isoformat() if i['taken'] else None, nadir=meta.is_nadir(i['dji']))
        out.append(d)
    if args.json: print(json.dumps(out, indent=1)); return 0
    for s, d in zip(shots, out):
        extra = ''
        if 'info' in d:
            i = d['info']; g = i['dji'].get('GimbalPitchDegree'); extra = f"  {i['width']}x{i['height']}  pitch {g if g is not None else '?'}{' nadir' if i['nadir'] else ''}  {i['taken'] or ''}"
        print(f"{s.stem:34} {s.kind:6} {'RAW ' if s.raw else '    '}{'JPG ' if s.jpeg else '    '}{'VID ' if s.video else '    '}{'+' + ','.join(t for t, _ in s.derived) if s.derived else ''}{extra}")
    print(f'{len(shots)} shots ({sum(s.kind == "still" for s in shots)} stills, {sum(s.kind == "video" for s in shots)} videos)'); return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog='stratastills'); sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('doctor', help='check the environment').set_defaults(fn=doctor)
    s = sub.add_parser('scan', help='list the shots in a folder'); s.add_argument('folder'); s.add_argument('--recursive', action='store_true'); s.add_argument('--info', action='store_true', help='read EXIF/XMP of each JPEG (slow on a network share)'); s.add_argument('--json', action='store_true'); s.set_defaults(fn=scan)
    a = p.parse_args(argv); return a.fn(a)
