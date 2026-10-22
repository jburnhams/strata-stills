"""stratastills command line: `doctor` checks the environment, `scan` lists the shots in a folder and what is known of each."""
import argparse, importlib, json, shutil, sys

from stratastills import __version__, ingest, meta, projects, settings

CHECKS = (('numpy', 'numpy'), ('scipy', 'scipy'), ('opencv', 'cv2'), ('pillow', 'PIL'), ('rawpy (DNG decoding)', 'rawpy'), ('pyyaml', 'yaml'), ('loguru', 'loguru'))


def doctor(args):
    ok = True; print(f'stratastills {__version__}, python {sys.version.split()[0]}')
    for label, mod in CHECKS:
        try: m = importlib.import_module(mod); print(f'  ok       {label} {getattr(m, "__version__", "")}'.rstrip())
        except ImportError: ok = False; print(f'  MISSING  {label}: pip install -r requirements.txt')
    have = bool(shutil.which('ffmpeg')); print(f'  {"ok      " if have else "missing "} ffmpeg' + ('' if have else ' (only needed for video; winget install Gyan.FFmpeg)'))
    try:
        from stratastills import gpu
        import torch
        if gpu.available():
            d = gpu.dev(); name = torch.cuda.get_device_name(0) if d.type == 'cuda' else 'Apple silicon (MPS)'; print(f'  ok       torch {torch.__version__}: the pixel work runs on the GPU: {name}')
        else: print(f'  ok       torch {torch.__version__}: no CUDA or MPS device (or STRATASTILLS_GPU=0): the pixel work runs on the CPU (slower, same result)')
    except ImportError: print('  missing  torch (optional: needed for the GPU path and the detector models; pip install torch, see README)')
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


def cmd_projects(args):
    ps = projects.list_projects(); print(f'projects folder: {settings.projects_dir()}')
    for p in ps: print(f"  {p['id']:30} {p['name']!r:30} {p['n_sources']} source folder(s)")
    if not ps: print('  (none yet: ./stratastills new NAME [--source FOLDER ...])')
    return 0


def cmd_new(args):
    try: d = projects.create(args.name, args.source or [], group=args.group)
    except projects.ProjectError as e: sys.exit(str(e))
    print(f"created {d['id']} in {projects.folder(d['id'])}"); return 0


def cmd_source(args):
    try: d = projects.remove_source(args.project, args.folder) if args.remove else projects.add_source(args.project, args.folder)
    except projects.ProjectError as e: sys.exit(str(e))
    print(f"{d['id']}: {len(d['sources'])} source folder(s)"); return 0


def cmd_set(args):
    if args.value is None: print(f'{args.key} = {settings.all_settings().get(args.key)}'); return 0
    try: settings.save({args.key: None if args.value == 'default' else args.value})
    except KeyError as e: sys.exit(str(e))
    print(f'{args.key} = {settings.all_settings()[args.key]}'); return 0


def cmd_serve(args):
    from stratastills.server import app
    app.main(['--host', args.host, '--port', str(args.port)]); return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog='stratastills'); sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('projects', help='list the projects').set_defaults(fn=cmd_projects)
    n = sub.add_parser('new', help='create a project'); n.add_argument('name'); n.add_argument('--source', action='append', help='a source folder (repeatable)'); n.add_argument('--group', help='a sub-folder of the projects folder to put it in'); n.set_defaults(fn=cmd_new)
    n = sub.add_parser('source', help='add (or --remove) a source folder of a project'); n.add_argument('project'); n.add_argument('folder'); n.add_argument('--remove', action='store_true'); n.set_defaults(fn=cmd_source)
    n = sub.add_parser('set', help='show or change a setting (projects_dir); value "default" resets it'); n.add_argument('key'); n.add_argument('value', nargs='?'); n.set_defaults(fn=cmd_set)
    n = sub.add_parser('serve', help='the web app: opens on the list of projects'); n.add_argument('--host', default='127.0.0.1'); n.add_argument('--port', type=int, default=8361); n.set_defaults(fn=cmd_serve)
    sub.add_parser('doctor', help='check the environment').set_defaults(fn=doctor)
    s = sub.add_parser('scan', help='list the shots in a folder'); s.add_argument('folder'); s.add_argument('--recursive', action='store_true'); s.add_argument('--info', action='store_true', help='read EXIF/XMP of each JPEG (slow on a network share)'); s.add_argument('--json', action='store_true'); s.set_defaults(fn=scan)
    a = p.parse_args(argv); return a.fn(a)
