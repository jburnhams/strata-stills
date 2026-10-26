"""Render the candidate raws with the other raw programs' command lines (darktable, RawTherapee), for scripts/build_report.py and scripts/parity_external.py. Cross-platform; stops with what to install when they are missing (stratastills/externaltools.py).
  python scripts/render_external.py darktable|rawtherapee [--set a7c] [--pp3 auto-matched|<file.pp3>] [--stems A,B]
Reads candidates/<set>/ (the raws of docs/<set>_candidates.json), writes candidates/renders/[<set>-]<program>[-<profile>]/<stem>.jpg, skipping what exists; the originals are never touched.
  darktable      darktable-cli with the configuration in candidates/dtconf (default modules, scene-referred workflow), asked for 3000 px on the long side
  rawtherapee    rawtherapee-cli full size; --pp3 auto-matched uses its own 'Auto-Matched Curve' profile for the picture's ISO (Low up to 400, Medium up to 1600, else High), no --pp3 its default profile
(render_external.ps1 is the older Windows-only version of the same.)"""
import argparse, json, os, subprocess, sys, time
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); sys.path.insert(0, os.path.join(ROOT, 'src'))
from stratastills import externaltools as ET


def out_dir(set_name, program, pp3=''):
    return os.path.join(ROOT, 'candidates', 'renders', (set_name + '-' if set_name != 'a7c' else '') + program + ('-' + os.path.splitext(os.path.basename(pp3))[0] if pp3 else ''))


def profile_for(iso):
    return os.path.join(ET.rt_profiles(), f"Auto-Matched Curve - ISO {'Low' if iso <= 400 else 'Medium' if iso <= 1600 else 'High'}.pp3")


def render(program, raw, dest, pp3='', iso=0):
    """One picture: False when the program left no file."""
    if program == 'darktable':
        cmd = [ET.find('darktable'), raw, dest, '--width', '3000', '--height', '3000', '--core', '--configdir', os.path.join(ROOT, 'candidates', 'dtconf'), '--library', ':memory:']
    else:
        prof = profile_for(iso) if pp3 == 'auto-matched' else pp3
        cmd = [ET.find('rawtherapee'), '-o', dest, '-q', '-Y'] + (['-p', prof] if prof else ['-d']) + ['-c', raw]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return os.path.exists(dest)


def render_missing(program, set_name, raws, pp3='', say=print):
    """Make the render of every raw ({stem: (path, iso)}) that has none yet; stops (exit 1) when one fails. The seconds each program took (the whole command line run: its start-up included, one picture at a time) are kept in timings.json beside the renders; a render made earlier (another machine, another day) with no time is made once more into a throw-away file just to time it."""
    ET.require(); out = out_dir(set_name, program, 'auto-matched' if program == 'rawtherapee' and pp3 == 'auto-matched' else (pp3 if program == 'rawtherapee' else '')); os.makedirs(out, exist_ok=True); tj = os.path.join(out, 'timings.json')
    times = json.load(open(tj)) if os.path.exists(tj) else {}; todo = [s for s in raws if not os.path.exists(os.path.join(out, s + '.jpg'))]
    if todo: say(f'{program} ({set_name}): rendering {len(todo)} of {len(raws)} pictures into {out}')
    for s in raws:
        path, iso = raws[s]; dest = os.path.join(out, s + '.jpg'); fresh = s in todo
        if not fresh and s in times: continue
        t0 = time.perf_counter(); ok = render(program, path, dest if fresh else dest + '.timing.jpg', pp3, iso); times[s] = round(time.perf_counter() - t0, 1)
        if not fresh and os.path.exists(dest + '.timing.jpg'): os.remove(dest + '.timing.jpg')
        if not ok: sys.exit(f'{program} made no picture of {path}')
        say(f'  {s}: {times[s]} s'); json.dump(times, open(tj, 'w'))
    return out


def timings(set_name, program, pp3=''):
    """{stem: seconds} kept by render_missing for a program's renders ({} when there are none)."""
    tj = os.path.join(out_dir(set_name, program, pp3), 'timings.json'); return json.load(open(tj)) if os.path.exists(tj) else {}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('program', choices=('darktable', 'rawtherapee')); ap.add_argument('--set', default='a7c'); ap.add_argument('--pp3', default=''); ap.add_argument('--stems'); a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import build_report as BR
    folder = os.path.join(ROOT, 'candidates', a.set); stems = a.stems.split(',') if a.stems else [c['stem'] for c in json.load(open(os.path.join(ROOT, 'docs', f'{a.set}_candidates.json')))]
    files = {os.path.splitext(f)[0]: os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in BR.RAW_EXTS}
    render_missing(a.program, a.set, {s: (files[s], (BR.shot(files[s])[4] or 0)) for s in stems if s in files}, a.pp3)
