"""User settings: one JSON file, `~/.stratastills/settings.json` (the folder can be moved with the STRATASTILLS_HOME environment variable).

  projects_dir   where the projects live (each in its own folder, see projects.py). Default: `projects/` in this repository (gitignored) until the person picks a place.
  source_roots   list of folders (mounts) that hold the photos on THIS machine, e.g. `\\\\Server\\Drone` and `\\\\Server\\Raw` on Windows, `~/drone` and `~/raw` on Linux. A project stores a source as a path relative to a root (projects.py), so the same project opens on every machine.

The environment variables STRATASTILLS_PROJECTS and STRATASTILLS_SOURCES (roots separated by os.pathsep) override the file (handy for tests and one-off runs)."""
import json, os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PROJECTS_DIR = os.path.join(REPO, 'projects')
KNOWN = ('projects_dir', 'source_roots')


def home():
    return os.environ.get('STRATASTILLS_HOME') or os.path.join(os.path.expanduser('~'), '.stratastills')


def _path(): return os.path.join(home(), 'settings.json')


def load():
    try: d = json.load(open(_path()))
    except (OSError, ValueError): d = {}
    return d if isinstance(d, dict) else {}


def save(changes):
    """Merge `changes` (known keys only) into the settings file; returns the saved settings. A None value removes the setting (back to its default)."""
    d = load()
    for k, v in changes.items():
        if k not in KNOWN: raise KeyError(f'unknown setting {k!r}')
        if v is None: d.pop(k, None)
        elif k == 'source_roots': d[k] = [str(x) for x in (v.split(os.pathsep) if isinstance(v, str) else v) if str(x).strip()]
        else: d[k] = str(v)
    os.makedirs(home(), exist_ok=True); tmp = _path() + '.tmp'
    with open(tmp, 'w') as f: json.dump(d, f, indent=1)
    os.replace(tmp, _path()); return d


def projects_dir():
    """The folder that holds the projects (created when asked for)."""
    p = os.environ.get('STRATASTILLS_PROJECTS') or load().get('projects_dir') or DEFAULT_PROJECTS_DIR
    p = os.path.abspath(os.path.expanduser(p)); os.makedirs(p, exist_ok=True); return p


def source_roots():
    """The folders that hold the photos on this machine, in search order (`~` expanded, trailing separators dropped)."""
    r = os.environ.get('STRATASTILLS_SOURCES')
    r = r.split(os.pathsep) if r else load().get('source_roots') or []
    return [os.path.normpath(os.path.expanduser(x)) for x in r if isinstance(x, str) and x.strip()]


def report_dir():
    """Where scripts/build_report.py writes the visual report (`out/report` of the repository), which the server shows at /report/: the STRATASTILLS_REPORT environment variable or `report_dir` of the settings file move it."""
    p = os.environ.get('STRATASTILLS_REPORT') or load().get('report_dir') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'out', 'report')
    return os.path.abspath(os.path.expanduser(p))


def all_settings(): return dict(projects_dir=projects_dir(), source_roots=source_roots())
