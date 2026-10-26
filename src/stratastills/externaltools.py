"""The other raw programs the comparison work needs (scripts/render_external.py, scripts/build_report.py): darktable (`darktable-cli`) and RawTherapee (`rawtherapee-cli`, with its profile folder for the Auto-Matched Curve profiles). They are not used by the pipeline itself, but every comparison against them is meaningless without them, so a script that needs them calls `require()` and stops with what to install rather than quietly showing less; `stratastills doctor` lists them."""
import glob, os, shutil, sys

NAMES = {'darktable': 'darktable-cli', 'rawtherapee': 'rawtherapee-cli'}
EXTRA = {                                                                               # where an installer puts them when they are not on the PATH
    'darktable': ['C:/Program Files/darktable/bin/darktable-cli.exe', os.path.expandvars('%LOCALAPPDATA%/Programs/darktable/bin/darktable-cli.exe'), '/Applications/darktable.app/Contents/MacOS/darktable-cli'],
    'rawtherapee': ['C:/Program Files/RawTherapee/*/rawtherapee-cli.exe', '/Applications/RawTherapee.app/Contents/MacOS/rawtherapee-cli'],
}
HINT = {'linux': 'sudo pacman -S darktable rawtherapee   (Arch; apt install darktable rawtherapee on Debian/Ubuntu)', 'darwin': 'brew install --cask darktable rawtherapee', 'win32': 'winget install darktable.darktable RawTherapee.RawTherapee'}


def find(program):
    """The path of the program's command line tool, or None."""
    exe = shutil.which(NAMES[program])
    if exe: return exe
    for pat in EXTRA[program]:
        hit = sorted(glob.glob(pat))
        if hit: return hit[-1]
    return None


def rt_profiles():
    """RawTherapee's folder of processing profiles (the Auto-Matched Curve ones are in it), or None."""
    exe = find('rawtherapee'); roots = [os.path.join(os.path.dirname(exe), 'profiles')] if exe else []
    if exe: roots += [os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(exe))), 'share', 'rawtherapee', 'profiles'), os.path.join(os.path.dirname(os.path.dirname(exe)), 'Resources', 'profiles')]
    roots += ['/usr/share/rawtherapee/profiles', '/usr/local/share/rawtherapee/profiles']
    return next((r for r in roots if os.path.exists(os.path.join(r, 'Auto-Matched Curve - ISO Low.pp3'))), None)


def missing():
    """What is not installed, as a list of names."""
    out = [p for p in NAMES if not find(p)]
    if find('rawtherapee') and not rt_profiles(): out.append("RawTherapee's profiles folder")
    return out


def require():
    """Stop (exit status 1, with what to install) unless both programs are there."""
    m = missing()
    if m: sys.exit(f"missing: {', '.join(m)}. The comparison against the other raw programs needs both installed: {HINT.get(sys.platform if sys.platform in HINT else 'linux')}")
