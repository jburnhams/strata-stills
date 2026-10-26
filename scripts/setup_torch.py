"""Install the newest PyTorch the machine's hardware can run, and check that it does, so that a new machine (or a card change) needs no hand-picked wheel.
  python scripts/setup_torch.py [--python .venv/bin/python] [--cuda auto|cu128|cu126|cpu] [--dry-run] [--check]
What it does:
  1. looks for NVIDIA cards (nvidia-smi: name and compute capability) and picks the plan from the oldest one:
       macOS                                the plain PyPI wheel (it has the Metal backend)
       no NVIDIA card                       the CPU wheel (Linux, Windows)
       compute capability under 7.5         Maxwell, Pascal (GTX 9xx and 10xx), Volta: torch < 2.10 with CUDA 12.6 (torch 2.9 is the last with kernels for them: the newer wheels start without sm_50 to sm_70 and every kernel fails with "no kernel image")
       7.5 and up                           Turing, Ampere, Ada, Hopper, Blackwell (RTX 50): the newest torch with CUDA 12.8 (cu128 is the first with sm_120)
  2. installs it (pip, from download.pytorch.org/whl/<plan>), torch and torchvision in one go so that they match;
  3. repairs a clashing cuDNN: pip puts the cuDNN of CUDA 12 and of CUDA 13 (nvidia-cudnn-cu12, nvidia-cudnn-cu13) in the SAME folder, so having both leaves one with the other's files. A GTX 1050 with the CUDA 13 cuDNN (9.24, which has no kernels for it) failed every convolution ('unable to find an engine'), which made the learned denoise 3 times slower and the whole GPU path run out of memory; the one that does not belong to the torch is removed and the right one put back;
  4. probes the result in a fresh process (a kernel and a cuDNN convolution in float32 and half on the card) and, if that fails, goes down the ladder to the next older plan (cu128 -> cu126 <2.10).
Run it again after changing the card, or when `stratastills doctor` says cuDNN is off. Everything but the probe is skipped with --check; --dry-run prints the commands. Standard library only."""
import argparse, json, platform, re, subprocess, sys

INDEX = 'https://download.pytorch.org/whl/'
PLANS = {                                              # name: (pip index suffix or None for PyPI, torch requirement, the CUDA major of its cuDNN or None)
    'cu128': ('cu128', 'torch', 12),
    'cu126': ('cu126', 'torch<2.10', 12),
    'cpu': ('cpu', 'torch', None),
    'mac': (None, 'torch', None),
}
LADDER = ['cu128', 'cu126']


def gpus():
    """[(name, (major, minor))] of the NVIDIA cards nvidia-smi sees (empty when there is no nvidia-smi or no card)."""
    try: out = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError): return []
    cards = []
    for line in out.splitlines():
        name, _, cc = line.rpartition(',')
        m = re.match(r'\s*(\d+)\.(\d+)', cc)
        if name and m: cards.append((name.strip(), (int(m.group(1)), int(m.group(2)))))
    return cards


def choose(cards, system):
    """(plan name, why) from the cards and the OS (platform.system())."""
    if system == 'Darwin': return 'mac', 'macOS: the PyPI wheel has the Metal (MPS) backend'
    if not cards: return 'cpu', 'no NVIDIA card found (nvidia-smi): the CPU wheel'
    low = min(cc for _, cc in cards); names = ', '.join(n for n, _ in cards)
    if low < (7, 5): return 'cu126', f'{names}: compute capability {low[0]}.{low[1]} is under 7.5, so torch 2.9 with CUDA 12.6 (the newest with kernels for it)'
    return 'cu128', f'{names}: compute capability {low[0]}.{low[1]}: the newest torch, CUDA 12.8 (the first build for RTX 50 / sm_120)'


def pip(py, *args, dry=False):
    cmd = [py, '-m', 'pip', *args]; print('$', ' '.join(cmd), flush=True)
    if dry: return 0
    return subprocess.run(cmd).returncode


def installed(py, prefix):
    """{name: version} of the pip packages whose name starts with `prefix`."""
    out = subprocess.run([py, '-m', 'pip', 'list', '--format=json'], capture_output=True, text=True).stdout
    try: return {p['name']: p['version'] for p in json.loads(out) if p['name'].lower().startswith(prefix)}
    except ValueError: return {}


def repair_cudnn(py, plan, dry=False):
    """Leave only the cuDNN package of the torch's CUDA major version, with its own files back in place (see the module doc). True when something was changed."""
    major = PLANS[plan][2]
    if major is None: return False
    have = installed(py, 'nvidia-cudnn'); wanted = f'nvidia-cudnn-cu{major}'; wrong = [n for n in have if n.lower() != wanted and re.match(r'nvidia-cudnn-cu\d+$', n.lower())]
    if not wrong or wanted not in {n.lower() for n in have}: return False
    print(f'two cuDNN packages are installed ({", ".join(sorted(have))}): removing {", ".join(wrong)} and putting {wanted} back')
    version = next(v for n, v in have.items() if n.lower() == wanted)
    pip(py, 'uninstall', '-y', *wrong, dry=dry); pip(py, 'install', '--no-deps', '--force-reinstall', f'{wanted}=={version}', dry=dry); return True


PROBE = r"""
import json, sys
r = dict(ok=False)
try:
    import torch
    r.update(torch=torch.__version__, cuda=torch.version.cuda, available=torch.cuda.is_available())
    if r['available']:
        r['card'] = torch.cuda.get_device_name(0); torch.zeros(4, device='cuda').sum().item(); r['kernel'] = True
        r['cudnn'] = torch.backends.cudnn.version(); torch.backends.cudnn.enabled = True; ok = True
        for dt in (torch.float32, torch.float16):
            try: torch.nn.functional.conv2d(torch.zeros(1, 1, 16, 16, device='cuda', dtype=dt), torch.ones(1, 1, 1, 9, device='cuda', dtype=dt)); torch.cuda.synchronize()
            except Exception as e: ok = False; r['cudnn_error'] = str(e).splitlines()[0][:120]
        r['cudnn_ok'] = ok; r['ok'] = ok
    else: r['ok'] = True                                  # no card: nothing more to check (CPU, or Metal, which the pipeline probes itself)
except Exception as e: r['error'] = str(e).splitlines()[0][:160] if str(e) else type(e).__name__
print(json.dumps(r))
"""


def probe(py):
    """The state of torch in `py`, from a fresh process: {ok, torch, cuda, available, card, kernel, cudnn, cudnn_ok, ...}."""
    out = subprocess.run([py, '-c', PROBE], capture_output=True, text=True).stdout.strip().splitlines()
    try: return json.loads(out[-1])
    except (ValueError, IndexError): return dict(ok=False, error='the probe printed nothing (torch not installed?)')


def install(py, plan, dry=False):
    index, req, _ = PLANS[plan]
    return pip(py, 'install', '--upgrade', req, 'torchvision', *(['--index-url', INDEX + index] if index else []), dry=dry)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); ap.add_argument('--python', default=sys.executable); ap.add_argument('--cuda', default='auto', choices=('auto', 'cu128', 'cu126', 'cpu')); ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--check', action='store_true'); a = ap.parse_args(argv)
    cards = gpus(); plan, why = choose(cards, platform.system()) if a.cuda == 'auto' else (a.cuda, f'asked for {a.cuda}'); print(f'plan: {plan}: {why}')
    if not a.check:
        start = LADDER.index(plan) if plan in LADDER else 0; tries = LADDER[start:] if plan in LADDER else [plan]
        for p in tries:
            if install(a.python, p, a.dry_run): print(f'pip failed for {p}'); continue
            repair_cudnn(a.python, p, a.dry_run)
            if a.dry_run: return 0
            r = probe(a.python); print('probe:', json.dumps(r))
            if r.get('ok'): return 0
            print(f'torch from the {p} plan does not work on this machine' + (': trying the next older plan' if p != tries[-1] else ''))
        return 1
    r = probe(a.python); print('probe:', json.dumps(r)); return 0 if r.get('ok') else 1


if __name__ == '__main__':
    sys.exit(main())
