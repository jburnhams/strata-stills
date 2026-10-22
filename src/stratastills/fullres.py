"""Full-resolution pictures for the loupe, which is only any use at full resolution: judging sharpness, grain and compression artefacts needs the real pixels.

Each variant gets one cached full-size copy in the project (`<project>/cache/<key>.npy`, 8 bit BGR, uncompressed), opened with np.load(mmap_mode='r'): a crop reads only the pages it needs, so it is instant however large the picture is.
Making the copy is the slow part and happens once, in the background: a JPEG, TIFF or generated picture is decoded (a second or two); a camera raw is DEVELOPED at full size with the default settings (develop.develop, about a minute and a half for 48 MP: the same recipe as the develop stage of the default pipeline, so what the loupe
shows of a raw is what Generate starts from). `status` says where that stands, `prepare` starts it if needed, and the loupe's crop endpoint answers 'preparing' until it is ready.

`dims(path)` is the full size of a file without reading it (an image header, or the largest level of a DNG): the loupe uses it to show the same scene area from variants of different sizes."""
import os, threading

import numpy as np

from stratastills import develop, images, projects as P

_jobs = {}                                  # (cache file) -> {state: running|failed, text}
_lock = threading.Lock()


def cache_file(pid, path):
    return images.cache_path(P.folder(pid), images.cache_key(path, 'fullres', develop.VERSION, develop.HIGHLIGHT_CONTRAST), '.npy')


def get(pid, path):
    """The full-size picture as a read-only memory map (H, W, 3) uint8 BGR, or None if it has not been made yet."""
    f = cache_file(pid, path)
    if not os.path.exists(f): return None
    try: return np.load(f, mmap_mode='r')
    except (OSError, ValueError): return None


def status(pid, path):
    """{state: 'ready' | 'running' | 'failed' | 'idle', text}."""
    if os.path.exists(cache_file(pid, path)): return dict(state='ready', text='')
    with _lock: j = _jobs.get(cache_file(pid, path))
    return dict(j) if j else dict(state='idle', text='')


def _build(pid, path, f):
    try:
        if images.is_camera_raw(path):
            try: rgb = develop.develop(path, half=False); arr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1])
            except Exception: arr = images.load(path, 100000)                                    # a Lightroom linear DNG, or a raw LibRaw cannot develop: its biggest embedded level
        else: arr = images.load(path, 100000)
        tmp = f + '.tmp.npy'; np.save(tmp, arr); os.replace(tmp, f)
        with _lock: _jobs.pop(f, None)
    except Exception as e:
        with _lock: _jobs[f] = dict(state='failed', text=str(e)[:200])


def prepare(pid, path):
    """Start making the full-size copy in the background unless it exists or is being made (a failed one is tried again); returns the status."""
    f = cache_file(pid, path)
    if os.path.exists(f): return dict(state='ready', text='')
    with _lock:
        j = _jobs.get(f)
        if j and j['state'] == 'running': return dict(j)
        _jobs[f] = dict(state='running', text='developing the raw at full size (about a minute and a half, once)' if images.is_camera_raw(path) else 'decoding the picture (once)')
    threading.Thread(target=_build, args=(pid, path, f), daemon=True).start()
    return dict(_jobs[f])


def dims(path):
    """(width, height) of the biggest picture in a file, without decoding it."""
    return images.dims(path)
