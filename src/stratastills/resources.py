"""How much of the machine a batch may use: cores and memory, without a dependency (psutil would do it; this is a page)."""
import contextlib, ctypes, os, sys

GPU_PICTURE_BYTES = 2.5e9       # what a picture being made on the card holds in system memory at its peak (48 MP: the decoded raw, the linear picture, the result: about 1.6 GB; a big panorama several times that)
PER_PICTURE_BYTES = 7e9         # the CPU path holds a few copies of a 48 MP picture as float32 (580 MB each): about 6 GB at its peak


def total_ram():
    """Physical memory in bytes (0 if it cannot be found)."""
    try:
        if sys.platform == 'win32':
            class MS(ctypes.Structure):
                _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong),
                            ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong), ('ullExtendedVirtual', ctypes.c_ulonglong)]
            m = MS(); m.dwLength = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)); return int(m.ullTotalPhys)
        return int(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'))
    except Exception: return 0


def auto_workers(gpu_in_use):
    """How many pictures to make at once. With a GPU the device slots (gpu.slots) already limit how many are on the card, and what is gained is overlapping the CPU parts (decode, encode, files, the network share) with the card's work: gpu.plan_workers (on a 12 GB card 4 pictures at once in tiles, 1.9x one at a time, 11 % better than the best whole-picture batch of 2 or 3 at 1.35x).
    Without one the NumPy code is mostly single-threaded per picture, so cores go unused: one picture per 4 cores, as many as the memory holds (about 7 GB each), at least 1 and at most 6."""
    if gpu_in_use:
        from stratastills import gpu
        ram = total_ram(); return max(1, min(gpu.plan_workers(), int(ram * 0.7 // GPU_PICTURE_BYTES) if ram else 1))                  # each picture in the batch holds a few arrays of its size in system memory
    ram = total_ram(); by_mem = int(ram * 0.8 // PER_PICTURE_BYTES) if ram else 1
    return max(1, min(6, (os.cpu_count() or 1) // 4, by_mem))


# ---- memory per picture: how big a tile may be -------------------------------------------------------------------------------------------------------------------------------------------------------------------

CPU_WHOLE_BYTES_PER_PX = 145    # the whole-picture NumPy development at its peak (about 7 GB for 48 MP)
CPU_RESIDENT_BYTES_PER_PX = 30  # what a picture developed in tiles keeps in system memory throughout (measured 24 to 40 B/px at tiles of 512 to 2048: the decoded raw 6, the linear picture 12, the result 12)
CPU_TILE_BYTES_PER_PX = 200     # per pixel of a tile with its margin: the NumPy development's temporaries (measured 147 at tiles of 2048 with a margin of 128; scripts/tile_memory.py)
MIN_TILE, MAX_TILE = 256, 2048
_workers = 1


@contextlib.contextmanager
def batch(workers):
    """For a batch of `workers` pictures at once: the machine's memory is shared between them (budget)."""
    global _workers
    old = _workers; _workers = max(1, int(workers))
    try: yield
    finally: _workers = old


def budget():
    """The system memory one picture may use now: 70 % of the machine's over the pictures being made at once; None when it cannot be found."""
    ram = total_ram(); return ram * 0.7 / _workers if ram else None


def tile_for(pixels, bytes_per_px, resident=0.0, halo=128, budget_bytes=None):
    """The tile side (multiple of 64, MIN_TILE to MAX_TILE) whose region (tile plus a margin of `halo` on each side) fits in `budget_bytes` at `bytes_per_px` after `resident` bytes a pixel of the picture itself; MAX_TILE when memory is not known."""
    b = budget() if budget_bytes is None else budget_bytes
    if b is None: return MAX_TILE
    room = max(b - pixels * resident, 0.0) / bytes_per_px; side = int(room ** 0.5) - 2 * halo
    return int(min(MAX_TILE, max(MIN_TILE, side // 64 * 64)))


def cpu_tile(pixels):
    """None when a picture of `pixels` is developed whole by the NumPy code (its peak fits the picture's share of memory), else the tile side to develop it in (cpu_tiled.py)."""
    if os.environ.get('STRATASTILLS_TILE'): return max(64, int(os.environ['STRATASTILLS_TILE']))                    # forced (the way to try tiling on a picture that would fit)
    b = budget()
    if b is None or pixels * CPU_WHOLE_BYTES_PER_PX <= b: return None
    return tile_for(pixels, CPU_TILE_BYTES_PER_PX, CPU_RESIDENT_BYTES_PER_PX, budget_bytes=b)
