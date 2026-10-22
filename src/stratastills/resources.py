"""How much of the machine a batch may use: cores and memory, without a dependency (psutil would do it; this is a page)."""
import ctypes, os, sys

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
