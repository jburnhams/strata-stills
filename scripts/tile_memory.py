"""Measure the memory of the CPU development, whole and in tiles of several sizes, and the time: the figures behind resources.CPU_*_BYTES_PER_PX. Windows (peak working set) or anything with `resource`.
Usage: python scripts/tile_memory.py <raw file>"""
import os, subprocess, sys, time


def peak_bytes():
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes

        class C(ctypes.Structure): _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD), ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t), ('a', ctypes.c_size_t), ('b', ctypes.c_size_t), ('c', ctypes.c_size_t), ('d', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
        k32 = ctypes.WinDLL('kernel32'); k32.GetCurrentProcess.restype = wintypes.HANDLE; ps = ctypes.WinDLL('psapi'); ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(C), wintypes.DWORD]; ps.GetProcessMemoryInfo.restype = wintypes.BOOL
        c = C(); c.cb = ctypes.sizeof(C)
        if not ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb): raise OSError('GetProcessMemoryInfo failed')
        return c.PeakWorkingSetSize
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)


def one(path, tile):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
    from stratastills import develop as D, cpu_tiled
    u16 = D._raw_u16(path, False); base = peak_bytes(); t = time.time()
    if tile == 0: D._raw_u16 = lambda p, half: u16; D.develop(path, backend='cpu')                                       # the whole-picture NumPy development (the decode is not repeated)
    else: cpu_tiled.develop_tiled(path, u16, 'auto', 'auto', 0.35, D.HIGHLIGHT_CONTRAST, None, tile=tile)
    px = u16.shape[0] * u16.shape[1]; print(f'tile {tile or "whole":>5}: {time.time() - t:6.1f} s  peak {peak_bytes() / 1e9:5.2f} GB ({peak_bytes() / px:5.0f} B/px, {(peak_bytes() - base) / px:5.0f} B/px above the decoded raw)', flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 3: one(sys.argv[1], int(sys.argv[2]))
    else:
        for tile in (0, 2048, 1024, 512): subprocess.run([sys.executable, '-I', __file__, sys.argv[1], str(tile)])
