"""The camera's own lens correction data, made into the DNG opcodes the development already applies (develop.py: GainMap for the shading, WarpRectilinear for distortion and lateral colour), so every path (CPU, GPU, tiled) corrects a raw that is not a DNG the way it corrects a DJI DNG, with no new code in the paths.
Sony ARWs record, per shot, three correction tables in the raw image's IFD, each as a count and 16 values on knots at (i + 0.5) / (n - 1) of the radius from the centre (1 = the corner): shading (0x7032, with its setting 0x7031), lateral chromatic aberration (0x7035: the red then the blue scale, setting 0x7034) and distortion (0x7037, setting 0x7036). The conventions are those RawTherapee and darktable use for them (rtengine/lensmetadata.cc, adapted there from darktable's iop/lens.cc and common/exif.cc; GPL-3.0): distortion factor 1 + d * 2^-14, colour scale 1 + c * 2^-21 for red and blue, shading gain 1 / 2^(0.5 - 2^(v * 2^-13 - 1)); between knots linear, beyond them the end value. The shading is applied as a gain map (65 x 65 grid, bilinear), the distortion and colour as a per plane polynomial of the radius fitted to the table (WarpRectilinear: source = output * (k0 + k1 r^2 + k2 r^4 + k3 r^6)).
`from_file(path, width, height)` gives the opcode list (bytes) or None; the sizes are those of the picture it will be applied to (the active area, the right way up)."""
import struct
import numpy as np

SHADING, CA, DISTORTION = 0x7032, 0x7035, 0x7037
GRID = 65
VERSION = 1                                    # of the conversion: part of the key of what a stage makes only through develop.VERSION, so bump that with it


def read_tables(path):
    """{'dist': [...], 'ca_r': [...], 'ca_b': [...], 'vig': [...]} (the raw integers of the file's tables) or None when the file does not carry a valid set."""
    import tifffile
    try:
        with tifffile.TiffFile(path) as t:
            found = {}
            for pg in [t.pages[0]] + list(getattr(t.pages[0], 'pages', []) or []):
                for tag in pg.tags.values():
                    if tag.code in (SHADING, CA, DISTORTION) and tag.code not in found: found[tag.code] = [int(v) for v in tag.value]
    except Exception: return None
    d, c, v = found.get(DISTORTION), found.get(CA), found.get(SHADING)
    if not (d and c and v): return None
    n = d[0]
    if not (1 < n <= 16 and c[0] == 2 * n and v[0] == n and len(d) > n and len(c) > 2 * n and len(v) > n): return None
    return dict(dist=d[1:n + 1], ca_r=c[1:n + 1], ca_b=c[n + 1:2 * n + 1], vig=v[1:n + 1])


def _interp(r, knots, values):
    return np.interp(r, knots, values)                                                                  # linear between the knots, the end values beyond them (np.interp clamps)


def curves(t):
    """The functions of the normalised radius r of a set of tables: (distortion factor, red scale, blue scale, shading gain), as arrays on knots (knots, dist, ca_r, ca_b, vig)."""
    n = len(t['dist']); k = (np.arange(n) + 0.5) / (n - 1)
    return k, np.array(t['dist'], float) * 2.0 ** -14 + 1, np.array(t['ca_r'], float) * 2.0 ** -21 + 1, np.array(t['ca_b'], float) * 2.0 ** -21 + 1, 1.0 / 2.0 ** (0.5 - 2.0 ** (np.array(t['vig'], float) * 2.0 ** -13 - 1))


def gain_grid(t, width, height, n=GRID):
    """The shading gain on an n x n grid over a picture of width x height (float32 [n, n])."""
    k, _d, _r, _b, vig = curves(t); ys = (np.linspace(0, 1, n) - 0.5) * height; xs = (np.linspace(0, 1, n) - 0.5) * width; r = np.hypot(xs[None, :], ys[:, None]) / np.hypot(width / 2, height / 2)
    return _interp(r, k, vig).astype(np.float32)


def warp_coefficients(t):
    """[3 planes (red, green, blue), 6] (kr0..kr3, kt0, kt1): the polynomial in r^2 that best follows distortion times colour scale of each plane over the radius the picture has (0 to the corner and a little beyond)."""
    k, dist, car, cab, _v = curves(t); r = np.linspace(0, 1.05, 200); d = _interp(r, k, dist); out = np.zeros((3, 6))
    for p, f in enumerate((d * _interp(r, k, car), d, d * _interp(r, k, cab))):
        out[p, :4] = np.linalg.lstsq(np.vander(r * r, 4, increasing=True), f, rcond=None)[0]
    return out


def opcodes(t, width, height):
    """The DNG opcode list (bytes, big-endian) of a set of tables for a picture of width x height: a GainMap of the shading, then a WarpRectilinear of distortion and colour."""
    g = gain_grid(t, width, height); n = g.shape[0]
    gm = struct.pack('>10I', 0, 0, height, width, 0, 3, 1, 1, n, n) + struct.pack('>4d', 1.0 / (n - 1), 1.0 / (n - 1), 0.0, 0.0) + struct.pack('>I', 1) + g.astype('>f4').tobytes()
    c = warp_coefficients(t); wp = struct.pack('>I', 3) + struct.pack('>18d', *c.ravel()) + struct.pack('>2d', 0.5, 0.5)
    return struct.pack('>I', 2) + struct.pack('>IIII', 9, 0x01030000, 1, len(gm)) + gm + struct.pack('>IIII', 1, 0x01030000, 1, len(wp)) + wp


def from_file(path, width, height):
    t = read_tables(path); return None if t is None else opcodes(t, width, height)
