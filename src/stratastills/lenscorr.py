"""Lens correction made into the DNG opcodes the development already applies (see from_file for where each part comes from: the camera's own shading table, the lensfun profile of the lens; and see USE_EMBEDDED_GEOMETRY for why the camera's distortion table is not used).
The camera's own tables, as the development reads them: the data of a Sony ARW (develop.py: GainMap for the shading, WarpRectilinear for distortion and lateral colour), so every path (CPU, GPU, tiled) corrects a raw that is not a DNG the way it corrects a DJI DNG, with no new code in the paths.
Sony ARWs record, per shot, three correction tables in the raw image's IFD, each as a count and 16 values on knots at (i + 0.5) / (n - 1) of the radius from the centre (1 = the corner): shading (0x7032, with its setting 0x7031), lateral chromatic aberration (0x7035: the red then the blue scale, setting 0x7034) and distortion (0x7037, setting 0x7036). The conventions are those RawTherapee and darktable use for them (rtengine/lensmetadata.cc, adapted there from darktable's iop/lens.cc and common/exif.cc; GPL-3.0): distortion factor 1 + d * 2^-14, colour scale 1 + c * 2^-21 for red and blue, shading gain 1 / 2^(0.5 - 2^(v * 2^-13 - 1)); between knots linear, beyond them the end value. The shading is applied as a gain map (65 x 65 grid, bilinear), the distortion and colour as a per plane polynomial of the radius fitted to the table (WarpRectilinear: source = output * (k0 + k1 r^2 + k2 r^4 + k3 r^6)).
`from_file(path, width, height)` gives the opcode list (bytes) or None; the sizes are those of the picture it will be applied to (the active area, the right way up)."""
import struct
import numpy as np

SHADING, CA, DISTORTION = 0x7032, 0x7035, 0x7037
SETTINGS = {'shading': 0x7031, 'ca': 0x7034, 'distortion': 0x7036}        # the camera's settings of the three corrections (their numbers: LightCraft's notes say shading 256 off 257/272 auto, colour 0 off 1 auto, distortion 0 off 1 auto 17 auto-by-the-lens 255 unavailable)
GRID = 65
VERSION = 1                                    # of the conversion: part of the key of what a stage makes only through develop.VERSION, so bump that with it


def read_tables(path):
    """{'dist': [...], 'ca_r': [...], 'ca_b': [...], 'vig': [...], 'settings': {...}} (the raw integers of the file's tables and of the camera's settings) or None when the file does not carry a valid set."""
    import tifffile
    try:
        with tifffile.TiffFile(path) as t:
            found = {}
            for pg in [t.pages[0]] + list(getattr(t.pages[0], 'pages', []) or []):
                for tag in pg.tags.values():
                    if tag.code in (SHADING, CA, DISTORTION) + tuple(SETTINGS.values()) and tag.code not in found: found[tag.code] = [int(v) for v in (tag.value if hasattr(tag.value, '__len__') else [tag.value])]
    except Exception: return None
    d, c, v = found.get(DISTORTION), found.get(CA), found.get(SHADING)
    if not (d and c and v): return None
    n = d[0]
    if not (1 < n <= 16 and c[0] == 2 * n and v[0] == n and len(d) > n and len(c) > 2 * n and len(v) > n): return None
    return dict(dist=d[1:n + 1], ca_r=c[1:n + 1], ca_b=c[n + 1:2 * n + 1], vig=v[1:n + 1], settings={k: (found.get(c)[0] if found.get(c) else None) for k, c in SETTINGS.items()})


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


def parts(lens, settings=None):
    """(shading, geometry): which of the two corrections the `lens` setting asks for. 'auto': both; 'off': none; 'shading', 'geometry': that one; 'camera': what the camera's own JPEG shows: the geometry when the camera's distortion setting is on (not 0 and not 255; unknown settings: no), never the shading (measured on the A7C with four lenses: its JPEGs are not corrected for it)."""
    if lens == 'off': return False, False
    if lens == 'shading': return True, False
    if lens == 'geometry': return False, True
    if lens == 'camera': d = (settings or {}).get('distortion'); return False, d not in (None, 0, 255)
    return True, True


def keep_opcodes(b, ids):
    """A DNG opcode list with only the opcodes of these ids (9: GainMap, 1: WarpRectilinear); the rest as they were."""
    b = bytes(b); n, = struct.unpack('>I', b[:4]); o = 4; out = []
    for _ in range(n):
        oid, ver, flags, nb = struct.unpack('>IIII', b[o:o + 16]); body = b[o:o + 16 + nb]; o += 16 + nb
        if oid in ids: out.append(body)
    return struct.pack('>I', len(out)) + b''.join(out)


def opcodes_from(gain, warp, width, height, n=GRID):
    """The DNG opcode list (bytes, big-endian) of a shading gain function of the radius (1 = the half diagonal; None: none) and WarpRectilinear coefficients ([3 planes, 6]; None: none) for a picture of width x height: a GainMap, then a WarpRectilinear. Where the correction comes from (the camera's tables, a lensfun profile) does not matter to the development."""
    ops = []
    if gain is not None:
        ys = (np.linspace(0, 1, n) - 0.5) * height; xs = (np.linspace(0, 1, n) - 0.5) * width; r = np.hypot(xs[None, :], ys[:, None]) / np.hypot(width / 2, height / 2); g = np.asarray(gain(r), np.float32)
        gm = struct.pack('>10I', 0, 0, height, width, 0, 3, 1, 1, n, n) + struct.pack('>4d', 1.0 / (n - 1), 1.0 / (n - 1), 0.0, 0.0) + struct.pack('>I', 1) + g.astype('>f4').tobytes(); ops.append(struct.pack('>IIII', 9, 0x01030000, 1, len(gm)) + gm)
    if warp is not None:
        wp = struct.pack('>I', 3) + struct.pack('>18d', *np.asarray(warp).ravel()[:18]) + struct.pack('>2d', 0.5, 0.5); ops.append(struct.pack('>IIII', 1, 0x01030000, 1, len(wp)) + wp)
    return struct.pack('>I', len(ops)) + b''.join(ops) if ops else None


def opcodes(t, width, height, lens='auto'):
    """The opcode list of a set of the camera's tables for a picture of width x height (see opcodes_from, parts)."""
    k, _d, _r, _b, vig = curves(t); sh, ge = parts(lens, t.get('settings'))
    return opcodes_from((lambda r: _interp(r, k, vig)) if sh else None, warp_coefficients(t) if ge else None, width, height)


USE_EMBEDDED_GEOMETRY = False        # the camera's own distortion and colour-fringe tables are NOT used: read with the conventions RawTherapee and darktable use for them they do not follow the A7C's JPEGs (FE 28-60 at 28 mm, 12 pictures, correlation of the edges with the camera JPEG: not corrected 0.79, with the table 0.58, with lensfun's profile 0.97; the JPEG is corrected, the table's amplitude or radius is not what those programs assume: LightCraft found the same for the A7CR and needed its own measured convention). The shading tables are used: they flatten the field (spread of the falloff over many scenes 0.78 -> 0.10 stops, Samyang 35).


def from_file(path, width, height, lens='auto'):
    """The opcode list of a raw's lens correction, each part from the best source there is: the shading from the camera's own table when the file has one (validated: flat fields), else from the lensfun profile of the lens; the geometry (distortion, colour fringes) from the lensfun profile (validated against the camera JPEG), never from the camera's table (USE_EMBEDDED_GEOMETRY). Nothing for a part nothing describes. `lens`: see parts."""
    from stratastills import lensfundb
    t = read_tables(path); sh, ge = parts(lens, t.get('settings') if t else None)
    if not (sh or ge): return None
    gain = warp = None
    if sh and t: k, _d, _r, _b, vig = curves(t); gain = lambda r: _interp(r, k, vig)
    if (sh and gain is None) or ge:
        g2, w2 = lensfundb.corrections_from_exif(path, width, height)
        if sh and gain is None: gain = g2
        if ge: warp = w2
        if ge and warp is None and t and USE_EMBEDDED_GEOMETRY: warp = warp_coefficients(t)
    return opcodes_from(gain, warp, width, height)
