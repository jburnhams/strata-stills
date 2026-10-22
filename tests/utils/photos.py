"""Synthetic photos for tests: no real pictures, models or network. Importable by name in any test (`from photos import ...`; tests/utils is on pythonpath).

  texture(w, h)            a sharp natural-like picture (a random field with a 1/f spectrum), float32 0.1..0.9
  make_jpeg(path, ...)     a JPEG with the XMP block a DJI drone writes (gimbal pitch, time), so the shot is read as a drone photo
  make_linear_dng(path)    a small Lightroom-style linear DNG (demosaiced RGB in a sub-IFD, DJI make and model, colour matrices, as-shot neutral): what lineardng.py reads
  make_shoot(folder)       a folder like the drone's: JPEG shots (one straight-down, the rest oblique), a panorama as a linear DNG, a video stub
"""
import os, struct

import numpy as np
from PIL import Image


def texture(w=1200, h=900, seed=1):
    rng = np.random.default_rng(seed); f = np.fft.fftfreq(w)[None, :], np.fft.fftfreq(h)[:, None]; r = np.hypot(*np.broadcast_arrays(*f)); r[0, 0] = 1
    z = np.fft.ifft2(np.fft.fft2(rng.standard_normal((h, w))) / r).real; z = (z - z.min()) / (z.max() - z.min()); return (0.1 + 0.8 * z).astype(np.float32)


def dji_xmp(pitch=-90.0, taken='2025-04-02T19:01:11', serial='1581F5BKB23C9001'):
    attrs = (f'drone-dji:GimbalPitchDegree="{pitch:+.2f}" drone-dji:GimbalYawDegree="+12.30" drone-dji:GimbalRollDegree="+0.00" drone-dji:AbsoluteAltitude="+58.20" '
             f'drone-dji:RelativeAltitude="+40.10" drone-dji:CameraSerialNumber="{serial}" xmp:CreateDate="{taken}"')
    return ('<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?><x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            '<rdf:Description rdf:about="" xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" xmlns:xmp="http://ns.adobe.com/xap/1.0/" ' + attrs + '/></rdf:RDF></x:xmpmeta><?xpacket end="w"?>').encode()


def make_jpeg(path, w=800, h=600, pitch=-90.0, taken='2025-04-02T19:01:11', seed=1, tilt=0.0):
    """A JPEG shot with the DJI XMP: a textured picture with a grid of lines on it, the whole turned by `tilt` degrees so that levelling has something to find."""
    import cv2
    y = texture(w, h, seed)
    for x in range(0, w, max(40, w // 12)): cv2.line(y, (x, 0), (x, h), 0.05, 3)
    for r in range(0, h, max(40, h // 9)): cv2.line(y, (0, r), (w, r), 0.05, 3)
    img = (np.dstack([y, y * 0.92, y * 0.82]) * 255).astype(np.uint8)
    if tilt: img = cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), tilt, 1.1), (w, h), borderMode=cv2.BORDER_REFLECT)
    Image.fromarray(img).save(path, quality=92, xmp=dji_xmp(pitch, taken)); return path


def make_linear_dng(path, w=320, h=240, seed=3, make='DJI', model='FC8482'):
    """A linear DNG: a small preview in the main IFD and the demosaiced 16 bit RGB in a sub-IFD marked LinearRaw (tifffile writes it as RGB; the photometric tag is patched afterwards, as it cannot write that value itself)."""
    import tifffile
    rng = np.random.default_rng(seed); base = texture(w, h, seed)[..., None] * np.array([0.9, 1.0, 0.8], np.float32); raw = np.clip(base * 40000 + rng.normal(0, 150, (h, w, 3)) + 4096, 0, 65535).astype(np.uint16)
    prev = np.ascontiguousarray(np.clip(base[::8, ::8] * 255, 0, 255).astype(np.uint8)); rat = lambda *v: [x for a, b in v for x in (a, b)]
    cm = rat(*[(int(v * 10000), 10000) for v in (1.2, -0.6, 0.1, -0.5, 1.4, 0.5, -0.03, 0.1, 0.9)])
    main = [(50706, 1, 4, (1, 4, 0, 0)), (50728, 5, 3, rat((4152, 10000), (1, 1), (6463, 10000))), (50721, 10, 9, cm), (50722, 10, 9, cm), (50778, 3, 1, 17), (50779, 3, 1, 21), (271, 2, len(make) + 1, make), (272, 2, len(model) + 1, model)]
    with tifffile.TiffWriter(path) as t:
        t.write(prev, photometric='rgb', subifds=1, extratags=main, subfiletype=1)
        t.write(raw, photometric='rgb', subfiletype=0, extratags=[(50714, 3, 3, (4096, 4096, 4096)), (50717, 3, 3, (65535, 65535, 65535))])
    with tifffile.TiffFile(path) as t: off, order = t.pages[0].pages[0].tags['PhotometricInterpretation'].valueoffset, t.byteorder
    with open(path, 'r+b') as f: f.seek(off); f.write(struct.pack(order + 'H', 34892))                                           # 34892 = LinearRaw
    return path


def make_shoot(folder, n=2):
    """A drone-like folder: `n` JPEG shots (the first straight down, the others oblique, a minute apart), a panorama as a linear DNG named after the first shot, and a video stub."""
    os.makedirs(folder, exist_ok=True)
    for i in range(n): make_jpeg(os.path.join(folder, f'DJI_2025040219{i + 1:02d}11_{i + 1:04d}_D.JPG'), pitch=-90.0 if i == 0 else -45.0, taken=f'2025-04-02T19:{i + 1:02d}:11', seed=i + 1, tilt=3.0 * (i + 1))
    make_linear_dng(os.path.join(folder, 'DJI_20250402190111_0001_D-Pano.dng'))
    with open(os.path.join(folder, 'DJI_20250402190511_0009_D.MP4'), 'wb') as f: f.write(b'not a video')
    return folder
