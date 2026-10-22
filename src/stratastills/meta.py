"""Metadata of one photo file: EXIF (what the camera wrote) and, for DJI drones, the XMP block of the JPEG (position, gimbal and flight attitude).

Only the first `HEAD_BYTES` of the file are read for XMP, so scanning a folder on a network share does not pull whole 35 MB JPEGs.
The DJI gimbal pitch is what tells a straight-down (nadir) map shot (-90) from an oblique one; it decides how a photo is levelled and framed."""
import datetime as dt, re

HEAD_BYTES = 262144
_XMP = re.compile(rb'<x:xmpmeta.*?</x:xmpmeta>', re.S)
_ATTR = re.compile(r'([\w-]+:[\w-]+)="([^"]*)"')
DJI_NUMERIC = ('GpsLatitude', 'GpsLongitude', 'AbsoluteAltitude', 'RelativeAltitude', 'GimbalRollDegree', 'GimbalYawDegree', 'GimbalPitchDegree', 'FlightRollDegree', 'FlightYawDegree', 'FlightPitchDegree')
DJI_TEXT = ('ProductName', 'GpsStatus', 'ShutterType', 'CameraSerialNumber')
NADIR_PITCH = -80.0           # a gimbal pitch at or below this is a straight-down shot


def read_xmp(path, head=HEAD_BYTES):
    """The XMP attributes of a JPEG (or other file with an XMP packet in its head) as {'drone-dji:GimbalPitchDegree': '-90.00', ...}; {} when there is none."""
    try:
        with open(path, 'rb') as f: b = f.read(head)
    except OSError: return {}
    m = _XMP.search(b)
    return dict(_ATTR.findall(m.group(0).decode('utf8', 'replace'))) if m else {}


def dji_fields(xmp):
    """The DJI drone fields of an XMP attribute dict: numbers as floats (a missing or unreadable one is left out), text as is, keyed without the namespace."""
    out = {}
    for k in DJI_NUMERIC:
        try: out[k] = float(xmp['drone-dji:' + k])
        except (KeyError, ValueError): pass
    for k in DJI_TEXT:
        if xmp.get('drone-dji:' + k): out[k] = xmp['drone-dji:' + k]
    return out


def parse_time(s):
    """A datetime from an XMP / EXIF time stamp ('2025-04-02T19:01:11+01:00', '2025-04-02T19:01:30.1+01:00' or '2025:04:02 19:01:11'); None when unreadable. Timezone-aware when it carries an offset."""
    if not s: return None
    s = s.strip().replace('Z', '+00:00')
    if re.match(r'\d{4}:\d\d:\d\d ', s): s = s[:10].replace(':', '-') + 'T' + s[11:]
    try: return dt.datetime.fromisoformat(s)
    except ValueError: return None


def photo_info(path):
    """What is known about a photo without decoding it: {taken, width, height, camera, dji: {...}, xmp_tool, merged_pano, preserved_name}. Missing facts are None / absent."""
    from PIL import Image
    info = dict(taken=None, width=None, height=None, camera=None, dji={}, xmp_tool=None, merged_pano=False, preserved_name=None)
    xmp = read_xmp(path)
    info['dji'] = dji_fields(xmp)
    info['xmp_tool'] = xmp.get('xmp:CreatorTool')
    info['merged_pano'] = xmp.get('aux:IsMergedPanorama') == 'True'
    info['preserved_name'] = xmp.get('xmpMM:PreservedFileName')
    info['taken'] = parse_time(xmp.get('xmp:CreateDate'))
    try:
        with Image.open(path) as im:
            info['width'], info['height'] = im.size
            ex = im.getexif()
            info['camera'] = ' '.join(x for x in (ex.get(271), ex.get(272)) if x) or None
            if info['taken'] is None: info['taken'] = parse_time(ex.get_ifd(0x8769).get(36867))
    except Exception: pass
    return info


def is_nadir(dji):
    p = dji.get('GimbalPitchDegree')
    return p is not None and p <= NADIR_PITCH
