"""Finding the items in a folder: files that belong to one exposure are grouped into an item (a "shot"), and the files of an item are its VARIANTS.

A DJI drone writes, for each press of the shutter, `DJI_<YYYYMMDDHHMMSS>_<NNNN>_D` with an extension per role:
  .DNG            the raw sensor data
  .JPG            the camera's own rendering
  .MP4 .LRF .SRT  a video, its low-resolution proxy and its flight log (video is not a still: listed, not processed)
Work done on a shot in an editor lands next to it with a suffix:
  `...-Enhanced-NR.dng`, `-HDR`, `-Edit`  another rendering of the same picture: a further VARIANT of the same item
  `...-Pano.jpg/.dng`                     a merged panorama of several shots: its OWN item (kind 'pano', stem `<stem>-Pano`, `parent` the shot it is named after), since it is a different picture
A variant is {id, label, path, kind: raw|jpeg|derived, ext}. The default primary variant is the raw when there is one, else the camera's JPEG, else the first derived file; the project can choose another (projects.py).

`scan(folder)` returns the items sorted by name (which, for DJI, is by time); nothing is decoded."""
import os, re
from dataclasses import dataclass, field

RAW_EXTS = ('.dng', '.arw', '.cr2', '.cr3', '.nef', '.orf', '.raf', '.rw2')
STILL_EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp')
VIDEO_EXTS = ('.mp4', '.mov', '.mkv')
SIDECAR_EXTS = ('.lrf', '.srt', '.xmp')
IGNORED = ('thumbs.db', 'desktop.ini', '.ds_store')
PANO_TAGS = ('pano', 'panorama')
_DERIVED = re.compile(r'^(?P<stem>.+?)-(?P<tag>Pano|HDR|Enhanced(?:-NR)?|Edit|Edited|Panorama|Pano-\d+|\d+)$', re.I)


@dataclass
class Shot:
    stem: str
    raw: str | None = None                         # the original raw file
    jpeg: str | None = None                        # the camera's JPEG (or other original still)
    video: str | None = None
    sidecars: list = field(default_factory=list)
    derived: list = field(default_factory=list)    # [(tag, path)] other renderings of this picture made elsewhere
    pano: bool = False                             # a merged panorama (its own item)
    parent: str | None = None                      # for a pano: the stem of the shot it is named after

    @property
    def kind(self): return 'video' if self.video and not (self.raw or self.jpeg or self.derived) else 'pano' if self.pano else 'still'

    @property
    def source(self):
        """The default file to process: the raw when there is one, else the camera JPEG, else the first derived file."""
        return self.raw or self.jpeg or (self.derived[0][1] if self.derived else None)

    def variants(self):
        """The files of this item as [{id, label, path, kind, ext}]: raw, jpeg, then the derived ones. Ids are stable (the extension, or the tag plus extension)."""
        out = []
        for kind, p in (('raw', self.raw), ('jpeg', self.jpeg)):
            if p: out.append(dict(id=kind, label='Raw' if kind == 'raw' else 'Camera JPEG', path=p, kind=kind, ext=os.path.splitext(p)[1].lower()))
        for tag, p in self.derived:
            ext = os.path.splitext(p)[1].lower(); out.append(dict(id=f'{tag.lower()}{ext}', label=f'{tag} ({ext[1:]})', path=p, kind='derived', ext=ext))
        return out

    def to_json(self): return dict(stem=self.stem, kind=self.kind, raw=self.raw, jpeg=self.jpeg, video=self.video, sidecars=self.sidecars, derived=[list(d) for d in self.derived], parent=self.parent, variants=self.variants())


def split_name(filename):
    """(stem, tag, ext) of a file name. `tag` is the editor's suffix ('Pano', 'Enhanced-NR') or '' for a camera original; ext is lowercase with the dot."""
    base, ext = os.path.splitext(filename); m = _DERIVED.match(base)
    return (m.group('stem'), m.group('tag'), ext.lower()) if m else (base, '', ext.lower())


def scan(folder, recursive=False):
    """The items in `folder` (paths are absolute). Files that are neither stills, raws, videos nor their sidecars are skipped."""
    shots = {}; derived = []
    walk = os.walk(folder) if recursive else [(folder, [], os.listdir(folder))]
    for root, _dirs, names in walk:
        for n in sorted(names):
            if n.lower() in IGNORED or n.startswith('.'): continue
            stem, tag, ext = split_name(n); path = os.path.join(root, n)
            if ext not in RAW_EXTS + STILL_EXTS + VIDEO_EXTS + SIDECAR_EXTS: continue
            key = os.path.join(root, stem) if recursive else stem
            if tag: derived.append((key, stem, tag, path, ext)); continue
            s = shots.setdefault(key, Shot(stem=stem))
            if ext in RAW_EXTS: s.raw = path
            elif ext in STILL_EXTS: s.jpeg = path
            elif ext in VIDEO_EXTS: s.video = path
            else: s.sidecars.append(path)
    for key, stem, tag, path, ext in derived:
        if ext not in STILL_EXTS + RAW_EXTS: continue
        if tag.lower() in PANO_TAGS or tag.lower().startswith('pano-'):
            pk = key + '-' + tag; s = shots.setdefault(pk, Shot(stem=stem + '-' + tag, pano=True, parent=stem))
            if ext in RAW_EXTS: s.raw = path
            else: s.jpeg = path
        elif key in shots: shots[key].derived.append((tag, path))
        else: shots[key] = Shot(stem=stem, derived=[(tag, path)])
    return [shots[k] for k in sorted(shots)]
