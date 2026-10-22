"""Finding the shots in a folder: files that belong to one exposure are grouped into a shot.

A DJI drone writes, for each press of the shutter, `DJI_<YYYYMMDDHHMMSS>_<NNNN>_D` with an extension per role:
  .DNG            the raw sensor data (the best source)
  .JPG            the camera's own rendering (the fallback source, and the reference for what the camera thought the picture should look like)
  .MP4 .LRF .SRT  a video, its low-resolution proxy and its flight log (video is not a still: noted, not processed)
Work already done on a shot in an editor lands next to it with a suffix: `...-Pano.jpg`, `...-Enhanced-NR.dng`. Those are DERIVED files: kept with the shot they came from (the stem is the same up to the suffix), never taken as a new shot, and never overwritten.

`scan(folder)` returns the shots sorted by name (which, for DJI, is by time); nothing is decoded."""
import os, re
from dataclasses import dataclass, field

RAW_EXTS = ('.dng', '.arw', '.cr2', '.cr3', '.nef', '.orf', '.raf', '.rw2')
STILL_EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp')
VIDEO_EXTS = ('.mp4', '.mov', '.mkv')
SIDECAR_EXTS = ('.lrf', '.srt', '.xmp')
IGNORED = ('thumbs.db', 'desktop.ini', '.ds_store')
_DERIVED = re.compile(r'^(?P<stem>.+?)-(?P<tag>Pano|HDR|Enhanced(?:-NR)?|Edit|Edited|Panorama|Pano-\d+|\d+)$', re.I)


@dataclass
class Shot:
    stem: str
    raw: str | None = None                         # the original raw file
    jpeg: str | None = None                        # the camera's JPEG (or other original still)
    video: str | None = None
    sidecars: list = field(default_factory=list)
    derived: list = field(default_factory=list)    # [(tag, path)] work done elsewhere on this shot

    @property
    def kind(self): return 'video' if self.video and not (self.raw or self.jpeg) else 'still'

    @property
    def source(self):
        """The file to process: the raw when there is one, else the camera JPEG."""
        return self.raw or self.jpeg

    def to_json(self): return dict(stem=self.stem, kind=self.kind, raw=self.raw, jpeg=self.jpeg, video=self.video, sidecars=self.sidecars, derived=[list(d) for d in self.derived])


def split_name(filename):
    """(stem, tag, ext) of a file name. `tag` is the editor's suffix ('Pano', 'Enhanced-NR') or '' for a camera original; ext is lowercase with the dot."""
    base, ext = os.path.splitext(filename); m = _DERIVED.match(base)
    return (m.group('stem'), m.group('tag'), ext.lower()) if m else (base, '', ext.lower())


def scan(folder, recursive=False):
    """The shots in `folder` (paths are absolute). Files that are neither stills, raws, videos nor their sidecars are skipped."""
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
        if key in shots: shots[key].derived.append((tag, path))
        elif ext in STILL_EXTS + RAW_EXTS: shots[key] = Shot(stem=stem, derived=[(tag, path)])          # the original is gone: the edited file is all there is, listed but not offered as a source
    return [shots[k] for k in sorted(shots)]
