"""Pipelines: making a finished picture out of a source variant is a sequence of STAGES, each with its own settings and its own version, so a pipeline is just data: a config (a list of stage settings) that can be saved as a named PRESET, run on one item from the app, or applied by RULES to every item of a source folder.

  develop   read the source at full size: a raw is developed (develop.py), any other still is read as it is                          settings: highlight_contrast, exposure ('auto', 'camera' or stops from the camera's)
  level     turn the picture square to the scene (transforms.py), measured on the working picture, applied at full size              (no settings; leave the stage out to skip it)
  resize    the size to make it: 'auto' (the detail limit measured on the picture itself, detail.py), 'native', or a long side in px   settings: size
  sharpen   a Wiener filter built from the measured blur of this very picture, luminance only, no halos                               settings: amount (1 undoes the blur; 1.5 is as crisp as a camera JPEG)
  encode    JPEG (4:4:4; quality 90 by default, or 'auto' = the lowest whose error the picture's own noise hides, jpeg_quality.py) or 16 bit TIFF      settings: format, quality

`develop` comes first and `encode` last; the others are in that order and each appears at most once. A generated picture is a new VARIANT of the item (projects.generated_variants), identified by the KEY of the stage settings that made it (item, source variant, every stage's settings and version):
asking for the same thing again finds the existing variant instead of making another, which is what greys Generate out in the app.

Presets: built in (BUILTIN: 'best' is the default and needs nothing chosen) or the person's own, saved as JSON under `<settings home>/pipelines/`. RULES in project.json (`auto`: [{source: <source id or '*'>, pipeline: <preset>, variant: <optional source variant id>}]) say which presets to apply to which source folders;
`apply_rules` generates what is missing, so adding a folder and pressing one button gives the preferred variants of every picture in it.

Nothing is ever written into a source folder."""
import contextlib, datetime as dt, hashlib, json, math, os, re

import cv2, numpy as np

from stratastills import detail, develop, export as X, images, inputs, jpeg_quality, match, preview, projects as P, settings, sizeplan, transforms

DEFAULT_QUALITY = 90            # the JPEG quality of the built-in presets: chosen by the user from side-by-side crops of q85 to q99 (docs/progress.md); 'auto' (jpeg_quality.py) stays available in the Advanced panel
SHARPEN_AMOUNT = 1.5            # past plain restoration (1.0) to the crispness of a camera's own JPEG, chosen by eye on brick fronts and roofs against the Mini 4 Pro JPEGs
ORDER = ('develop', 'level', 'resize', 'sharpen', 'encode')
LABELS = dict(develop='Develop', level='Level', resize='Size', sharpen='Sharpen', encode='Encode')


# ---- the stages -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _num(params, k, lo, hi, default):
    try: v = round(float(params.get(k, default)), 3)
    except (TypeError, ValueError): raise P.ProjectError(f'{k} must be a number')
    if not lo <= v <= hi: raise P.ProjectError(f'{k} must be between {lo:g} and {hi:g}')
    return v


def _exposure(v):
    """'auto' (the median of the picture is brought to a fixed lightness), 'camera' (the exposure the camera's own JPEG has: needs the device's tone curve, see inputs.py) or a number of stops added to the camera's exposure; 0 is 'camera'."""
    if str(v).lower() in ('auto', 'camera'): return str(v).lower()
    try: n = round(float(v), 2)
    except (TypeError, ValueError): raise P.ProjectError("develop exposure is 'auto', 'camera' or a number of stops")
    if not -5 <= n <= 5: raise P.ProjectError('exposure must be between -5 and 5 stops')
    return 'camera' if n == 0 else n


def _develop_params(p):
    m = str(p.get('match', 'auto')).lower()
    if m not in ('auto', 'off'): raise P.ProjectError("develop match is 'auto' or 'off'")
    return dict(highlight_contrast=_num(p, 'highlight_contrast', 0, 3, develop.HIGHLIGHT_CONTRAST), match=m, exposure=_exposure(p.get('exposure', 'auto')))                    # match: a merged panorama is brought to the levels of its neighbouring frames (match.py)
def _level_params(p): return dict(nadir=bool(p.get('nadir', False)))                                                    # straight-down shots are not levelled unless asked: their lines (roofs, kerbs) are weak evidence


def _resize_params(p):
    s = str(p.get('size', 'auto')).lower()
    if s in ('auto', 'native'): return dict(size=s)
    try: n = int(float(s))
    except ValueError: raise P.ProjectError(f'bad size {s!r}')
    if n < 64: raise P.ProjectError('the size is too small')
    return dict(size=str(n))


def _sharpen_params(p): return dict(amount=_num(p, 'amount', 0, 3, SHARPEN_AMOUNT))


def _encode_params(p):
    fmt = p.get('format', 'jpeg')
    if fmt not in X.FORMATS: raise P.ProjectError(f'unknown format {fmt!r}')
    q = p.get('quality', DEFAULT_QUALITY)
    if fmt == 'tiff': return dict(format='tiff', quality='auto')                       # a TIFF has no quality: a setting that does nothing must not make a different key
    if str(q).lower() == 'auto': return dict(format='jpeg', quality='auto')
    try: q = int(q)
    except (TypeError, ValueError): raise P.ProjectError('the JPEG quality must be a number or auto')
    if not 60 <= q <= 100: raise P.ProjectError('the JPEG quality must be between 60 and 100')
    return dict(format='jpeg', quality=q)


PARAMS = dict(develop=_develop_params, level=_level_params, resize=_resize_params, sharpen=_sharpen_params, encode=_encode_params)


def version(stage):
    """The version of a stage's code: part of a picture's key, so a change of the recipe is a different picture."""
    return dict(develop=develop.VERSION, level=transforms.VERSION['level'], resize=1000 * detail.TILE + 2, sharpen=3, encode=2)[stage]


class State:
    """What flows through the stages of one run."""
    def __init__(self, pid, iid, item, variant, say):
        self.pid, self.iid, self.item, self.variant, self.say = pid, iid, item, variant, say
        self.path = variant['path']; self.rgb = None; self.native = None; self.scale = 1.0; self._res = None; self.rec = dict(steps=[]); self.quality = None; self.file = None; self.analysis = None

    @property
    def res(self):
        if self._res is None: self.say('measuring the real detail'); self._res = X.measure_detail(self.pid, self.path, luma=self.analysis)
        return self._res


def _reference_picture(item):
    """A developed (half size) neighbouring frame to match a panorama to: its raw when it has one, else its JPEG."""
    v = next((x for x in item['variants'] if x['kind'] == 'raw'), None) or next((x for x in item['variants'] if x['kind'] == 'jpeg'), None)
    if v is None: return None
    if X._is_camera_raw(v['path']): return develop.develop(v['path'], half=True)
    return images.load(v['path'], 100000)[:, :, ::-1].astype(np.float32) / 255.0


def run_develop(st, p):
    st.say('developing the full-size picture'); an = {}
    st.rgb, how = X._read_full(st.path, analysis=an, highlight_contrast=p['highlight_contrast'], ev=p.get('exposure', 'auto')) if X._is_camera_raw(st.path) else X._read_full(st.path); st.analysis = an.get('luma')
    st.rec['input'] = inputs.describe(st.path); st.rec['steps'].append(how); h, w = st.rgb.shape[:2]; st.native = [w, h]; st.rec['native'] = [w, h]
    if p.get('match', 'auto') == 'auto' and st.rec['input']['kind'] == 'linear-dng':                                      # a merge is exposed by its own content: bring it to the levels of the frames it was made from
        refs = match.references(st.item, P.source_items(st.pid, st.item['id'].partition('~')[0]))
        if not refs: st.rec['match'] = dict(applied=False, reason='no neighbouring frames found by time stamp'); return
        st.say('matching the levels of the neighbouring frames'); pics = [r for r in (_reference_picture(it) for it in refs) if r is not None]
        st.rgb, rec = match.match_levels(st.rgb, pics); st.rec['match'] = dict(rec, references=[it['stem'] for it in refs])
        if rec['applied']: st.rec['steps'].append(f"levels matched to {', '.join(it['stem'] for it in refs)} ({rec['points']} points, rms {rec['rms_before']:.3f} -> {rec['rms_after']:.3f})")


def run_level(st, p):
    st.say('levelling')
    if st.analysis is not None:                                                                                      # the tilt is measured on the analysis luminance the develop pass already made (reduced to 2048 px): no second develop
        a = st.analysis; k = min(1.0, 2048 / max(a.shape)); small = cv2.resize(a, (round(a.shape[1] * k), round(a.shape[0] * k)), interpolation=cv2.INTER_AREA); info = transforms.level_info((small * 255 + 0.5).astype(np.uint8), dict(preview._ctx(st.item), level_nadir=p['nadir']))
    else: work = images.working(P.folder(st.pid), st.path, preview.WORK_PX); info = transforms.apply('level', work, dict(preview._ctx(st.item), level_nadir=p['nadir']))[1]
    st.rec['level'] = {k: info[k] for k in ('applied', 'angle', 'confidence', 'reason') if k in info}
    if info.get('applied'): st.rgb = transforms.rotate_fill(st.rgb, info['angle']); st.rec['steps'].append(f"levelled by {info['angle']:+.2f} degrees")


def run_resize(st, p):
    res = st.res; st.rec['detail'] = {k: res.get(k) for k in ('ok', 'f50', 'f30', 'f20', 'scale50', 'scale30', 'scale20', 'recommended', 'alpha')}
    h, w = st.rgb.shape[:2]; long_native = max(w, h); out_long, why = X.choose_size(p['size'], long_native, res)
    if p['size'] in (None, 'auto'):                                                                                  # the folder's size for pictures of this native size, so they all come out alike
        try: fs = sizeplan.folder_size(st.pid, st.item, (w, h), lambda path: X.measure_detail(st.pid, path), lambda it: source_variant(it))
        except Exception: fs = None
        if fs: out_long, why = fs, 'the detail limit measured over this folder (median of a sample of its pictures of this size)'
    floor = sizeplan.min_share(st.rec.get('input', {}).get('kind'))
    if p['size'] in (None, 'auto') and out_long < floor * long_native: out_long, why = math.ceil(floor * long_native), f'{floor:.0%} of the native size: auto never goes below it'
    if p['size'] in (None, 'auto'): ow, oh = sizeplan.output_dims(w, h, out_long)                                      # tidy dimensions; a size asked for in pixels is made exactly
    else: ow, oh = (out_long, max(1, round(h * out_long / w))) if w >= h else (max(1, round(w * out_long / h)), out_long)
    st.scale = max(ow, oh) / long_native; st.rec['size'] = dict(long_side=max(ow, oh), why=why)
    if (ow, oh) != (w, h):
        st.say('resizing'); st.rgb = cv2.resize(st.rgb, (ow, oh), interpolation=cv2.INTER_LANCZOS4 if st.scale > 0.5 else cv2.INTER_AREA); st.rec['steps'].append(f'resized to {ow}x{oh}')


def run_sharpen(st, p):
    res = st.res
    if not res.get('ok'): st.rec['steps'].append('not sharpened: too little texture to measure the blur'); return
    st.say('sharpening to the measured blur'); st.rgb = detail.sharpen_picture(st.rgb, np.array(res['freq']), np.array(res['mtf']), st.scale, amount=p['amount']); st.rec['steps'].append(f"sharpened with the Wiener filter of the measured MTF (amount {p['amount']:g})")


def run_encode(st, p, out_base):
    fmt = p['format']; quality = None
    if fmt == 'jpeg':
        if p['quality'] == 'auto':
            st.say('choosing the JPEG quality'); quality, qi = jpeg_quality.choose_quality(X.to_u8(st.rgb)); st.rec['quality_info'] = qi
            st.rec['steps'].append(f"JPEG quality {quality} chosen automatically (error {qi.get('rms')} against a target of {qi.get('target')} grey levels; the picture's own noise is {qi.get('sigma')})")
        else: quality = int(p['quality']); st.rec['steps'].append(f'JPEG quality {quality} as asked')
    out = out_base + ('.tif' if fmt == 'tiff' else '.jpg'); st.say('writing'); X.write_picture(out, st.rgb, fmt, quality, X._exif_of(st.item)); st.file = out; st.quality = quality


# ---- configs and presets --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

BEST = [dict(stage='develop'), dict(stage='level'), dict(stage='resize', size='auto'), dict(stage='sharpen', amount=SHARPEN_AMOUNT), dict(stage='encode', format='jpeg', quality=DEFAULT_QUALITY)]
BUILTIN = {
    'best': dict(label='Best quality (automatic)', stages=BEST),
    'archive': dict(label='Archive: 16 bit TIFF, native size, as developed', stages=[dict(stage='develop'), dict(stage='level'), dict(stage='resize', size='native'), dict(stage='encode', format='tiff')]),
    'web': dict(label='Web: 2048 px JPEG', stages=[dict(stage='develop'), dict(stage='level'), dict(stage='resize', size='2048'), dict(stage='sharpen', amount=1.0), dict(stage='encode', format='jpeg', quality=DEFAULT_QUALITY)]),
}
DEFAULT_PRESET = 'best'


def normalize(stages):
    """The stage settings with every default filled in and every value checked, as a new list [{stage, ...settings}]. `develop` must be first and `encode` last, the others in ORDER and at most once; an unknown stage or setting is refused."""
    out = []; last = -1
    for s in stages or []:
        name = s.get('stage')
        if name not in PARAMS: raise P.ProjectError(f'unknown stage {name!r} (one of {", ".join(ORDER)})')
        k = ORDER.index(name)
        if k <= last: raise P.ProjectError(f'stage {name!r} is out of order or repeated (the order is {", ".join(ORDER)})')
        extra = set(s) - {'stage'} - set(PARAMS[name]({}))
        if extra: raise P.ProjectError(f'stage {name!r} has no setting {sorted(extra)[0]!r}')
        last = k; out.append(dict(stage=name, **PARAMS[name](s)))
    if not out or out[0]['stage'] != 'develop' or out[-1]['stage'] != 'encode': raise P.ProjectError('a pipeline starts with develop and ends with encode')
    return out


def apply_options(stages, options):
    """The stages with the Advanced panel's flat options laid over them (only those given): size, format, quality, level (False removes the stage), sharpen (False removes it), sharpen_amount, highlight_contrast. Returns a new, normalised list."""
    st = [dict(s) for s in stages]; o = dict(options or {})
    unknown = set(o) - {'size', 'format', 'quality', 'level', 'sharpen', 'sharpen_amount', 'highlight_contrast', 'match', 'exposure'}
    if unknown: raise P.ProjectError(f'unknown option {sorted(unknown)[0]!r}')

    def get(name): return next((s for s in st if s['stage'] == name), None)

    def ensure(name, **kw):
        s = get(name)
        if s is None:
            s = dict(stage=name); st.append(s); st.sort(key=lambda x: ORDER.index(x['stage']))
        s.update(kw); return s
    if o.get('highlight_contrast') is not None: ensure('develop', highlight_contrast=o['highlight_contrast'])
    if o.get('match') is not None: ensure('develop', match=o['match'])
    if o.get('exposure') is not None: ensure('develop', exposure=o['exposure'])
    if o.get('size') is not None: ensure('resize', size=o['size'])
    if o.get('level') is False: st = [s for s in st if s['stage'] != 'level']
    elif o.get('level') is True: ensure('level')
    if o.get('sharpen') is False: st = [s for s in st if s['stage'] != 'sharpen']
    elif o.get('sharpen') is True or o.get('sharpen_amount') is not None: ensure('sharpen', **({'amount': o['sharpen_amount']} if o.get('sharpen_amount') is not None else {}))
    if o.get('format') is not None: ensure('encode', format=o['format'])
    if o.get('quality') is not None: ensure('encode', quality=o['quality'])
    return normalize(st)


def _preset_dir(): return os.path.join(settings.home(), 'pipelines')
def _safe(name):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,40}', name or ''): raise P.ProjectError('a preset name is lower-case letters, digits, - and _ (up to 41)')
    return name


def list_presets():
    """[{name, label, builtin, stages}]: the built-in presets, then the person's own (a saved preset of the same name as a built-in one is not allowed)."""
    out = [dict(name=n, label=d['label'], builtin=True, stages=normalize(d['stages'])) for n, d in BUILTIN.items()]
    d = _preset_dir()
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if not f.endswith('.json') or f[:-5] in BUILTIN: continue
            try: r = json.load(open(os.path.join(d, f))); out.append(dict(name=f[:-5], label=r.get('label') or f[:-5], builtin=False, stages=normalize(r['stages'])))
            except (OSError, ValueError, KeyError, P.ProjectError): continue
    return out


def get_preset(name):
    for p in list_presets():
        if p['name'] == name: return p
    raise P.ProjectError(f'no pipeline preset {name!r}')


def save_preset(name, label, stages):
    name = _safe(name)
    if name in BUILTIN: raise P.ProjectError(f'{name!r} is a built-in preset: save yours under another name')
    st = normalize(stages); os.makedirs(_preset_dir(), exist_ok=True)
    with open(os.path.join(_preset_dir(), name + '.json'), 'w') as f: json.dump(dict(label=label or name, stages=st), f, indent=1)
    return get_preset(name)


def delete_preset(name):
    if name in BUILTIN: raise P.ProjectError('a built-in preset cannot be deleted')
    p = os.path.join(_preset_dir(), _safe(name) + '.json')
    if not os.path.exists(p): raise P.ProjectError(f'no pipeline preset {name!r}')
    os.remove(p)


def resolve(preset=None, options=None, stages=None):
    """The normalised stages to run: `stages` if given, else the preset (default 'best') with the flat `options` laid over it."""
    if stages is not None: return normalize(stages)
    return apply_options(get_preset(preset or DEFAULT_PRESET)['stages'], options)


def stage_key(iid, variant_id, stages):
    """A short identity of what is made: the item, the source variant, and every stage's settings and version."""
    return hashlib.sha1(json.dumps([iid, variant_id, [({k: v for k, v in s.items() if not (k in ('match', 'exposure') and v == 'auto')}, version(s['stage'])) for s in normalize(stages)]], sort_keys=True).encode()).hexdigest()[:10]


def summary(stages):
    """One line for a person: 'JPEG, auto quality · 6016 px auto · levelled · sharpened 1.5' (from the stage settings)."""
    d = {s['stage']: s for s in stages}; e = d['encode']; parts = ['TIFF 16 bit' if e['format'] == 'tiff' else f"JPEG {'auto quality' if e['quality'] == 'auto' else 'q' + str(e['quality'])}"]
    r = d.get('resize'); parts.append('size ' + ({'auto': 'auto', 'native': 'native'}.get(r['size'], r['size'] + ' px') if r else 'native'))
    if 'level' in d: parts.append('levelled')
    if 'sharpen' in d: parts.append(f"sharpened {d['sharpen']['amount']:g}")
    if d['develop']['highlight_contrast'] != develop.HIGHLIGHT_CONTRAST: parts.append(f"highlight contrast {d['develop']['highlight_contrast']:g}")
    ex = d['develop'].get('exposure', 'auto')
    if ex != 'auto': parts.append('camera exposure' if ex == 'camera' else f'camera exposure {ex:+g} EV')
    return ' · '.join(parts)


# ---- running ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def source_variant(item, variant=None):
    """The variant to make a picture from: the one asked for, else the item's primary, else (if the primary is itself a generated one) the best source file. Never a generated variant."""
    vs = {v['id']: v for v in item['variants']}
    if variant is not None:
        if variant not in vs: raise P.ProjectError(f"item {item['id']!r} has no variant {variant!r}")
        v = vs[variant]
    else:
        v = vs.get(item['primary'])
        if v is None or v['kind'] == 'generated': v = next((x for x in item['variants'] if x['kind'] != 'generated'), None)
    if v is None or v['kind'] == 'generated': raise P.ProjectError('there is no source picture to generate from')
    return v


def find_existing(pid, iid, variant=None, stages=None, preset=None, options=None):
    """(key, existing generated variant or None) for these settings: what the app asks before it offers Generate."""
    st = resolve(preset, options, stages); item = P.get_item(pid, iid); v = source_variant(item, variant); key = stage_key(iid, v['id'], st)
    return key, next((g for g in item['variants'] if g['kind'] == 'generated' and g.get('key') == key), None)


def generate(pid, iid, variant=None, preset=None, options=None, stages=None, progress=None):
    """Run a pipeline on a source variant and return the record of the new generated variant (`key`, `file`, `stages`, ...; `existing` True if these exact settings had been generated before, in which case nothing is done). `progress(text)` is called as the steps go."""
    say = progress or (lambda t: None); st = resolve(preset, options, stages)
    item = P.get_item(pid, iid); v = source_variant(item, variant); key = stage_key(iid, v['id'], st); d = P.generated_dir(pid, iid); base = os.path.join(d, key)
    if any(g['kind'] == 'generated' and g.get('key') == key for g in item['variants']):
        try: r = json.load(open(base + '.json')); r['existing'] = True; return r
        except (OSError, ValueError): pass
    state = State(pid, iid, item, v, say); runners = dict(develop=run_develop, level=run_level, resize=run_resize, sharpen=run_sharpen)
    for s in st:
        p = {k: val for k, val in s.items() if k != 'stage'}
        if s['stage'] == 'encode': os.makedirs(d, exist_ok=True); run_encode(state, p, base)
        else: runners[s['stage']](state, p)
    rec = dict(state.rec, version=1, key=key, created=dt.datetime.now().astimezone().isoformat(timespec='seconds'), item=iid, stem=item['stem'], source_variant=v['id'], source=v['path'], preset=preset, stages=st, summary=summary(st),
               output=[int(state.rgb.shape[1]), int(state.rgb.shape[0])], file=state.file, bytes=os.path.getsize(state.file), format=st[-1]['format'], quality=state.quality, existing=False)
    with open(base + '.json', 'w') as f: json.dump(rec, f, indent=1)
    return rec


# ---- rules: apply presets to source folders ---------------------------------------------------------------------------------------------------------------------------------------------------------------

def rules(pid): return P.load(pid).get('auto', [])


def set_rules(pid, new):
    """Save the project's auto-apply rules: [{source: <source id or '*'>, pipeline: <preset name>, variant: <optional source variant id>}] (each checked)."""
    clean = []
    for r in new or []:
        get_preset(r.get('pipeline')); src = r.get('source', '*')
        if src != '*': P._source(pid, src)
        clean.append({k: v for k, v in dict(source=src, pipeline=r['pipeline'], variant=r.get('variant') or None).items() if v is not None})
    doc = P.load(pid); doc['auto'] = clean; P._save(doc); return clean


def pending(pid):
    """What applying the rules would generate: [(item id, source variant id, preset name)] for every still and panorama of the sources a rule names whose settings have no generated variant yet."""
    out = []
    for r in rules(pid):
        stages = get_preset(r['pipeline'])['stages']
        for s in P.source_summaries(pid):
            if r['source'] not in ('*', s['id']) or not s['reachable']: continue
            for it in P.source_items(pid, s['id']):
                if it['kind'] == 'video': continue
                try: v = source_variant(it, r.get('variant'))
                except P.ProjectError: continue
                key = stage_key(it['id'], v['id'], stages)
                if not any(g['kind'] == 'generated' and g.get('key') == key for g in it['variants']): out.append((it['id'], v['id'], r['pipeline']))
    return out


def apply_rules(pid, progress=None, stop=lambda: False, workers=None):
    """Generate everything the rules ask for and is missing, several pictures at once; returns {done, failed: [(item, message)], total, workers}. `progress(done, total, text)` is called as pictures finish and as the running ones move on; `stop()` is polled before each picture is started (those already running finish).
    `workers`: how many at once (default resources.auto_workers: 2 with a GPU, where the device slots (gpu.slots) decide how many are on the card and the CPU parts of the others (decode, encode, files, the network share) overlap its work; fewer without one). One picture is made by one thread and distinct pictures write distinct files, so nothing is shared."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from stratastills import gpu, resources
    todo = pending(pid); total = len(todo); done = [0]; failed = []; lock = threading.Lock(); n = max(1, workers or resources.auto_workers(gpu.available()))

    def work(iid, vid, name):
        if stop(): return
        short = iid.split('~')[-1]
        try: generate(pid, iid, vid, name, progress=lambda t: progress and progress(done[0] + len(failed), total, f'{short}: {t}'))
        except Exception as e:
            with lock: failed.append((iid, str(e)))
        else:
            with lock: done[0] += 1
        if progress: progress(done[0] + len(failed), total, f'{short} done')
    with resources.batch(n), (gpu.batch(n) if gpu.available() else contextlib.nullcontext()), ThreadPoolExecutor(n) as ex: list(ex.map(lambda a: work(*a), todo))
    return dict(done=done[0], failed=failed, total=total, workers=n)
