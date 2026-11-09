"""Do the vision tags (photos.analysis kind 'vision_tags', schema v1) help the neutral -> final edit model? Trains XGBoost and the network of scripts/lr_neutral_model.py on the TAGGED training pictures only, with and without tag features, and compares on the held-out shoots (all of which are tagged).
  python scripts/lr_tag_test.py [--epochs N]"""
import argparse, os, sys
import numpy as np, psycopg, torch
sys.path.insert(0, os.path.dirname(__file__))
import lr_dataset as D, lr_look as K, lr_neutral_model as NM, lr_tmodel as TM

LIGHT = ['daylight', 'overcast sky', 'shade', 'sunset sun', 'tungsten', 'halogen', 'fluorescent', 'LED white', 'LED coloured', 'stage lights', 'candle', 'fire', 'flash', 'screen', 'street light', 'mixed']
CAST = ['warm', 'orange', 'yellow', 'red', 'cool', 'blue', 'green', 'magenta', 'purple', 'pink', 'none']


def tag_vector(t):
    """A fixed-length numeric vector from one tag record (NaN where it says nothing)."""
    s, L, C, E = t.get('setting') or {}, t.get('lighting') or {}, t.get('colour') or {}, t.get('exposure') or {}
    v = [s.get('indoor_outdoor') == 'indoor', s.get('indoor_outdoor') == 'mixed', s.get('time_of_day') in ('night', 'dusk'), s.get('time_of_day') == 'golden hour']
    share = {x: 0.0 for x in LIGHT}
    for src in L.get('sources') or []:
        if src.get('type') in share:
            try: share[src['type']] += float(src.get('share') or 0.5)
            except (TypeError, ValueError): share[src['type']] += 0.5
    v += [share[x] for x in LIGHT]
    v += [bool(L.get('mixed')), {'very dark': 0, 'dark': 1, 'normal': 2, 'bright': 3, 'very bright': 4}.get(L.get('level'), np.nan), {'low': 0, 'medium': 1, 'high': 2}.get(L.get('contrast'), np.nan)]
    cast = str(C.get('cast_now') or '').lower(); v += [w in cast for w in CAST]
    v += [bool(C.get('skin_visible')), str(C.get('skin_looks') or '') in ('too orange', 'too red'), bool(E.get('intended_dark')), {'none': 0, 'some': 1, 'large': 2}.get(E.get('clipped_highlights'), np.nan), {'none': 0, 'some': 1, 'large': 2}.get(E.get('crushed_shadows'), np.nan), {'low': 0, 'medium': 1, 'high': 2}.get(E.get('noise'), np.nan)]
    return np.array(v, np.float32)


def main(a):
    neu, fin, nidx = NM.load_neutral(); nf = np.load(os.path.join(D.OUT, 'nfit.npz')); rows, X, names, Y, idx, G = K.load(); xpos = {int(i): r for r, i in enumerate(idx)}
    keep = np.array([k for k, i in enumerate(nidx) if int(i) in xpos]); neu, fin, nidx, P = neu[keep], fin[keep], nidx[keep], nf['params'][keep]; xr = np.array([xpos[int(i)] for i in nidx]); Xn, Gn = X[xr], G[xr]
    con = psycopg.connect('dbname=omarchy', options='-csearch_path=photos,public'); rawid = dict(con.execute("select relpath, photo_id from file where role='raw'").fetchall())
    tags = dict(con.execute("select photo_id, data from analysis where kind='vision_tags'").fetchall())
    pid = [rawid.get(os.path.relpath(rows[int(i)]['raw'], os.path.expanduser('~/raw'))) for i in nidx]; has = np.array([p in tags for p in pid])
    T = np.stack([tag_vector(tags[p]) if p in tags else np.full(len(tag_vector({})), np.nan, np.float32) for p in pid])
    test = np.array([K.test_shoot(g) for g in Gn]); tr = np.where(~test & has)[0]; ti = np.where(test)[0]; hard = TM.hard_mask(Xn, names)
    print(f'{len(tr)} tagged training pictures, {len(ti)} held out ({has[ti].mean():.0%} tagged), {T.shape[1]} tag features', flush=True)
    res = {}; w = np.where(hard, 3.0, 1.0)
    for tag_on in (False, True):
        F = np.hstack([Xn, T]) if tag_on else Xn; nm = ' + tags' if tag_on else ''
        m = TM.train(F[tr], P[tr], w[tr]); res[f'XGBoost{nm}'] = NM.dist(neu[ti], fin[ti], TM.predict(m, F[ti]))
        mu, sd = np.nanmean(F[tr], 0), np.nanstd(F[tr], 0) + 1e-6; xs = torch.from_numpy(np.nan_to_num((F - mu) / sd).clip(-5, 5).astype(np.float32))
        imgs = torch.from_numpy(np.ascontiguousarray(neu[..., ::-1])).permute(0, 3, 1, 2).contiguous()
        lab = lambda arr: torch.cat([NM.FIT.to_lab(arr[s:s + 2000]) for s in range(0, len(arr), 2000)])
        ln, lf = lab(neu[tr]), lab(fin[tr]); net = NM.train_cnn(imgs[tr], ln, lf, xs[tr], w[tr], np.median(P[tr], 0), a.epochs)
        res[f'network{nm}'] = NM.dist(neu[ti], fin[ti], NM.predict_cnn(net, imgs[ti], xs[ti])); print(f'done{nm}', flush=True)
    import json
    H = json.load(open(os.path.join(D.OUT, 'hard', 'hard_cases.json'))); sets = {'all held out': np.ones(len(ti), bool), 'old hard mask': hard[ti], 'hard (151)': np.isin([pid[t] for t in ti], [h['photo_id'] for h in H['hard']]), 'core (40)': np.isin([pid[t] for t in ti], [h['photo_id'] for h in H['hard'] if h['core']])}
    print('trained on the tagged training pictures only; mean |dL| / colour distance to the final'); print(' ' * 22 + ''.join(f'{k:>18s}' for k in sets))
    for nm, d in res.items(): print(nm.ljust(22) + ''.join(f'{d[m, 0].mean():>8.2f} /{d[m, 1].mean():>7.2f} ' for m in sets.values()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); ap.add_argument('--epochs', type=int, default=25); main(ap.parse_args())
