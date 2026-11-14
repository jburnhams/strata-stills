"""Freeze the learned look into a bundle the app can run on any picture (stratastills.look, the pipeline's `look` stage and the arena).
  python scripts/export_look_bundle.py [--out DIR] [--meta-records FILE] [--clip FILE]
-> DIR (default settings.look_dir()): manifest.json (feature names, how each block was made, the files' hashes), look.pkl, tmodel.pkl,
   ftarget.pkl, skin.pkl, neutral-net.pt, neutral-range.npz, meta-encoder.json, clip-basis.npz, clip-prompts.json.

The models are the ones the arena uses: trained on the shoots NOT held out (lr_look.test_shoot), so the arena's held-out pictures stay
unseen. Each block of the feature row is rebuilt here and CHECKED against the dataset before anything is written:
- metadata: the encoder (columns kept, text values ranked by frequency) from the catalogue's exiftool records (`--meta-records`, a JSON
  list in catalogue order as lr_dataset._meta_batch gives them; read with exiftool when not given: slow on the archive disk), then every
  row encoded and compared with meta.npz;
- CLIP: the principal components and prompt embeddings from the catalogue's image embeddings (`--clip`, an npz with E, idx and text_<group>
  as lr_clip.embed_with_service gives them; made through strata-models when not given), compared with clip.npz.
PRIVATE: trained on the person's own pictures. The bundle never goes into the repository."""

import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import lr_dataset as D
import lr_face_first as FF
import lr_ftarget as FT
import lr_look as LK
import lr_neutral_model as NM
import lr_tmodel as TM

from stratastills import settings
from stratastills.look import features as F


def meta_encoder(records):
    """The metadata encoder, checked: every catalogue row encoded with it must give meta.npz's row."""
    z = np.load(os.path.join(D.OUT, "meta.npz"))
    enc = F.meta_encoder(records)
    want = [str(n) for n in z["names"]]
    if enc["names"] != want:
        raise SystemExit(f"metadata columns differ from meta.npz: {set(enc['names']) ^ set(want)}")
    M = np.stack([F.meta_row(e, enc) for e in records])
    bad = ~np.isclose(np.nan_to_num(M, nan=-9e9), np.nan_to_num(z["M"], nan=-9e9), rtol=1e-5, atol=1e-4)
    if bad.any():
        cols = sorted({want[j] for j in np.flatnonzero(bad.any(0))})
        raise SystemExit(f"{bad.sum()} metadata values differ from meta.npz, in {cols}")
    print(
        f"metadata encoder: {len(want)} columns, {len(enc['ranks'])} of them text; all {len(records)} rows reproduced"
    )
    return enc


def clip_basis(clip_file):
    """mu, vt and the prompt embeddings, checked: the catalogue's own embeddings projected with them must give clip.npz."""
    import lr_clip as C

    if clip_file and os.path.exists(clip_file):
        z = np.load(clip_file)
        E, idx, text = z["E"], z["idx"], {g: z[f"text_{g}"] for g in C.GROUPS}
    else:
        got = C.embed_with_service()
        if got is None:
            raise SystemExit("strata-models is not running: no CLIP embeddings (or pass --clip)")
        E, text, idx = got
    mu = E.mean(0)
    _, _, vt = np.linalg.svd(E[::4] - mu, full_matrices=False)
    vt = vt[:32]
    ref = np.load(os.path.join(D.OUT, "clip.npz"))
    pos = {int(i): k for k, i in enumerate(ref["idx"])}
    basis = dict(mu=mu, vt=vt, texts=[(g, ps, text[g]) for g, ps in C.GROUPS.items()])
    rows = np.stack([F.clip_row(e, basis) for e in E])
    want = np.stack([np.concatenate([ref["pca"][pos[int(i)]], ref["labels"][pos[int(i)]]]) for i in idx])
    err = np.abs(rows - want).max()
    if err > 1e-3:
        raise SystemExit(f"CLIP columns differ from clip.npz by up to {err:.4f}")
    print(
        f"CLIP basis: 32 components, {sum(len(p) for p in C.GROUPS.values())} prompts; all {len(idx)} rows reproduced (max diff {err:.1e})"
    )
    return mu, vt, text, C.GROUPS


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None)
    ap.add_argument("--meta-records")
    ap.add_argument("--clip")
    a = ap.parse_args()
    out = os.path.abspath(a.out or settings.look_dir())
    tmp = out + ".new"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)

    rows, X, names, Y, idx, G = LK.load()
    test = np.array([LK.test_shoot(g) for g in G])
    base = [str(n) for n in D.load_features()[1]]
    n_meta = len([n for n in names if n.startswith("meta:")])
    n_clip = len([n for n in names if n.startswith("clip:")])
    assert names == base + names[len(base) :] and len(names) == len(base) + n_meta + n_clip

    if a.meta_records:
        records = json.load(open(a.meta_records))
    else:
        cat = [r["raw"] for r in D.load_catalogue()]
        got = {}
        for k in range(0, len(cat), 400):
            got.update(D._meta_batch(cat[k : k + 400]))
        records = [got.get(r) for r in cat]
    enc = meta_encoder(records)
    mu, vt, text, groups = clip_basis(a.clip)

    import pickle

    pickle.dump(LK.train_cached(X[~test], Y[~test]), open(os.path.join(tmp, "look.pkl"), "wb"))
    pickle.dump(TM.train_cached(), open(os.path.join(tmp, "tmodel.pkl"), "wb"))
    pickle.dump(FT.train_cached(), open(os.path.join(tmp, "ftarget.pkl"), "wb"))
    if FF.models() is not None:
        pickle.dump(FF.models(), open(os.path.join(tmp, "skin.pkl"), "wb"))
    shutil.copy(NM.NET_FILE, os.path.join(tmp, "neutral-net.pt"))
    q = NM._train_fits()
    np.savez(
        os.path.join(tmp, "neutral-range.npz"),
        lo=np.percentile(q, 0.5, 0),
        hi=np.percentile(q, 99.5, 0),
        typical=np.median(q, 0).astype(np.float32),
    )
    json.dump(enc, open(os.path.join(tmp, "meta-encoder.json"), "w"))
    np.savez(os.path.join(tmp, "clip-basis.npz"), mu=mu, vt=vt, **{f"text_{g}": t for g, t in text.items()})
    json.dump([[g, ps] for g, ps in groups.items()], open(os.path.join(tmp, "clip-prompts.json"), "w"))
    files = sorted(os.listdir(tmp))
    json.dump(
        dict(
            names=names,
            base_names=base,
            n_meta=n_meta,
            n_clip=n_clip,
            trained_on=int((~test).sum()),
            files={f: sha(os.path.join(tmp, f)) for f in files},
        ),
        open(os.path.join(tmp, "manifest.json"), "w"),
        indent=1,
    )
    if os.path.exists(out):
        shutil.rmtree(out + ".old", ignore_errors=True)
        os.replace(out, out + ".old")
    os.replace(tmp, out)
    print(f"look bundle: {out} ({len(names)} features, trained on {(~test).sum()} pictures)")


if __name__ == "__main__":
    main()
