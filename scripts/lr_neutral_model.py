"""Learn the person's edit FROM OUR NEUTRAL picture (scripts/lr_neutral_thumbs.py) instead of from the camera's preview: the automatic look is applied to our neutral, so the transform it predicts should be one that turns our neutral into their final.
  python scripts/lr_neutral_model.py fit              per pair, the 19-parameter transform (scripts/lr_fit.py) that takes our neutral to the final -> candidates/personal/nfit.npz (the ceiling)
  python scripts/lr_neutral_model.py eval [--epochs N] [--no-cnn] [--cam]
      on the held-out shoots (lr_look.test_shoot), mean |dL| / colour distance to the final for: our neutral as it is; the current camera-trained transform model applied to our neutral (what auto2/auto3 do); XGBoost on the same features trained on the neutral fits; a small network (ImageNet ResNet-18 on the neutral thumb, plus the scalar features) trained end to end on the distance after the transform; the per-pair fit. Also over the hard cases (candidates/personal/hard/hard_cases.json) and the old hard mask.
PRIVATE: the person's own pictures. Needs `uv sync --group research` and torchvision."""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import lr_dataset as D
import lr_fit as FIT
import lr_look as K
import lr_tmodel as TM

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_neutral():
    """neu, fin (N, T, T, 3 uint8 BGR) and idx (catalogue rows) of every finished batch."""
    z = [np.load(f) for f in sorted(glob.glob(os.path.join(D.OUT, "nthumbs", "t*.npz")))]
    return (
        np.concatenate([x["neu"] for x in z]),
        np.concatenate([x["fin"] for x in z]),
        np.concatenate([x["idx"] for x in z]),
    )


def fit(a):
    neu, fin, idx = load_neutral()
    n = len(idx)
    P = np.zeros((n, FIT.NP), np.float32)
    before = np.zeros((n, 2), np.float32)
    after = np.zeros((n, 2), np.float32)
    for s in range(0, n, 128):
        c, f = FIT.to_lab(neu[s : s + 128]).to(DEV), FIT.to_lab(fin[s : s + 128]).to(DEV)
        p = FIT.fit_batch(c, f, a.steps)
        with torch.no_grad():
            b0, a0 = FIT.distance(c, f), FIT.distance(FIT.transform(c, p), f)
        P[s : s + len(p)] = p.cpu().numpy()
        before[s : s + len(p)] = torch.stack(b0, 1).cpu().numpy()
        after[s : s + len(p)] = torch.stack(a0, 1).cpu().numpy()
    np.savez(os.path.join(D.OUT, "nfit.npz"), params=P, before=before, after=after, idx=idx, names=np.array(FIT.NAMES))
    print(
        f"{n} pairs: neutral as it is |dL| {before[:, 0].mean():.2f}, colour {before[:, 1].mean():.2f} -> after the fit {after[:, 0].mean():.2f} / {after[:, 1].mean():.2f}"
    )


def dist(src, fin, p):
    """Per-pair (|dL|, colour) of `src` after the transform `p` (None: as it is) against `fin` (uint8 BGR thumbs)."""
    out = []
    for s in range(0, len(src), 256):
        c, f = FIT.to_lab(src[s : s + 256]).to(DEV), FIT.to_lab(fin[s : s + 256]).to(DEV)
        with torch.no_grad():
            o = c if p is None else FIT.transform(c, torch.from_numpy(np.asarray(p[s : s + 256], np.float32)).to(DEV))
        out.append(torch.stack(FIT.distance(o, f), 1).cpu().numpy())
    return np.concatenate(out)


class Net(torch.nn.Module):
    """ResNet-18 (ImageNet weights) on the thumb(s), its pooled features joined with the standardised scalar features, two layers to the 19 parameters, started at the person's typical transform."""

    def __init__(self, n_scalar, typical, n_img=1):
        super().__init__()
        import torchvision

        self.cnn = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        w = self.cnn.conv1.weight.data
        if n_img > 1:
            self.cnn.conv1 = torch.nn.Conv2d(3 * n_img, 64, 7, 2, 3, bias=False)
            self.cnn.conv1.weight.data = torch.cat([w / n_img] * n_img, 1)
        self.cnn.fc = torch.nn.Identity()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(512 + n_scalar, 256), torch.nn.GELU(), torch.nn.Dropout(0.2), torch.nn.Linear(256, FIT.NP)
        )
        torch.nn.init.zeros_(self.head[-1].weight)
        self.head[-1].bias.data = torch.as_tensor(typical, dtype=torch.float32)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406] * n_img).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225] * n_img).view(1, -1, 1, 1))

    def forward(self, img, x):  # img: (N, 3*n_img, H, W) RGB 0..1
        return self.head(torch.cat([self.cnn((img - self.mean) / self.std), x], 1))


def train_cnn(imgs, lab_src, lab_fin, xs, w, typical, epochs, log=print):
    """End to end: the loss is the distance after applying the predicted transform to our neutral (smooth L1 on blurred CIELAB, as the per-pair fit), weighted per pair."""
    net = Net(xs.shape[1], typical, imgs.shape[1] // 3).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    n = len(imgs)
    bs = 128
    steps = epochs * (n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
    k = 5
    import cv2

    ker = cv2.getGaussianKernel(k, 1.2).astype(np.float32)
    Kt = torch.from_numpy(ker @ ker.T).to(DEV).expand(3, 1, k, k)
    bl = lambda t: torch.nn.functional.conv2d(torch.nn.functional.pad(t, (2,) * 4, mode="replicate"), Kt, groups=3)
    wt = torch.from_numpy(w.astype(np.float32))
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for s in range(0, n - bs + 1, bs):
            b = perm[s : s + bs]
            img = imgs[b].to(DEV, non_blocking=True).float() / 255
            if torch.rand(1).item() < 0.5:
                img = img.flip(3)
                src, fin = lab_src[b].to(DEV).flip(3), lab_fin[b].to(DEV).flip(3)  # a mirrored picture keeps its edit
            else:
                src, fin = lab_src[b].to(DEV), lab_fin[b].to(DEV)
            p = net(img, xs[b].to(DEV))
            l = torch.nn.functional.smooth_l1_loss(bl(FIT.transform(src, p)), bl(fin), beta=2.0, reduction="none").mean(
                (1, 2, 3)
            )
            loss = (l * wt[b].to(DEV)).sum() / wt[b].sum().to(DEV)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(b)
        log(f"  epoch {ep + 1}/{epochs}: loss {tot / n:.3f} ({time.time() - t0:.0f} s)")
    return net


def predict_cnn(net, imgs, xs):
    net.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(imgs), 256):
            out.append(net(imgs[s : s + 256].to(DEV).float() / 255, xs[s : s + 256].to(DEV)).cpu().numpy())
    return np.concatenate(out)


def evaluate(a):
    neu, fin, nidx = load_neutral()
    nf = np.load(os.path.join(D.OUT, "nfit.npz"))
    assert (nf["idx"] == nidx).all(), "nfit.npz is older than the neutral thumbs: run fit again"
    rows, X, names, Y, idx, G = K.load()
    xpos = {int(i): r for r, i in enumerate(idx)}
    keep = np.array([k for k, i in enumerate(nidx) if int(i) in xpos])
    neu, fin, nidx, NP_ = neu[keep], fin[keep], nidx[keep], nf["params"][keep]
    xr = np.array([xpos[int(i)] for i in nidx])
    Xn, Gn = X[xr], G[xr]
    test = np.array([K.test_shoot(g, a.share) for g in Gn])
    hard = TM.hard_mask(
        Xn, names
    )  # the same held-out shoots as every other report (share 0.1); a bigger share only to try the script on a few batches
    cam = {}
    for f in sorted(glob.glob(os.path.join(D.OUT, "thumbs", "t*.npz"))):
        z = np.load(f)
        cam.update({int(i): c for i, c in zip(z["idx"], z["cam"])})
    print(
        f"{len(nidx)} pairs with our neutral, {test.sum()} held out ({(hard & test).sum()} in the old hard mask); training on {(~test).sum()}",
        flush=True,
    )
    sets = {"all held out": test, "old hard mask": test & hard}
    hf = os.path.join(D.OUT, "hard", "hard_cases.json")
    if os.path.exists(hf):  # the hard-case sets need the photo ids of the shared database; the main comparison does not
        import psycopg

        rawid = dict(
            psycopg.connect("dbname=omarchy", options="-csearch_path=photos,public")
            .execute("select relpath, photo_id from file where role='raw'")
            .fetchall()
        )
        pid = np.array([rawid.get(os.path.relpath(rows[int(i)]["raw"], os.path.expanduser("~/raw")), -1) for i in nidx])
        H = json.load(open(hf))
        sets["hard (151)"] = np.isin(pid, [h["photo_id"] for h in H["hard"]])
        sets["core (40)"] = np.isin(pid, [h["photo_id"] for h in H["hard"] if h["core"]])
        sets["control (30)"] = np.isin(pid, [h["photo_id"] for h in H["control"]])
    ti = np.where(test)[0]
    res = {}
    res["neutral as it is"] = dist(neu[ti], fin[ti], None)
    if a.share == 0.1:
        res["camera-trained model on neutral (now)"] = dist(
            neu[ti], fin[ti], TM.predict(TM.train_cached(), Xn[ti])
        )  # that model is trained with the default 0.1 held-out share: with another share some of these shoots trained it
    typical = np.median(NP_[~test], 0)
    res["typical neutral edit"] = dist(neu[ti], fin[ti], np.tile(typical, (len(ti), 1)))
    w = np.where(hard, 3.0, 1.0)
    t0 = time.time()
    xm = TM.train(Xn[~test], NP_[~test], w[~test])
    res["XGBoost on neutral fits"] = dist(neu[ti], fin[ti], TM.predict(xm, Xn[ti]))
    print(f"XGBoost trained in {time.time() - t0:.0f} s", flush=True)
    if not a.no_cnn:
        mu, sd = np.nanmean(Xn[~test], 0), np.nanstd(Xn[~test], 0) + 1e-6
        xs = torch.from_numpy(np.nan_to_num((Xn - mu) / sd).clip(-5, 5).astype(np.float32))
        lab_n, lab_f = (
            torch.cat([FIT.to_lab(neu[s : s + 2000]) for s in range(0, len(neu), 2000)]),
            torch.cat([FIT.to_lab(fin[s : s + 2000]) for s in range(0, len(fin), 2000)]),
        )
        for tag, use_cam in (("", False), (" + camera preview", True)) if a.cam else (("", False),):
            im = neu[..., ::-1]
            if use_cam:
                im = np.concatenate(
                    [im, np.stack([cam.get(int(i), neu[k])[..., ::-1] for k, i in enumerate(nidx)])], -1
                )
            imgs = torch.from_numpy(np.ascontiguousarray(im)).permute(0, 3, 1, 2).contiguous()
            tr = np.where(~test)[0]
            net = train_cnn(imgs[tr], lab_n[tr], lab_f[tr], xs[tr], w[tr], typical, a.epochs)
            res[f"network on neutral{tag}"] = dist(neu[ti], fin[ti], predict_cnn(net, imgs[ti], xs[ti]))
            if a.save and not use_cam:
                torch.save(
                    dict(state=net.state_dict(), mu=mu, sd=sd, names=names, typical=typical),
                    os.path.join(D.ROOT, "out", "cache", "lr-neutral-net.pt"),
                )
    res["fit per pair (ceiling)"] = dist(neu[ti], fin[ti], NP_[ti])
    cols = {k: m[ti] for k, m in sets.items()}
    wd = max(len(k) for k in res) + 2
    print("mean |dL| / colour distance to the final, 128 px, our neutral as the starting picture")
    print(" " * wd + "".join(f"{k:>18s}" for k in cols))
    print("n".ljust(wd) + "".join(f"{int(m.sum()):>18d}" for m in cols.values()))
    for nm, d in res.items():
        print(
            nm.ljust(wd)
            + "".join(
                f"{d[m, 0].mean():>8.2f} /{d[m, 1].mean():>7.2f} " if m.any() else " " * 18 for m in cols.values()
            )
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["fit", "eval"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--no-cnn", action="store_true")
    ap.add_argument("--cam", action="store_true", help="also a network that sees the camera preview beside our neutral")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--share", type=float, default=0.1)
    a = ap.parse_args()
    {"fit": fit, "eval": evaluate}[a.step](a)


NET_FILE = os.path.join(D.ROOT, "out", "cache", "lr-neutral-net.pt")
_net = {}
import threading

_net_lock = threading.Lock()  # build_arena makes versions in several threads: loading and running the network one at a time (two threads at once gave flat, NaN-filled pictures)


def predict_params(neu_bgr, x):
    """The 19 transform parameters the saved network (`eval --save`) predicts for our neutral picture `neu_bgr` (uint8 BGR, laid on the final's crop like the training thumbs) with the feature row `x` (lr_look.load order). The network is loaded once per process."""

    with _net_lock:
        return _predict_params(neu_bgr, x)


def _predict_params(neu_bgr, x):
    import cv2

    if "net" not in _net:
        z = torch.load(NET_FILE, map_location=DEV, weights_only=False)
        net = Net(len(z["mu"]), z["typical"]).to(DEV)
        net.load_state_dict(z["state"])
        net.eval()
        _net.update(net=net, mu=z["mu"], sd=z["sd"])
    t = cv2.resize(neu_bgr, (D.THUMB, D.THUMB), interpolation=cv2.INTER_AREA)[..., ::-1].copy()
    img = torch.from_numpy(t).permute(2, 0, 1)[None].float().to(DEV) / 255
    xs = torch.from_numpy(
        np.nan_to_num((np.asarray(x, np.float64) - _net["mu"]) / _net["sd"]).clip(-5, 5).astype(np.float32)
    )[None].to(DEV)
    with torch.no_grad():
        out = _net["net"](img, xs)[0].cpu().numpy()
    if "lo" not in _net:
        q = _train_fits()
        _net.update(lo=np.percentile(q, 0.5, 0), hi=np.percentile(q, 99.5, 0))
    return np.clip(
        out, _net["lo"], _net["hi"]
    )  # an unusual picture (features far outside training) can get a transform no edit ever had (cast -21, hue 26 degrees, a near-white curve): each parameter is held to the range the per-pair fits span (0.5-99.5 %)


def typical_params():
    """The median per-pair neutral fit: the person's typical edit, the fallback when the network's transform would flatten a picture."""
    if "typical" not in _net:
        _net["typical"] = np.median(_train_fits(), 0).astype(np.float32)
    return _net["typical"]


def _train_fits():
    """The per-pair neutral fits of the TRAINING shoots only (lr_look.test_shoot False): the range and median used at inference must not come from the held-out finals the arena judges against."""
    z = np.load(os.path.join(D.OUT, "nfit.npz"))
    rows = D.load_catalogue()
    train = np.array([not K.test_shoot(os.path.dirname(rows[int(i)]["final"])) for i in z["idx"]])
    return z["params"][train]
