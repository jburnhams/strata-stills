"""Scene features for the look model: CLIP (ViT-B/32, laion2b weights from the Hugging Face cache) on the camera previews of candidates/personal/thumbs (scripts/lr_dataset.py thumbs), on the GPU.
  python scripts/lr_clip.py     -> candidates/personal/clip.npz: `idx` (catalogue rows), `pca` (32 principal components of the embedding), `labels` (similarity to a list of scene prompts, softmax over the prompts of each group) and the names of both.
The prompts are the zero-shot scene labels (indoor or outdoor, what is in the picture, what the light is like); the principal components carry the rest of what CLIP sees.
The embeddings come from strata-models (the `image-embed` role, model clip-vit-b-32, the same weights, the previews squeezed to 224 px as here) when it is running, so
the GPU work goes through its queue and the model is shared with the other apps; otherwise open_clip is loaded here as before (STRATASTILLS_LOCAL_CLIP=1 forces that)."""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import lr_dataset as D

GROUPS = {
    'where': ['an indoor photo', 'an outdoor photo'],
    'subject': ['a portrait of one person', 'a group of people', 'a child', 'a landscape', 'a city street', 'a building', 'food on a table', 'an animal', 'a flower or plant', 'a vehicle', 'a close-up detail', 'a sports or action photo'],
    'light': ['a photo in bright sunlight', 'a photo on an overcast day', 'a photo in warm tungsten indoor light', 'a photo lit by a flash', 'a photo at night', 'a photo at sunset or sunrise', 'a photo in shade', 'a photo lit by candles or fairy lights'],
    'season': ['a snowy scene', 'a beach or sea', 'a christmas scene with decorations', 'a birthday party', 'a wedding', 'a holiday abroad'],
}


def embed_with_service():
    """(image embeddings in thumbs order, {group: text embeddings}, catalogue idx) from strata-models, or None when it is not running."""
    if os.environ.get('STRATASTILLS_LOCAL_CLIP') == '1': return None
    try: import strata_models_client as sm
    except ImportError: return None
    if not sm.available(): return None
    import tempfile
    E, IDX = [], []
    with tempfile.TemporaryDirectory(prefix='lr_clip_') as tmp:
        for f in sorted(glob.glob(os.path.join(D.OUT, 'thumbs', 't*.npz'))):
            out = os.path.join(tmp, 'e.npz')
            sm.run('image-embed', dict(array=dict(path=os.path.abspath(f), key='cam', bgr=True), resize='square', out=out), app='strata-stills', model='clip-vit-b-32')
            E.append(np.load(out)['images']); IDX.append(np.load(f)['idx']); print(f, flush=True)
    text = {g: np.array(sm.run('image-embed', dict(texts=ps), app='strata-stills', model='clip-vit-b-32')['texts'], dtype=np.float32) for g, ps in GROUPS.items()}
    return np.concatenate(E), text, np.concatenate(IDX)


def write(E, text, idx):
    """clip.npz from the embeddings: 32 principal components and the softmax scene labels."""
    mu = E.mean(0); U, S, Vt = np.linalg.svd(E[::4] - mu, full_matrices=False); pca = (E - mu) @ Vt[:32].T
    cols, names = [], []
    for g, ps in GROUPS.items():
        sim = 100 * E @ text[g].T; p = np.exp(sim - sim.max(1, keepdims=True)); p /= p.sum(1, keepdims=True); cols.append(p); names += [f'{g}:{q}' for q in ps]
    np.savez(os.path.join(D.OUT, 'clip.npz'), idx=idx, pca=pca.astype(np.float32), labels=np.hstack(cols).astype(np.float32), pca_names=np.array([f'pc{i}' for i in range(32)]), label_names=np.array(names))
    print('done', E.shape)


def main():
    got = embed_with_service()
    if got is not None: write(*got); return
    import open_clip, torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; model, _, pre = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k'); model = model.to(dev).eval().half() if dev == 'cuda' else model.eval(); tok = open_clip.get_tokenizer('ViT-B-32')
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)
    with torch.no_grad():
        text = {g: torch.nn.functional.normalize(model.encode_text(tok(ps).to(dev)).float(), dim=-1) for g, ps in GROUPS.items()}
    E, IDX = [], []
    for f in sorted(glob.glob(os.path.join(D.OUT, 'thumbs', 't*.npz'))):
        z = np.load(f); cam = torch.from_numpy(z['cam'][..., ::-1].copy()).permute(0, 3, 1, 2).float().div(255)                               # BGR -> RGB
        for s in range(0, len(cam), 128):
            x = torch.nn.functional.interpolate(cam[s:s + 128].to(dev), size=224, mode='bicubic', align_corners=False).clamp(0, 1); x = ((x - mean) / std).to(next(model.parameters()).dtype)
            with torch.no_grad(): E.append(torch.nn.functional.normalize(model.encode_image(x).float(), dim=-1).cpu().numpy())
        IDX.append(z['idx']); print(f, flush=True)
    write(np.concatenate(E), {g: t.cpu().numpy() for g, t in text.items()}, np.concatenate(IDX))


if __name__ == '__main__':
    main()
