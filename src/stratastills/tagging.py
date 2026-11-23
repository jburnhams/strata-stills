"""Vision tags for the archive through strata-models: two short calls per photo, chosen by the photo-presets evals (docs repo
`tech/code/strata-models/evals/2026-11-14-photo-presets.md`), in place of the Claude Sonnet sub-agents of the 2026-11 pilot (schema-v1):

  role photo-fields   (local Qwen3.5-9B, preset photo-fields-v2)  setting, people, light, sharpness, skin, text present, nudity, quality
  role photo-caption  (Claude Haiku,     preset photo-caption-v1) caption, description, search tags, legible text, place, event

Both see the camera's own rendering at 1024 px (the JPEG beside the raw when the camera wrote one, else the preview inside the raw), with the
capture metadata as context, as the pilot did. A photo the nudity filter flags is never sent (the caption call leaves this machine).

Each photo's result is one `photos.analysis` row: kind `vision_tags`, producer `strata-models`, producer_version VERSION, input_key the first
16 hex of the SHA-256 of the preview, label the caption, score the aesthetic rating, data the two answers merged (fields flat, as the preset
answers them) with the models that made them under `_models`."""

import hashlib
import json
import os
import time

import cv2

from stratastills import images

KIND, PRODUCER, VERSION = "vision_tags", "strata-models", "fields-v2+caption-v1"
CALLS = (("photo-fields", "photo-fields-v2", 600), ("photo-caption", "photo-caption-v1", 1500))
PX = 1024


class TagError(RuntimeError):
    """A photo that could not be tagged (no picture to show, a call that failed); the run goes on with the next."""


def camera_picture(raw, px=PX):
    """The camera's own rendering of a raw, upright, long side <= px (BGR): the JPEG beside it when there is one, else the raw's embedded
    preview; None if neither."""
    for ext in (".JPG", ".jpg", ".JPEG", ".jpeg"):
        j = os.path.splitext(raw)[0] + ext
        if os.path.exists(j):
            return images.load(j, px)
    return images.preview_bgr(raw, px)


def preview(raw, folder, px=PX):
    """A JPEG of the camera's rendering at `px` on the long side, cached in `folder` by the raw's path, size and time: (path, input_key)."""
    os.makedirs(folder, exist_ok=True)
    out = os.path.join(folder, images.cache_key(raw, px) + ".jpg")
    if not os.path.exists(out):
        im = camera_picture(raw, px)
        if im is None:
            raise TagError(f"no camera picture for {raw}")
        tmp = out + ".tmp.jpg"
        cv2.imwrite(tmp, im, [cv2.IMWRITE_JPEG_QUALITY, 85])
        os.replace(tmp, out)
    with open(out, "rb") as f:
        return out, hashlib.sha256(f.read()).hexdigest()[:16]


def context(meta):
    """The capture metadata the models see (as the pilot gave the sub-agents), from a photo_meta row's values."""
    if not meta:
        return {}
    gps = meta.get("gps_lat"), meta.get("gps_lon")
    return dict(
        taken=str(meta.get("taken_at")) if meta.get("taken_at") else None,
        camera=meta.get("model"),
        lens=meta.get("lens"),
        iso=meta.get("iso"),
        shutter_s=meta.get("exposure_s"),
        f=meta.get("f_number"),
        focal_mm=meta.get("focal_mm"),
        gps=None if gps[0] is None or gps[1] is None else [round(gps[0], 4), round(gps[1], 4)],
    )


def retry(sm, fn, wait_s=30, give_up_s=3600, sleep=time.sleep):
    """fn() again while strata-models is unreachable (a restart for a deploy takes a minute or two); gives up after `give_up_s`."""
    waited = 0
    while True:
        try:
            return fn()
        except sm.Unavailable:
            if waited >= give_up_s:
                raise
            sleep(wait_s)
            waited += wait_s


def submit(sm, image, ctx, app, priority="batch"):
    """Queue both calls for one picture; [(role, job id)]."""
    return [
        (
            role,
            retry(
                sm,
                lambda r=role, p=preset, m=mt: sm.submit(
                    r, dict(image=image, preset=p, context=ctx, max_tokens=m), app, priority
                ),
            )["id"],
        )
        for role, preset, mt in CALLS
    ]


def collect(sm, jobs):
    """Wait for a photo's calls and merge their answers into one record; TagError if any failed or answered without JSON."""
    out, models = {}, {}
    for role, jid in jobs:
        j = retry(sm, lambda j=jid: sm.wait_job(j))  # a queued job survives a restart of the service
        r = j.get("result") or {}
        if j.get("status") != "done" or not isinstance(r.get("json"), dict):
            raise TagError(f"{role}: {j.get('error') or j.get('status')}")
        out.update(r["json"])
        models[role] = r.get("model")
    out["_models"] = models
    return out


def row(photo_id, key, rec):
    """The analysis row's values for a tagged photo."""
    q = rec.get("aesthetic")
    return dict(
        photo_id=photo_id,
        kind=KIND,
        producer=PRODUCER,
        producer_version=VERSION,
        input_key=key,
        label=(rec.get("caption") or "")[:500] or None,
        score=float(q) if isinstance(q, (int, float)) and not isinstance(q, bool) else None,
        data=json.loads(json.dumps(rec)),
    )
