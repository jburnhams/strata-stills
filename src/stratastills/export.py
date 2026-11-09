"""The building blocks of generating the best-quality final picture of an item (the stages themselves, their configs, presets and batch runs are in pipeline.py):
reading and developing the source at full size, measuring its real detail (cached in the project), choosing the export size, and writing the JPEG or 16 bit TIFF with the sRGB profile and the camera's EXIF (orientation reset).
Nothing is ever written into a source folder.
"""

import json
import os
import threading

import cv2
import numpy as np

from stratastills import detail, develop, images
from stratastills import projects as P

FORMATS = ("jpeg", "tiff")


def _is_camera_raw(path):
    return os.path.splitext(path)[1].lower() in (".dng", ".arw", ".cr2", ".cr3", ".nef", ".orf", ".raf", ".rw2")


def _read_full(path, analysis=None, **kw):
    """(float32 RGB 0..1 at full size, how it was made). A raw that is not a plain camera DNG (a Lightroom linear DNG) is read at its largest embedded level."""
    if _is_camera_raw(path):
        try:
            return develop.develop(path, half=False, analysis=analysis, **kw), "developed from the raw"
        except Exception as e:
            try:
                return images.load(path, 100000)[:, :, ::-1].astype(
                    np.float32
                ) / 255.0, f"embedded level of the DNG ({str(e)[:60]})"
            except Exception as e2:
                raise images.ImageError(str(e2))
    return images.load(path, 100000)[:, :, ::-1].astype(np.float32) / 255.0, "read as it is (not a raw)"


def measure_detail(pid, path, luma=None):
    """The detail measurement of a file, from the cache of the project when there (it takes some seconds on a 48 MP raw). `luma`: the analysis luminance when the caller already has it (develop(analysis=)), so the raw is not decoded again."""
    p = images.cache_path(P.folder(pid), images.cache_key(path, "detail", develop.VERSION, detail.TILE), ".json")
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except ValueError:
            pass
    y = (
        luma
        if luma is not None
        else develop.analysis_luma(path)
        if _is_camera_raw(path)
        else cv2.cvtColor(images.load(path, 100000), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    )
    r = detail.effective_resolution(y)
    r["native"] = [int(y.shape[1]), int(y.shape[0])]
    r["recommended"] = detail.recommended_size(r, max(y.shape))
    with open(p, "w") as f:
        json.dump(r, f)
    return r


def choose_size(size, native_long, res):
    """The long side to export at, and why: `size` is 'auto', 'native' or a number of pixels."""
    if size in (None, "auto"):
        rec = res.get("recommended")
        return (
            (rec, "the detail limit measured on this picture")
            if rec
            else (native_long, "detail could not be measured (too little texture): native size")
        )
    if size == "native":
        return native_long, "native size, as asked"
    n = int(size)
    if n < 64:
        raise P.ProjectError("the size is too small")
    return min(n, native_long), ("as asked" if n <= native_long else "as asked, limited to native: nothing is invented")


def _icc():
    from PIL import ImageCms

    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _exif_of(item):
    """The EXIF block of the item's camera JPEG (orientation reset: the pixels are already upright), or None."""
    from PIL import Image

    for v in item["variants"]:
        if v["kind"] != "generated" and v["ext"] in (".jpg", ".jpeg"):
            try:
                with Image.open(v["path"]) as im:
                    ex = im.getexif()
                    ex[0x0112] = 1
                    return ex.tobytes()
            except Exception:
                return None
    return None


def _app_segments(exif, icc):
    """The JPEG application segments Pillow writes for `exif` (the Exif block, starting 'Exif\\0\\0') and `icc` (the profile, split into APP2 chunks of at most 65519 bytes): bytes to put after the JFIF segment."""
    out = b""
    if exif:
        if not exif.startswith(b"Exif\0\0"):
            exif = b"Exif\0\0" + exif
        out += b"\xff\xe1" + (len(exif) + 2).to_bytes(2, "big") + exif
    if icc:
        chunks = [icc[i : i + 65519] for i in range(0, len(icc), 65519)]
        for k, c in enumerate(chunks, 1):
            body = b"ICC_PROFILE\0" + bytes([k, len(chunks)]) + c
            out += b"\xff\xe2" + (len(body) + 2).to_bytes(2, "big") + body
    return out


NVJPEG_MIN_PIXELS = 10_000_000  # below this Pillow is as fast or faster (measured on the RTX 3060: a 2.6 MP picture 0.06 s with Pillow, 0.23 s with nvJPEG, whose fixed cost is about 0.2 s; 13 MP 0.31 s against 0.25 s, 16 MP 0.47 s against 0.25 s)


_nvjpeg_lock = threading.Lock()


def _decodes(data, ref=None, tol=12.0):
    """True when the JPEG bytes decode completely and, given `ref` (a small uint8 (h, w, 3) RGB copy of the picture that was encoded), look like it: a damaged entropy stream often decodes without any error but with grey blocks from the damage onwards, so the decoded picture, reduced to `ref`'s size, must be within `tol` grey levels of it on average (JPEG at q90 differs by 1-2)."""
    import io

    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as im:
            if ref is not None:
                im.draft(
                    "RGB", (max(1, ref.shape[1]), max(1, ref.shape[0]))
                )  # a reduced decode (DCT scaling): fast, and still reads the whole entropy stream
            im.load()
            got = np.asarray(im.convert("RGB"))
    except Exception:
        return False
    if ref is None:
        return True
    got = cv2.resize(got, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    return float(np.abs(got.astype(np.float32) - ref.astype(np.float32)).mean()) <= tol


def encode_jpeg_gpu(rgb, quality, exif=None, icc=None):
    """The picture (float 0..1: a (H, W, 3) array, or a (3, H, W) tensor on the device) as JPEG bytes made by the card's encoder (nvJPEG through torchvision: 4:4:4, no Huffman optimisation, so a file about 1 % larger than Pillow's `optimize`, and decoded pixels within 3 levels of Pillow's at the same quality), with the Exif and ICC segments put after the JFIF one. None when the picture is under NVJPEG_MIN_PIXELS, there is no CUDA card, torchvision or nvJPEG, STRATASTILLS_NVJPEG=0, or anything fails: the caller writes with Pillow (the reference)."""
    if os.environ.get("STRATASTILLS_NVJPEG", "1").strip().lower() in ("0", "false", "off", "no"):
        return None
    try:
        from stratastills import gpu

        shape = rgb.shape[-2:] if not isinstance(rgb, np.ndarray) else rgb.shape[:2]
        if shape[0] * shape[1] < NVJPEG_MIN_PIXELS or not gpu.available() or gpu.dev().type != "cuda":
            return None
        from torchvision.io import encode_jpeg

        t = gpu.torch_()
        with (
            gpu.slot(),
            _nvjpeg_lock,
        ):  # one encode at a time in this process, after the picture's kernels have finished: with other work on the card the encoder returned full-size files whose bitstream was corrupt ('bad Huffman code', 'premature end of data segment'), which decode as flat grey (arena build, 2026-11-09)
            x = rgb if isinstance(rgb, t.Tensor) else gpu.to_gpu(rgb)
            u8 = (x.clamp(0, 1) * 255 + 0.5).to(t.uint8).contiguous()
            t.cuda.synchronize()
            data = bytes(encode_jpeg(u8, quality=int(quality)).cpu().numpy().tobytes())
            ref = t.nn.functional.avg_pool2d(u8[None].float(), 8)[0].round().to(t.uint8).permute(1, 2, 0).cpu().numpy()
            del u8  # an 1/8 copy to check the file against
        if data[:2] != b"\xff\xd8" or not _decodes(data, ref):
            return None  # a corrupt bitstream is never written: the caller encodes with Pillow instead
        i = 2
        if data[2:4] == b"\xff\xe0":
            i = 4 + int.from_bytes(data[4:6], "big")  # after the JFIF segment
        return data[:i] + _app_segments(exif, icc) + data[i:]
    except Exception:
        return None


def write_picture(path, rgb, fmt, quality=95, exif=None):
    """Write the float picture (an array, or for a JPEG also a (3, H, W) tensor on the device): 16 bit TIFF, or JPEG at `quality` (an int; 'auto' is chosen by the caller from the 8-bit picture, see jpeg_quality). The JPEG is made by the card's encoder when there is one (encode_jpeg_gpu), else by Pillow, the reference."""
    from PIL import Image, ImageFile

    ImageFile.MAXBLOCK = max(
        ImageFile.MAXBLOCK, 1 << 27
    )  # with optimize=True Pillow fails ("Suspension not allowed here") when the encoder's buffer is smaller than the file
    if fmt != "tiff":
        data = encode_jpeg_gpu(rgb, quality, exif, _icc())
        if data is not None:
            with open(path, "wb") as f:
                f.write(data)
            return
    if not isinstance(rgb, np.ndarray):
        from stratastills import gpu

        rgb = gpu.to_numpy(rgb)
    if fmt == "tiff":
        import tifffile

        icc = _icc()
        tifffile.imwrite(
            path,
            (np.clip(rgb, 0, 1) * 65535 + 0.5).astype(np.uint16),
            photometric="rgb",
            compression="deflate",
            extratags=[(34675, 7, len(icc), icc, True)],
        )
    else:
        Image.fromarray(to_u8(rgb)).save(
            path,
            "JPEG",
            quality=int(quality),
            subsampling=0,
            optimize=True,
            icc_profile=_icc(),
            **({"exif": exif} if exif else {}),
        )


def to_u8(rgb):
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
