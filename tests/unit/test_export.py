def test_a_corrupt_or_truncated_jpeg_bitstream_is_refused():
    """export._decodes guards the GPU encoder: a file whose bitstream is cut short or damaged must not be written (it decodes, often silently, as flat grey from the damage on)."""
    import io

    import cv2
    import numpy as np
    from PIL import Image

    from stratastills import export

    y, x = np.mgrid[0:256, 0:384]
    a = np.stack([x % 256, y % 256, (x + y) % 256], -1).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "JPEG", quality=90)
    good = buf.getvalue()
    ref = cv2.resize(a, (48, 32), interpolation=cv2.INTER_AREA)
    m = len(good) // 3
    assert export._decodes(good) and export._decodes(good, ref)
    assert not export._decodes(good[:m])  # cut short: Pillow refuses it
    assert not export._decodes(good[:m] + b"\xff\xd9", ref)  # ended early by an EOI: decodes, but grey after it
    assert not export._decodes(
        good[:m] + bytes(np.random.default_rng(2).integers(0, 255, 400, dtype=np.uint8)) + good[m + 400 :], ref
    )
