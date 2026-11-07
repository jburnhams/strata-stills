import numpy as np


def inner_rect(ok, pad=0, step=4):
    """(x0, y0, x1, y1) of the largest axis-aligned rectangle (by area) that lies wholly inside the boolean mask `ok`, less `pad` px each side; (0, 0, W, H) when `ok` is empty. A moved picture (rotated a little and shifted) has its invalid wedges along two or more edges, so no whole row or column is valid and a rule on rows and columns finds nothing (that fell back to the whole frame and kept the smear): the rectangle is found on the mask shrunk by `step` (the classic histogram method, row by row), then scaled back and kept inside the mask."""
    H, W = ok.shape; m = ok[:H // step * step, :W // step * step].reshape(H // step, step, W // step, step).all(axis=(1, 3)); h, w = m.shape; hist = np.zeros(w, int); best = (0, 0, 0, 0, 0)
    for y in range(h):
        hist = np.where(m[y], hist + 1, 0); st = []                                                  # stack of (start column, height) with rising heights
        for x in range(w + 1):
            cur = hist[x] if x < w else 0; start = x
            while st and st[-1][1] >= cur:
                sx, sh = st.pop(); a = sh * (x - sx)
                if a > best[0]: best = (a, sx, y - sh + 1, x, y + 1)
                start = sx
            st.append((start, cur))
    if best[0] == 0: return (0, 0, W, H)
    _, x0, y0, x1, y1 = best; return (x0 * step + pad, y0 * step + pad, x1 * step - pad, y1 * step - pad)
