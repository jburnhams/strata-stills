import numpy as np
from stratastills.framing import inner_rect


def test_whole_mask_is_the_whole_frame():
    assert inner_rect(np.ones((40, 60), bool)) == (0, 0, 60, 40)


def test_empty_mask_falls_back_to_the_frame():
    assert inner_rect(np.zeros((40, 60), bool)) == (0, 0, 60, 40)


def test_a_wedge_in_the_corner_is_cut_off_where_no_row_or_column_is_whole():
    ok = np.ones((80, 120), bool); ok[60:, :30] = False; ok[:, 100:] = False       # the bottom-left corner and the right strip: every row and every column has invalid pixels at the end
    x0, y0, x1, y1 = inner_rect(ok, step=1); assert ok[y0:y1, x0:x1].all() and (x1 - x0) * (y1 - y0) >= 100 * 60


def test_the_rectangle_lies_inside_the_mask_with_a_pad_and_a_coarse_step():
    yy, xx = np.mgrid[0:200, 0:300]; ok = (xx - 0.2 * yy) > 10 + 0 * yy; ok &= (yy < 190)         # a slanted left edge as a small rotation leaves
    x0, y0, x1, y1 = inner_rect(ok, pad=3, step=4); assert x1 > x0 and y1 > y0 and ok[y0:y1, x0:x1].all()
