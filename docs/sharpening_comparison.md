# Sharpening comparison (2026-10-24)

Made with `scripts/sharpen_sweep.py` (pictures kept before sharpening, CPU pipeline) and `scripts/compare_detail.py --report`. Our JPEGs are written at **q95** for comparison runs (`COMPARE_QUALITY`; the product's own default stays q90): the others write about q92 (RawTherapee), q95 (darktable) and q96-99 (camera JPEGs), estimated from their quantisation tables, and the JPEG quality alone moves the flat-area `noise` metric (the same pictures: 0.62 at q90, 0.73 at q95, 0.97 at q98 on the A7C set), so noise and detail are only comparable between files of similar quality. 'new-default' is amount 3, roll-off 0.2-0.5 / 0.1-0.3, overshoot 0; 'previous-default' 1.5, 0.12-0.30 / 0.06-0.16, 0.05. RawTherapee: Auto-Matched profile; darktable: default; both rendered on the Windows machine. Metrics: see the docstring of compare_detail.py.

## A7C set

46 pictures from `a7c_candidates.json`, all at 3000 px on the long side. sharp and detail: higher is better; halo, noise, cnoise: lower is better. Win = better by more than 3% (a tie otherwise).

### All pictures

| program | sharp | halo | ring | noise | cnoise | detail | halo/sharp |
|---|---:|---:|---:|---:|---:|---:|---:|
| camera | 18.49 | 20.10 | 5.11 | 0.97 | 0.51 | 3.33 | 1.075 |
| new-default | 18.88 | 16.87 | 4.72 | 0.73 | 0.50 | 3.19 | 0.904 |
| previous-default | 17.21 | 19.60 | 4.60 | 0.73 | 0.50 | 2.98 | 1.149 |
| rawtherapee | 15.91 | 18.47 | 4.56 | 0.96 | 0.81 | 2.47 | 1.194 |
| darktable | 17.40 | 19.31 | 5.32 | 2.62 | 2.04 | 1.81 | 1.151 |

Pictures where new-default wins / ties / loses:

| against | sharp | halo | ring | noise | cnoise | detail |
|---|---|---|---|---|---|---|
| camera | 24 / 6 / 16 | 43 / 2 / 1 | 35 / 4 / 7 | 19 / 3 / 24 | 23 / 6 / 17 | 16 / 5 / 25 |
| previous-default | 39 / 7 / 0 | 41 / 5 / 0 | 8 / 19 / 19 | 0 / 36 / 10 | 0 / 46 / 0 | 32 / 14 / 0 |
| rawtherapee | 36 / 0 / 10 | 42 / 4 / 0 | 16 / 4 / 26 | 30 / 3 / 13 | 42 / 1 / 3 | 32 / 1 / 13 |
| darktable | 28 / 4 / 14 | 39 / 2 / 5 | 26 / 2 / 18 | 41 / 0 / 5 | 45 / 0 / 1 | 38 / 0 / 8 |

### ISO 0-250 (12 pictures)

| program | sharp | halo | ring | noise | cnoise | detail | halo/sharp |
|---|---:|---:|---:|---:|---:|---:|---:|
| camera | 26.64 | 24.71 | 8.04 | 0.62 | 0.45 | 6.03 | 0.994 |
| new-default | 24.68 | 19.56 | 8.29 | 0.90 | 0.51 | 5.39 | 0.676 |
| previous-default | 23.53 | 23.77 | 7.69 | 0.85 | 0.51 | 5.16 | 1.011 |
| rawtherapee | 23.10 | 23.84 | 6.78 | 0.71 | 0.71 | 4.60 | 1.039 |
| darktable | 18.47 | 20.97 | 6.06 | 0.74 | 0.60 | 3.03 | 1.103 |

Pictures where new-default wins / ties / loses:

| against | sharp | halo | ring | noise | cnoise | detail |
|---|---|---|---|---|---|---|
| camera | 8 / 0 / 4 | 12 / 0 / 0 | 7 / 2 / 3 | 2 / 0 / 10 | 3 / 5 / 4 | 2 / 1 / 9 |
| previous-default | 12 / 0 / 0 | 12 / 0 / 0 | 2 / 5 / 5 | 0 / 5 / 7 | 0 / 12 / 0 | 9 / 3 / 0 |
| rawtherapee | 11 / 0 / 1 | 11 / 1 / 0 | 3 / 1 / 8 | 5 / 2 / 5 | 11 / 1 / 0 | 7 / 1 / 4 |
| darktable | 11 / 1 / 0 | 7 / 1 / 4 | 0 / 1 / 11 | 7 / 0 / 5 | 11 / 0 / 1 | 7 / 0 / 5 |

### ISO 251-800 (7 pictures)

| program | sharp | halo | ring | noise | cnoise | detail | halo/sharp |
|---|---:|---:|---:|---:|---:|---:|---:|
| camera | 16.52 | 19.70 | 7.35 | 0.65 | 0.47 | 2.77 | 1.027 |
| new-default | 17.47 | 16.82 | 6.22 | 0.72 | 0.68 | 2.48 | 0.901 |
| previous-default | 15.69 | 19.96 | 5.87 | 0.72 | 0.68 | 2.24 | 1.107 |
| rawtherapee | 13.63 | 16.61 | 5.25 | 0.97 | 1.01 | 1.78 | 1.125 |
| darktable | 16.06 | 18.09 | 5.34 | 0.90 | 1.09 | 2.26 | 1.079 |

Pictures where new-default wins / ties / loses:

| against | sharp | halo | ring | noise | cnoise | detail |
|---|---|---|---|---|---|---|
| camera | 4 / 1 / 2 | 7 / 0 / 0 | 6 / 0 / 1 | 0 / 1 / 6 | 0 / 0 / 7 | 0 / 1 / 6 |
| previous-default | 6 / 1 / 0 | 6 / 1 / 0 | 0 / 2 / 5 | 0 / 6 / 1 | 0 / 7 / 0 | 5 / 2 / 0 |
| rawtherapee | 6 / 0 / 1 | 6 / 1 / 0 | 0 / 1 / 6 | 5 / 1 / 1 | 6 / 0 / 1 | 7 / 0 / 0 |
| darktable | 5 / 0 / 2 | 6 / 1 / 0 | 2 / 1 / 4 | 7 / 0 / 0 | 7 / 0 / 0 | 5 / 0 / 2 |

### ISO 801-2500 (10 pictures)

| program | sharp | halo | ring | noise | cnoise | detail | halo/sharp |
|---|---:|---:|---:|---:|---:|---:|---:|
| camera | 17.36 | 19.31 | 3.62 | 0.94 | 0.54 | 3.00 | 1.150 |
| new-default | 17.02 | 15.48 | 3.46 | 0.89 | 0.59 | 2.82 | 0.935 |
| previous-default | 15.49 | 18.06 | 3.20 | 0.87 | 0.59 | 2.56 | 1.174 |
| rawtherapee | 15.60 | 18.14 | 3.33 | 1.43 | 1.11 | 1.82 | 1.258 |
| darktable | 15.67 | 18.93 | 3.60 | 2.29 | 2.04 | 1.43 | 1.202 |

Pictures where new-default wins / ties / loses:

| against | sharp | halo | ring | noise | cnoise | detail |
|---|---|---|---|---|---|---|
| camera | 5 / 2 / 3 | 10 / 0 / 0 | 7 / 1 / 2 | 4 / 0 / 6 | 3 / 1 / 6 | 4 / 3 / 3 |
| previous-default | 10 / 0 / 0 | 10 / 0 / 0 | 4 / 2 / 4 | 0 / 8 / 2 | 0 / 10 / 0 | 8 / 2 / 0 |
| rawtherapee | 8 / 0 / 2 | 10 / 0 / 0 | 4 / 1 / 5 | 7 / 0 / 3 | 8 / 0 / 2 | 7 / 0 / 3 |
| darktable | 7 / 1 / 2 | 10 / 0 / 0 | 8 / 0 / 2 | 10 / 0 / 0 | 10 / 0 / 0 | 9 / 0 / 1 |

### ISO 2501-6000 (7 pictures)

| program | sharp | halo | ring | noise | cnoise | detail | halo/sharp |
|---|---:|---:|---:|---:|---:|---:|---:|
| camera | 16.62 | 18.44 | 4.22 | 1.27 | 0.63 | 2.68 | 1.074 |
| new-default | 17.47 | 16.79 | 3.26 | 0.69 | 0.48 | 3.35 | 0.909 |
| previous-default | 16.12 | 19.00 | 3.23 | 0.68 | 0.48 | 3.04 | 1.179 |
| rawtherapee | 14.51 | 17.74 | 2.81 | 0.81 | 0.78 | 2.57 | 1.218 |
| darktable | 15.62 | 18.46 | 4.81 | 3.43 | 4.73 | 1.38 | 1.183 |

Pictures where new-default wins / ties / loses:

| against | sharp | halo | ring | noise | cnoise | detail |
|---|---|---|---|---|---|---|
| camera | 6 / 1 / 0 | 7 / 0 / 0 | 5 / 1 / 1 | 4 / 2 / 1 | 7 / 0 / 0 | 3 / 0 / 4 |
| previous-default | 7 / 0 / 0 | 7 / 0 / 0 | 0 / 4 / 3 | 0 / 7 / 0 | 0 / 7 / 0 | 6 / 1 / 0 |
| rawtherapee | 7 / 0 / 0 | 6 / 1 / 0 | 1 / 0 / 6 | 4 / 0 / 3 | 7 / 0 / 0 | 4 / 0 / 3 |
| darktable | 5 / 1 / 1 | 6 / 0 / 1 | 6 / 0 / 1 | 7 / 0 / 0 | 7 / 0 / 0 | 7 / 0 / 0 |

### ISO 6001-1000000 (10 pictures)

| program | sharp | halo | ring | noise | cnoise | detail | halo/sharp |
|---|---:|---:|---:|---:|---:|---:|---:|
| camera | 21.05 | 24.03 | 6.01 | 1.52 | 0.51 | 2.90 | 1.141 |
| new-default | 19.98 | 18.87 | 4.30 | 0.64 | 0.38 | 3.52 | 1.000 |
| previous-default | 18.54 | 22.86 | 4.69 | 0.64 | 0.38 | 3.21 | 1.248 |
| rawtherapee | 18.25 | 23.05 | 4.91 | 1.09 | 0.67 | 2.60 | 1.264 |
| darktable | 21.83 | 25.31 | 7.39 | 5.03 | 6.23 | 1.31 | 1.127 |

Pictures where new-default wins / ties / loses:

| against | sharp | halo | ring | noise | cnoise | detail |
|---|---|---|---|---|---|---|
| camera | 1 / 2 / 7 | 7 / 2 / 1 | 10 / 0 / 0 | 9 / 0 / 1 | 10 / 0 / 0 | 7 / 0 / 3 |
| previous-default | 4 / 6 / 0 | 6 / 4 / 0 | 2 / 6 / 2 | 0 / 10 / 0 | 0 / 10 / 0 | 4 / 6 / 0 |
| rawtherapee | 4 / 0 / 6 | 9 / 1 / 0 | 8 / 1 / 1 | 9 / 0 / 1 | 10 / 0 / 0 | 7 / 0 / 3 |
| darktable | 0 / 1 / 9 | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |


## DJI set (Mini 4 Pro, ISO 100)

13 pictures from `dji_candidates.json`, all at 3000 px on the long side. sharp and detail: higher is better; halo, noise, cnoise: lower is better. Win = better by more than 3% (a tie otherwise).

### All pictures

| program | sharp | halo | ring | noise | cnoise | detail | halo/sharp |
|---|---:|---:|---:|---:|---:|---:|---:|
| camera | 65.70 | 48.61 | 18.74 | 1.92 | 0.56 | 7.54 | 0.748 |
| new-default | 52.03 | 32.82 | 14.23 | 1.27 | 0.59 | 8.07 | 0.636 |
| previous-default | 47.26 | 43.52 | 13.30 | 1.26 | 0.59 | 6.83 | 0.915 |
| rawtherapee | 51.57 | 42.07 | 13.44 | 4.43 | 1.87 | 2.06 | 0.815 |
| darktable | 45.51 | 38.24 | 13.30 | 4.00 | 2.21 | 1.93 | 0.840 |

Pictures where new-default wins / ties / loses:

| against | sharp | halo | ring | noise | cnoise | detail |
|---|---|---|---|---|---|---|
| camera | 0 / 0 / 13 | 13 / 0 / 0 | 12 / 1 / 0 | 12 / 1 / 0 | 3 / 5 / 5 | 11 / 0 / 2 |
| previous-default | 12 / 1 / 0 | 12 / 1 / 0 | 0 / 2 / 11 | 0 / 10 / 3 | 0 / 13 / 0 | 12 / 1 / 0 |
| rawtherapee | 5 / 7 / 1 | 13 / 0 / 0 | 5 / 3 / 5 | 13 / 0 / 0 | 13 / 0 / 0 | 13 / 0 / 0 |
| darktable | 12 / 0 / 1 | 9 / 2 / 2 | 1 / 2 / 10 | 13 / 0 / 0 | 13 / 0 / 0 | 13 / 0 / 0 |

### ISO 0-250 (13 pictures)

| program | sharp | halo | ring | noise | cnoise | detail | halo/sharp |
|---|---:|---:|---:|---:|---:|---:|---:|
| camera | 65.70 | 48.61 | 18.74 | 1.92 | 0.56 | 7.54 | 0.748 |
| new-default | 52.03 | 32.82 | 14.23 | 1.27 | 0.59 | 8.07 | 0.636 |
| previous-default | 47.26 | 43.52 | 13.30 | 1.26 | 0.59 | 6.83 | 0.915 |
| rawtherapee | 51.57 | 42.07 | 13.44 | 4.43 | 1.87 | 2.06 | 0.815 |
| darktable | 45.51 | 38.24 | 13.30 | 4.00 | 2.21 | 1.93 | 0.840 |

Pictures where new-default wins / ties / loses:

| against | sharp | halo | ring | noise | cnoise | detail |
|---|---|---|---|---|---|---|
| camera | 0 / 0 / 13 | 13 / 0 / 0 | 12 / 1 / 0 | 12 / 1 / 0 | 3 / 5 / 5 | 11 / 0 / 2 |
| previous-default | 12 / 1 / 0 | 12 / 1 / 0 | 0 / 2 / 11 | 0 / 10 / 3 | 0 / 13 / 0 | 12 / 1 / 0 |
| rawtherapee | 5 / 7 / 1 | 13 / 0 / 0 | 5 / 3 / 5 | 13 / 0 / 0 | 13 / 0 / 0 | 13 / 0 / 0 |
| darktable | 12 / 0 / 1 | 9 / 2 / 2 | 1 / 2 / 10 | 13 / 0 / 0 | 13 / 0 / 0 | 13 / 0 / 0 |

