# Sharpening comparison (2026-10-24)

Made with `scripts/sharpen_sweep.py` (pictures kept before sharpening, CPU pipeline) and `scripts/compare_detail.py --report`. In the tables 'sw-*-c3o0' is the new default (amount 3, roll-off 0.2-0.5 / 0.1-0.3, overshoot 0), 'sw-*-base' the previous one (1.5, 0.12-0.30 / 0.06-0.16, 0.05). RawTherapee: Auto-Matched profile; darktable: default; both rendered on the Windows machine. Metrics: see the docstring of compare_detail.py.

## A7C set

46 pictures from `a7c_candidates.json`, all at 3000 px on the long side. sharp and detail: higher is better; halo, noise, cnoise: lower is better. Win = better by more than 3% (a tie otherwise).

### All pictures

| program | sharp | halo | ring | noise | cnoise | detail | halo/sharp |
|---|---:|---:|---:|---:|---:|---:|---:|
| camera | 18.49 | 20.10 | 5.11 | 0.97 | 0.51 | 3.33 | 1.075 |
| sw-a7c-c3o0 | 18.88 | 16.33 | 4.64 | 0.97 | 0.49 | 2.62 | 0.880 |
| sw-a7c-base | 17.21 | 19.54 | 4.62 | 0.97 | 0.49 | 2.38 | 1.152 |
| rawtherapee-auto-matched | 15.91 | 18.47 | 4.56 | 0.96 | 0.81 | 2.47 | 1.194 |
| darktable | 17.40 | 19.31 | 5.32 | 2.62 | 2.04 | 1.81 | 1.151 |

Pictures where sw-a7c-c3o0 wins / ties / loses:

| against | sharp | halo | ring | noise | cnoise | detail |
|---|---|---|---|---|---|---|
| camera | 24 / 6 / 16 | 44 / 2 / 0 | 34 / 5 / 7 | 18 / 0 / 28 | 29 / 4 / 13 | 13 / 1 / 32 |
| sw-a7c-base | 39 / 7 / 0 | 41 / 5 / 0 | 11 / 17 / 18 | 0 / 38 / 8 | 0 / 46 / 0 | 31 / 15 / 0 |
| rawtherapee-auto-matched | 36 / 0 / 10 | 44 / 2 / 0 | 16 / 5 / 25 | 20 / 1 / 25 | 43 / 1 / 2 | 26 / 3 / 17 |
| darktable | 28 / 5 / 13 | 41 / 2 / 3 | 26 / 1 / 19 | 35 / 0 / 11 | 46 / 0 / 0 | 33 / 0 / 13 |

### ISO 0-250 (12 pictures)

| program | sharp | halo | ring | noise | cnoise | detail | halo/sharp |
|---|---:|---:|---:|---:|---:|---:|---:|
| camera | 26.64 | 24.71 | 8.04 | 0.62 | 0.45 | 6.03 | 0.994 |
| sw-a7c-c3o0 | 24.67 | 18.90 | 8.25 | 0.98 | 0.44 | 5.18 | 0.651 |
| sw-a7c-base | 23.53 | 23.68 | 7.66 | 0.93 | 0.44 | 4.93 | 1.002 |
| rawtherapee-auto-matched | 23.10 | 23.84 | 6.78 | 0.71 | 0.71 | 4.60 | 1.039 |
| darktable | 18.47 | 20.97 | 6.06 | 0.74 | 0.60 | 3.03 | 1.103 |

Pictures where sw-a7c-c3o0 wins / ties / loses:

| against | sharp | halo | ring | noise | cnoise | detail |
|---|---|---|---|---|---|---|
| camera | 8 / 0 / 4 | 12 / 0 / 0 | 6 / 3 / 3 | 1 / 0 / 11 | 9 / 1 / 2 | 2 / 0 / 10 |
| sw-a7c-base | 12 / 0 / 0 | 12 / 0 / 0 | 4 / 3 / 5 | 0 / 6 / 6 | 0 / 12 / 0 | 9 / 3 / 0 |
| rawtherapee-auto-matched | 11 / 0 / 1 | 12 / 0 / 0 | 3 / 1 / 8 | 3 / 0 / 9 | 12 / 0 / 0 | 4 / 1 / 7 |
| darktable | 11 / 1 / 0 | 8 / 2 / 2 | 0 / 0 / 12 | 5 / 0 / 7 | 12 / 0 / 0 | 6 / 0 / 6 |

### ISO 251-800 (7 pictures)

| program | sharp | halo | ring | noise | cnoise | detail | halo/sharp |
|---|---:|---:|---:|---:|---:|---:|---:|
| camera | 16.52 | 19.70 | 7.35 | 0.65 | 0.47 | 2.77 | 1.027 |
| sw-a7c-c3o0 | 17.46 | 16.07 | 6.18 | 1.05 | 0.56 | 1.87 | 0.860 |
| sw-a7c-base | 15.68 | 19.87 | 5.87 | 1.05 | 0.56 | 1.72 | 1.103 |
| rawtherapee-auto-matched | 13.63 | 16.61 | 5.25 | 0.97 | 1.01 | 1.78 | 1.125 |
| darktable | 16.06 | 18.09 | 5.34 | 0.90 | 1.09 | 2.26 | 1.079 |

Pictures where sw-a7c-c3o0 wins / ties / loses:

| against | sharp | halo | ring | noise | cnoise | detail |
|---|---|---|---|---|---|---|
| camera | 4 / 1 / 2 | 7 / 0 / 0 | 6 / 0 / 1 | 0 / 0 / 7 | 0 / 0 / 7 | 0 / 0 / 7 |
| sw-a7c-base | 6 / 1 / 0 | 6 / 1 / 0 | 0 / 2 / 5 | 0 / 6 / 1 | 0 / 7 / 0 | 5 / 2 / 0 |
| rawtherapee-auto-matched | 6 / 0 / 1 | 7 / 0 / 0 | 0 / 1 / 6 | 0 / 0 / 7 | 6 / 0 / 1 | 5 / 1 / 1 |
| darktable | 5 / 0 / 2 | 7 / 0 / 0 | 2 / 1 / 4 | 3 / 0 / 4 | 7 / 0 / 0 | 1 / 0 / 6 |

### ISO 801-2500 (10 pictures)

| program | sharp | halo | ring | noise | cnoise | detail | halo/sharp |
|---|---:|---:|---:|---:|---:|---:|---:|
| camera | 17.36 | 19.31 | 3.62 | 0.94 | 0.54 | 3.00 | 1.150 |
| sw-a7c-c3o0 | 17.04 | 15.01 | 3.48 | 1.18 | 0.56 | 2.23 | 0.913 |
| sw-a7c-base | 15.50 | 18.08 | 3.24 | 1.18 | 0.56 | 2.02 | 1.176 |
| rawtherapee-auto-matched | 15.60 | 18.14 | 3.33 | 1.43 | 1.11 | 1.82 | 1.258 |
| darktable | 15.67 | 18.93 | 3.60 | 2.29 | 2.04 | 1.43 | 1.202 |

Pictures where sw-a7c-c3o0 wins / ties / loses:

| against | sharp | halo | ring | noise | cnoise | detail |
|---|---|---|---|---|---|---|
| camera | 5 / 2 / 3 | 10 / 0 / 0 | 7 / 1 / 2 | 4 / 0 / 6 | 3 / 3 / 4 | 2 / 1 / 7 |
| sw-a7c-base | 10 / 0 / 0 | 10 / 0 / 0 | 4 / 3 / 3 | 0 / 9 / 1 | 0 / 10 / 0 | 7 / 3 / 0 |
| rawtherapee-auto-matched | 8 / 0 / 2 | 10 / 0 / 0 | 4 / 2 / 4 | 7 / 0 / 3 | 8 / 1 / 1 | 7 / 0 / 3 |
| darktable | 7 / 2 / 1 | 10 / 0 / 0 | 8 / 0 / 2 | 10 / 0 / 0 | 10 / 0 / 0 | 9 / 0 / 1 |

### ISO 2501-6000 (7 pictures)

| program | sharp | halo | ring | noise | cnoise | detail | halo/sharp |
|---|---:|---:|---:|---:|---:|---:|---:|
| camera | 16.62 | 18.44 | 4.22 | 1.27 | 0.63 | 2.68 | 1.074 |
| sw-a7c-c3o0 | 17.47 | 16.24 | 3.24 | 0.94 | 0.48 | 2.65 | 0.883 |
| sw-a7c-base | 16.12 | 18.93 | 3.27 | 0.93 | 0.48 | 2.42 | 1.178 |
| rawtherapee-auto-matched | 14.51 | 17.74 | 2.81 | 0.81 | 0.78 | 2.57 | 1.218 |
| darktable | 15.62 | 18.46 | 4.81 | 3.43 | 4.73 | 1.38 | 1.183 |

Pictures where sw-a7c-c3o0 wins / ties / loses:

| against | sharp | halo | ring | noise | cnoise | detail |
|---|---|---|---|---|---|---|
| camera | 6 / 1 / 0 | 7 / 0 / 0 | 5 / 1 / 1 | 4 / 0 / 3 | 7 / 0 / 0 | 3 / 0 / 4 |
| sw-a7c-base | 7 / 0 / 0 | 7 / 0 / 0 | 0 / 4 / 3 | 0 / 7 / 0 | 0 / 7 / 0 | 6 / 1 / 0 |
| rawtherapee-auto-matched | 7 / 0 / 0 | 6 / 1 / 0 | 1 / 0 / 6 | 2 / 1 / 4 | 7 / 0 / 0 | 4 / 0 / 3 |
| darktable | 5 / 1 / 1 | 6 / 0 / 1 | 6 / 0 / 1 | 7 / 0 / 0 | 7 / 0 / 0 | 7 / 0 / 0 |

### ISO 6001-1000000 (10 pictures)

| program | sharp | halo | ring | noise | cnoise | detail | halo/sharp |
|---|---:|---:|---:|---:|---:|---:|---:|
| camera | 21.05 | 24.03 | 6.01 | 1.52 | 0.51 | 2.90 | 1.141 |
| sw-a7c-c3o0 | 19.98 | 18.35 | 4.24 | 0.80 | 0.36 | 3.05 | 0.975 |
| sw-a7c-base | 18.52 | 22.58 | 4.68 | 0.80 | 0.36 | 2.84 | 1.236 |
| rawtherapee-auto-matched | 18.25 | 23.05 | 4.91 | 1.09 | 0.67 | 2.60 | 1.264 |
| darktable | 21.83 | 25.31 | 7.39 | 5.03 | 6.23 | 1.31 | 1.127 |

Pictures where sw-a7c-c3o0 wins / ties / loses:

| against | sharp | halo | ring | noise | cnoise | detail |
|---|---|---|---|---|---|---|
| camera | 1 / 2 / 7 | 8 / 2 / 0 | 10 / 0 / 0 | 9 / 0 / 1 | 10 / 0 / 0 | 6 / 0 / 4 |
| sw-a7c-base | 4 / 6 / 0 | 6 / 4 / 0 | 3 / 5 / 2 | 0 / 10 / 0 | 0 / 10 / 0 | 4 / 6 / 0 |
| rawtherapee-auto-matched | 4 / 0 / 6 | 9 / 1 / 0 | 8 / 1 / 1 | 8 / 0 / 2 | 10 / 0 / 0 | 6 / 1 / 3 |
| darktable | 0 / 1 / 9 | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |


## DJI set (Mini 4 Pro, ISO 100)

13 pictures from `dji_candidates.json`, all at 3000 px on the long side. sharp and detail: higher is better; halo, noise, cnoise: lower is better. Win = better by more than 3% (a tie otherwise).

### All pictures

| program | sharp | halo | ring | noise | cnoise | detail | halo/sharp |
|---|---:|---:|---:|---:|---:|---:|---:|
| camera | 65.70 | 48.61 | 18.74 | 1.92 | 0.56 | 7.54 | 0.748 |
| sw-dji-c3o0 | 52.03 | 31.77 | 14.14 | 1.37 | 0.54 | 7.95 | 0.619 |
| sw-dji-base | 47.26 | 43.22 | 13.25 | 1.37 | 0.54 | 6.61 | 0.907 |
| dji-rawtherapee-auto-matched | 51.57 | 42.07 | 13.44 | 4.43 | 1.87 | 2.06 | 0.815 |
| dji-darktable | 45.51 | 38.24 | 13.30 | 4.00 | 2.21 | 1.93 | 0.840 |

Pictures where sw-dji-c3o0 wins / ties / loses:

| against | sharp | halo | ring | noise | cnoise | detail |
|---|---|---|---|---|---|---|
| camera | 0 / 0 / 13 | 13 / 0 / 0 | 12 / 1 / 0 | 12 / 0 / 1 | 10 / 2 / 1 | 10 / 1 / 2 |
| sw-dji-base | 12 / 1 / 0 | 12 / 1 / 0 | 0 / 2 / 11 | 0 / 10 / 3 | 0 / 13 / 0 | 12 / 1 / 0 |
| dji-rawtherapee-auto-matched | 5 / 7 / 1 | 13 / 0 / 0 | 5 / 3 / 5 | 13 / 0 / 0 | 13 / 0 / 0 | 13 / 0 / 0 |
| dji-darktable | 12 / 0 / 1 | 11 / 0 / 2 | 1 / 3 / 9 | 13 / 0 / 0 | 13 / 0 / 0 | 13 / 0 / 0 |

### ISO 0-250 (13 pictures)

| program | sharp | halo | ring | noise | cnoise | detail | halo/sharp |
|---|---:|---:|---:|---:|---:|---:|---:|
| camera | 65.70 | 48.61 | 18.74 | 1.92 | 0.56 | 7.54 | 0.748 |
| sw-dji-c3o0 | 52.03 | 31.77 | 14.14 | 1.37 | 0.54 | 7.95 | 0.619 |
| sw-dji-base | 47.26 | 43.22 | 13.25 | 1.37 | 0.54 | 6.61 | 0.907 |
| dji-rawtherapee-auto-matched | 51.57 | 42.07 | 13.44 | 4.43 | 1.87 | 2.06 | 0.815 |
| dji-darktable | 45.51 | 38.24 | 13.30 | 4.00 | 2.21 | 1.93 | 0.840 |

Pictures where sw-dji-c3o0 wins / ties / loses:

| against | sharp | halo | ring | noise | cnoise | detail |
|---|---|---|---|---|---|---|
| camera | 0 / 0 / 13 | 13 / 0 / 0 | 12 / 1 / 0 | 12 / 0 / 1 | 10 / 2 / 1 | 10 / 1 / 2 |
| sw-dji-base | 12 / 1 / 0 | 12 / 1 / 0 | 0 / 2 / 11 | 0 / 10 / 3 | 0 / 13 / 0 | 12 / 1 / 0 |
| dji-rawtherapee-auto-matched | 5 / 7 / 1 | 13 / 0 / 0 | 5 / 3 / 5 | 13 / 0 / 0 | 13 / 0 / 0 | 13 / 0 / 0 |
| dji-darktable | 12 / 0 / 1 | 11 / 0 / 2 | 1 / 3 / 9 | 13 / 0 / 0 | 13 / 0 / 0 | 13 / 0 / 0 |

