"""Hold the GPU development of one raw to the CPU reference on the real card (the unit tests use synthetic pictures): the plan the card gets (budget, tile, largest whole picture), the times, and the difference between the two pictures, also after a 8 px blur (the grain differs a little between the routes, the tones and colours must not).
  python scripts/gpu_parity.py <raw> [--half]
Exit status 1 when the blurred difference reaches 0.01 (2.5 levels of 255) anywhere: the GPU route must then not be trusted on this card (STRATASTILLS_GPU=0)."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import cv2, numpy as np
from stratastills import develop as D, gpu

LIMIT = 0.01


def main():
    f = sys.argv[1]; half = '--half' in sys.argv
    print('gpu available', gpu.available(), '| budget %.2f GB' % ((gpu._cuda_total() or 0) / 1e9), '| tile', gpu.tile_size(), '| largest whole picture', gpu.max_pixels())
    t = time.time(); c = D.develop(f, half=half, backend='cpu'); print('cpu %.1f s' % (time.time() - t))
    t = time.time(); g = D.develop(f, half=half, backend='auto'); print('gpu %.1f s' % (time.time() - t))
    d = np.abs(c - g); b = np.abs(cv2.GaussianBlur(c - g, (0, 0), 8)); print('difference: max %.3f mean %.4f | blurred: max %.4f mean %.5f' % (d.max(), d.mean(), b.max(), b.mean()))
    return 1 if b.max() >= LIMIT else 0


if __name__ == '__main__': sys.exit(main())
