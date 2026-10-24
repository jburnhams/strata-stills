"""Download the lensfun lens database (data/db/*.xml of https://github.com/lensfun/lensfun, licence CC-BY-SA 3.0: it stays a separate download and is not part of this repository) into models/lensfun/ (git-ignored), where lensfundb.py looks for it (or set STRATASTILLS_LENSFUN_DIR to another copy).
  python scripts/fetch_lensfun.py [--dir models/lensfun] [--only mil-sony mil-samyang ...]"""
import argparse, json, os, urllib.request

API = 'https://api.github.com/repos/lensfun/lensfun/contents/data/db'

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--dir', default=os.path.join(os.path.dirname(__file__), '..', 'models', 'lensfun')); ap.add_argument('--only', nargs='*'); a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    listing = json.load(urllib.request.urlopen(urllib.request.Request(API, headers={'Accept': 'application/vnd.github+json'})))
    names = [e for e in listing if e['name'].endswith('.xml') and (not a.only or e['name'][:-4] in a.only)]
    for e in names:
        with open(os.path.join(a.dir, e['name']), 'wb') as f: f.write(urllib.request.urlopen(e['download_url']).read())
    print(len(names), 'files into', os.path.abspath(a.dir))
