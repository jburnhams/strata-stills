"""List the generated pictures of a project, newest first, with how each was made. Usage: python scripts/show_generated.py <projects dir>/<project> [name filter]"""
import glob, json, os, sys


def main(project, flt=''):
    rows = []
    for j in glob.glob(os.path.join(project, 'generated', '*', '*.json')):
        try: r = json.load(open(j))
        except ValueError: continue
        if flt in r.get('stem', ''): rows.append(r)
    for r in sorted(rows, key=lambda r: r['created'], reverse=True):
        print(r['created'], r['stem'], os.path.basename(r['source']), 'native', r.get('native'), '->', r['output'], f"{r['bytes'] // 1000} kB q{r['quality']}")
        print('   ', ' | '.join(r['steps'])); print('    size:', r['size']['why'], '| level:', r.get('level'))


if __name__ == '__main__': main(*sys.argv[1:3])
