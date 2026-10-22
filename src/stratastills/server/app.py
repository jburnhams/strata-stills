"""The stratastills server: `./stratastills serve` opens on the list of projects to choose from.

    ./stratastills serve [--host 127.0.0.1] [--port 8361]

API (JSON): GET /api/settings; PUT /api/settings {projects_dir}; GET /api/projects; POST /api/projects {name, sources?, group?};
GET /api/project?id=; GET /api/project/sources?id=; GET /api/project/items?id=&source=; GET /api/project/item?id=&item=; PUT /api/project/primary?id=&item= {variant}; GET /api/project/image?id=&item=&variant=&transform=&px=; GET /api/project/crop?id=&item=&variant=&transform=&x=&y=&size=&zoom= (the loupe: the same scene area of any variant and transform at full resolution; 202 while the variant's full-size copy is being made); GET /api/project/fullres?id=&item=&variant= (starts making it and reports {state: ready|running|failed|idle, text}); GET /api/project/transform?id=&item=&variant=&transform=; GET /api/project/detail?id=&item=&variant=; GET /api/pipelines, PUT/DELETE /api/pipeline?name= (presets); POST /api/project/generate/check?id=&item= and POST /api/project/generate?id=&item= {variant, preset, options, stages} (a new generated variant; runs in the background), GET /api/project/generate?id=&item= (progress), GET /api/project/generated/file?id=&item=&variant=; GET/PUT /api/project/rules?id=, POST/GET/DELETE /api/project/apply?id= (apply the rules to the sources); GET /api/project/shots?id=; POST /api/project/sources?id= {path}; DELETE /api/project/sources?id=&path=.
Project ids contain slashes, so they are query parameters. The page at / is a plain single-file stand-in until the React app (web/) exists. Binds to 127.0.0.1 by default: the person using it can name any folder as a source (read-only)."""
import os, threading

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from stratastills import export as X, pipeline as PL, preview, projects as P, settings

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
LANDING = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'landing.html')


class NewProject(BaseModel):
    name: str
    sources: list[str] = []
    group: str | None = None


class SourcePath(BaseModel):
    path: str


class PrimaryIn(BaseModel):
    variant: str


class GenerateIn(BaseModel):
    """What to make: a preset (default 'best') with the Advanced panel's flat options laid over it, or the full `stages` of a custom pipeline."""
    variant: str | None = None          # the source variant to start from (default: the item's primary)
    preset: str | None = None
    options: dict | None = None         # size, format, quality, level, sharpen, sharpen_amount, highlight_contrast
    stages: list[dict] | None = None


class PresetIn(BaseModel):
    label: str | None = None
    stages: list[dict]


class RulesIn(BaseModel):
    rules: list[dict]


class SettingsIn(BaseModel):
    projects_dir: str | None = None


def create_app():
    app = FastAPI(title='strata-stills', docs_url='/api/docs', openapi_url='/api/openapi.json')
    jobs = {}                                                           # (project, item) -> {state: running|done|failed, text, result, error}: generating takes a minute or two, so it runs in a thread and is polled
    batches = {}                                                        # project -> {state, done, total, text, failed, stop}: applying the rules to a whole source

    def guard(fn, *a, **kw):
        try: return fn(*a, **kw)
        except P.ProjectError as e: raise HTTPException(400, str(e))

    @app.get('/', response_class=HTMLResponse)
    def index():
        built = os.path.join(STATIC, 'index.html')          # the React app when it has been built (cd web; npm run build), else the plain stand-in page
        return open(built if os.path.exists(built) else LANDING, encoding='utf8').read()

    @app.get('/api/settings')
    def get_settings(): return settings.all_settings()

    @app.put('/api/settings')
    def put_settings(s: SettingsIn):
        if s.projects_dir is not None and not os.path.isabs(os.path.expanduser(s.projects_dir)): raise HTTPException(400, 'the projects folder must be an absolute path')
        settings.save(dict(projects_dir=s.projects_dir)); return settings.all_settings()

    @app.get('/api/projects')
    def list_projects(): return P.list_projects()

    @app.post('/api/projects')
    def new_project(b: NewProject): return guard(P.create, b.name, b.sources, b.group)

    @app.get('/api/project')
    def get_project(id: str): return guard(P.load, id)

    @app.get('/api/project/shots')
    def get_shots(id: str):
        out = []
        for src, sh in guard(P.shots, id): out.append(dict(source=src, unreachable=True) if sh is None else dict(sh.to_json(), source=src))
        return out

    @app.get('/api/project/sources')
    def get_sources(id: str): return guard(P.source_summaries, id)

    @app.get('/api/project/items')
    def get_items(id: str, source: str): return guard(P.source_items, id, source)

    @app.get('/api/project/item')
    def get_item(id: str, item: str): return guard(P.get_item, id, item)

    @app.put('/api/project/primary')
    def put_primary(id: str, item: str, b: PrimaryIn): return guard(P.set_primary, id, item, b.variant)

    @app.get('/api/project/image')
    def get_image(id: str, item: str, variant: str | None = None, transform: str = 'original', px: int = 1600):
        path, _ = guard(preview.render, id, item, variant, transform, px); return FileResponse(path, media_type='image/jpeg')

    @app.get('/api/project/crop')
    def get_crop(id: str, item: str, variant: str | None = None, transform: str = 'original', x: float = 0.5, y: float = 0.5, size: int = 200, zoom: int = 2):
        try: return Response(guard(preview.crop, id, item, variant, transform, x, y, size, zoom), media_type='image/jpeg')
        except preview.NotReady as e: return JSONResponse(dict(e.status, detail='the full-resolution copy of this picture is still being made'), status_code=202)

    @app.get('/api/project/fullres')
    def get_fullres(id: str, item: str, variant: str | None = None): return guard(preview.fullres_status, id, item, variant)

    @app.get('/api/project/transform')
    def get_transform(id: str, item: str, variant: str | None = None, transform: str = 'original'):
        _, info = guard(preview.render, id, item, variant, transform, 1600); return info

    @app.get('/api/project/detail')
    def get_detail(id: str, item: str, variant: str | None = None):
        it = guard(P.get_item, id, item); v = next((x for x in it['variants'] if x['id'] == (variant or it['primary'])), None)
        if v is None: raise HTTPException(400, 'no such variant')
        try: return X.measure_detail(id, v['path'])
        except Exception as e: raise HTTPException(400, str(e))

    # ---- pipelines: presets, generating a variant, rules for a whole source -------------------------------------------------------------------------------------------------------------------------------------
    @app.get('/api/pipelines')
    def get_pipelines(): return PL.list_presets()

    @app.put('/api/pipeline')
    def put_pipeline(name: str, b: PresetIn): return guard(PL.save_preset, name, b.label, b.stages)

    @app.delete('/api/pipeline')
    def delete_pipeline(name: str): guard(PL.delete_preset, name); return PL.list_presets()

    @app.post('/api/project/generate/check')
    def check_generate(id: str, item: str, b: GenerateIn):
        key, have = guard(PL.find_existing, id, item, b.variant, b.stages, b.preset, b.options); st = guard(PL.resolve, b.preset, b.options, b.stages)
        return dict(key=key, exists=have is not None, variant=have['id'] if have else None, stages=st, summary=PL.summary(st))

    @app.post('/api/project/generate')
    def start_generate(id: str, item: str, b: GenerateIn):
        key, have = guard(PL.find_existing, id, item, b.variant, b.stages, b.preset, b.options)
        if have is not None: return dict(state='exists', variant=have['id'])
        if jobs.get((id, item), {}).get('state') == 'running': raise HTTPException(409, 'a picture of this item is already being generated')
        job = jobs[(id, item)] = dict(state='running', text='starting', result=None, error=None)

        def run():
            try: job['result'] = PL.generate(id, item, b.variant, b.preset, b.options, b.stages, progress=lambda t: job.update(text=t)); job['state'] = 'done'
            except Exception as e: job.update(state='failed', error=str(e))
        threading.Thread(target=run, daemon=True).start(); return job

    @app.get('/api/project/generate')
    def generate_status(id: str, item: str): return jobs.get((id, item), dict(state='idle'))

    @app.get('/api/project/generated/file')
    def generated_file(id: str, item: str, variant: str):
        v = next((x for x in guard(P.get_item, id, item)['variants'] if x['id'] == variant and x['kind'] == 'generated'), None)
        if v is None or not os.path.isfile(v['path']): raise HTTPException(404, 'no such generated picture')
        return FileResponse(v['path'], filename=f"{item.split('~')[-1]}-{v['output'][0]}x{v['output'][1]}{v['ext']}")

    @app.get('/api/project/rules')
    def get_rules(id: str): return dict(rules=guard(PL.rules, id), pending=len(guard(PL.pending, id)))

    @app.put('/api/project/rules')
    def put_rules(id: str, b: RulesIn): guard(PL.set_rules, id, b.rules); return dict(rules=PL.rules(id), pending=len(PL.pending(id)))

    @app.post('/api/project/apply')
    def start_apply(id: str):
        guard(PL.rules, id)
        if batches.get(id, {}).get('state') == 'running': raise HTTPException(409, 'this project is already being processed')
        job = batches[id] = dict(state='running', done=0, total=0, text='starting', failed=[], stop=False)

        def run():
            def prog(i, n, t): job.update(done=i, total=n, text=t)
            try:
                r = PL.apply_rules(id, prog, stop=lambda: job['stop']); job.update(state='stopped' if job['stop'] else 'done', done=r['done'] + len(r['failed']), total=r['total'], failed=r['failed'])
            except Exception as e: job.update(state='failed', text=str(e))
        threading.Thread(target=run, daemon=True).start(); return job

    @app.get('/api/project/apply')
    def apply_status(id: str): return batches.get(id, dict(state='idle'))

    @app.delete('/api/project/apply')
    def stop_apply(id: str):
        if id in batches: batches[id]['stop'] = True
        return batches.get(id, dict(state='idle'))

    @app.post('/api/project/sources')
    def add_source(id: str, b: SourcePath): return guard(P.add_source, id, b.path)

    @app.delete('/api/project/sources')
    def remove_source(id: str, path: str): return guard(P.remove_source, id, path)

    if os.path.isdir(os.path.join(STATIC, 'assets')): app.mount('/assets', StaticFiles(directory=os.path.join(STATIC, 'assets')), name='assets')
    return app


def main(argv=None):
    import argparse, uvicorn
    a = argparse.ArgumentParser(); a.add_argument('--host', default='127.0.0.1'); a.add_argument('--port', type=int, default=8361); a = a.parse_args(argv)
    print(f'projects folder: {settings.projects_dir()}\nhttp://{a.host}:{a.port}/'); uvicorn.run(create_app(), host=a.host, port=a.port, log_level='warning')
