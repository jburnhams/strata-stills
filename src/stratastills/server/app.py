"""The stratastills server: `./stratastills serve` opens on the list of projects to choose from.

    ./stratastills serve [--host 127.0.0.1] [--port 8361]

API (JSON): GET /api/settings; PUT /api/settings {projects_dir}; GET /api/projects; POST /api/projects {name, sources?, group?};
GET /api/project?id=; GET /api/project/sources?id=; GET /api/project/items?id=&source=; GET /api/project/item?id=&item=; PUT /api/project/primary?id=&item= {variant}; GET /api/project/image?id=&item=&variant=&transform=&px=; GET /api/project/transform?id=&item=&variant=&transform=; GET /api/project/shots?id=; POST /api/project/sources?id= {path}; DELETE /api/project/sources?id=&path=.
Project ids contain slashes, so they are query parameters. The page at / is a plain single-file stand-in until the React app (web/) exists. Binds to 127.0.0.1 by default: the person using it can name any folder as a source (read-only)."""
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from stratastills import preview, projects as P, settings

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


class SettingsIn(BaseModel):
    projects_dir: str | None = None


def create_app():
    app = FastAPI(title='strata-stills', docs_url='/api/docs', openapi_url='/api/openapi.json')

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

    @app.get('/api/project/transform')
    def get_transform(id: str, item: str, variant: str | None = None, transform: str = 'original'):
        _, info = guard(preview.render, id, item, variant, transform, 1600); return info

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
