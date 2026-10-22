import { http, HttpResponse } from 'msw'
import { setupServer } from 'msw/node'
import type { Item, Project, SourceSummary } from '../../src/api'

export const project: Project = { id: 'demo', name: 'Demo', created: null, n_sources: 1, sources: [{ path: 'C:\\shoot' }] }
export const sources: SourceSummary[] = [{ id: 'abcd1234', path: 'C:\\shoot', reachable: true, stills: 2, panos: 1, videos: 1 }]
export const items: Item[] = [
  { id: 'abcd1234~A_1', stem: 'A_1', kind: 'still', parent: null, primary: 'raw', chosen: false, n_sidecars: 0, video: null,
    variants: [{ id: 'raw', label: 'Raw', path: 'C:\\shoot\\A_1.DNG', kind: 'raw', ext: '.dng' }, { id: 'jpeg', label: 'Camera JPEG', path: 'C:\\shoot\\A_1.JPG', kind: 'jpeg', ext: '.jpg' }] },
  { id: 'abcd1234~A_2', stem: 'A_2', kind: 'still', parent: null, primary: 'jpeg', chosen: false, n_sidecars: 0, video: null, variants: [{ id: 'jpeg', label: 'Camera JPEG', path: 'C:\\shoot\\A_2.JPG', kind: 'jpeg', ext: '.jpg' }] },
  { id: 'abcd1234~A_3', stem: 'A_3', kind: 'video', parent: null, primary: null, chosen: false, n_sidecars: 1, video: 'C:\\shoot\\A_3.MP4', variants: [] },
]

// Every endpoint the app calls has a handler here; one without a handler fails the test (setup.ts: onUnhandledFrame 'error').
export const handlers = [
  http.get('/api/settings', () => HttpResponse.json({ projects_dir: 'C:\\projects' })),
  http.get('/api/projects', () => HttpResponse.json([project])),
  http.post('/api/projects', async ({ request }) => { const b = await request.json() as { name: string }; return HttpResponse.json({ ...project, id: b.name.toLowerCase(), name: b.name }) }),
  http.get('/api/project', () => HttpResponse.json(project)),
  http.get('/api/project/sources', () => HttpResponse.json(sources)),
  http.post('/api/project/sources', async ({ request }) => { const b = await request.json() as { path: string }; return b.path === 'bad' ? HttpResponse.json({ detail: 'not a folder: bad' }, { status: 400 }) : HttpResponse.json(project) }),
  http.delete('/api/project/sources', () => HttpResponse.json(project)),
  http.get('/api/project/items', () => HttpResponse.json(items)),
  http.get('/api/project/item', ({ request }) => HttpResponse.json(items.find(i => i.id === new URL(request.url).searchParams.get('item')))),
  http.put('/api/project/primary', async ({ request }) => { const b = await request.json() as { variant: string }; return HttpResponse.json({ ...items[0], primary: b.variant, chosen: true }) }),
  http.get('/api/project/transform', ({ request }) => new URL(request.url).searchParams.get('transform') === 'level'
    ? HttpResponse.json({ applied: true, angle: -1.37, confidence: 0.49, lines: 751, nadir: true, pitch: -90 })
    : HttpResponse.json({ applied: false })),
]

export const server = setupServer(...handlers)
