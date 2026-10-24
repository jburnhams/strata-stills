import { http, HttpResponse } from 'msw'
import { setupServer } from 'msw/node'
import type { GenerateBody, GenerateRecord, Item, Preset, Project, Rule, SourceSummary, Stage } from '../../src/api'

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

export const detail = { ok: true, native: [8064, 6048], recommended: 6016, scale50: 0.35, scale30: 0.75, scale20: 0.88, f50: 0.17, f30: 0.37, f20: 0.44 }
export const BEST: Stage[] = [{ stage: 'develop', highlight_contrast: 0.8 }, { stage: 'level' }, { stage: 'resize', size: 'auto' }, { stage: 'sharpen', amount: 3 }, { stage: 'encode', format: 'jpeg', quality: 90 }]
export const presets: Preset[] = [
  { name: 'best', label: 'Best quality (automatic)', builtin: true, stages: BEST },
  { name: 'archive', label: 'Archive: 16 bit TIFF, native size, as developed', builtin: true, stages: [{ stage: 'develop', highlight_contrast: 0.8 }, { stage: 'level' }, { stage: 'resize', size: 'native' }, { stage: 'encode', format: 'tiff', quality: 'auto' }] },
]
export const generatedRecord: GenerateRecord = { key: 'abc123', file: 'C:\p\generated\abc123.jpg', output: [6016, 4512], bytes: 15000000, format: 'jpeg', quality: 90, summary: 'JPEG q90 · size auto · levelled · sharpened 3', steps: ['developed from the raw'] }

/** What the mock server remembers within a test (reset after each one). */
export const mock = { fullres: { state: 'ready', text: '' } as { state: string; text: string }, exists: false, existingVariant: 'gen-abc123', checks: [] as GenerateBody[], generates: [] as GenerateBody[], saved: [] as unknown[], rules: [] as Rule[], applies: 0, statusCalls: 0 }
export const resetMock = () => { mock.fullres = { state: 'ready', text: '' }; mock.exists = false; mock.checks = []; mock.generates = []; mock.saved = []; mock.rules = []; mock.applies = 0; mock.statusCalls = 0 }

/** The stages a body asks for: the preset with the flat options laid over it (a small copy of what the server does, enough to show the summary and the Advanced values). */
export function stagesOf(b: GenerateBody): Stage[] {
  let st = (presets.find(p => p.name === (b.preset ?? 'best')) ?? presets[0]).stages.map(s => ({ ...s })); const o = b.options ?? {}, set = (n: string, k: string, v: string | number | boolean) => { const s = st.find(x => x.stage === n); if (s) s[k] = v }
  if (o.size) set('resize', 'size', o.size); if (o.format) set('encode', 'format', o.format); if (o.quality !== undefined) set('encode', 'quality', o.quality); if (o.sharpen_amount) set('sharpen', 'amount', o.sharpen_amount)
  if (o.highlight_contrast !== undefined) set('develop', 'highlight_contrast', o.highlight_contrast); if (o.exposure !== undefined) set('develop', 'exposure', o.exposure); if (o.level === false) st = st.filter(s => s.stage !== 'level'); if (o.sharpen === false) st = st.filter(s => s.stage !== 'sharpen')
  return st
}
const summaryOf = (st: Stage[]) => { const e = st.find(s => s.stage === 'encode')!, r = st.find(s => s.stage === 'resize')!; return [e.format === 'tiff' ? 'TIFF 16 bit' : `JPEG ${e.quality === 'auto' ? 'auto quality' : 'q' + e.quality}`, `size ${r.size}`, st.some(s => s.stage === 'level') ? 'levelled' : null, st.find(s => s.stage === 'sharpen') ? `sharpened ${st.find(s => s.stage === 'sharpen')!.amount}` : null].filter(Boolean).join(' · ') }

handlers.push(
  http.get('/api/project/detail', () => HttpResponse.json(detail)),
  http.get('/api/project/fullres', () => HttpResponse.json(mock.fullres)),
  http.get('/api/pipelines', () => HttpResponse.json(presets)),
  http.put('/api/pipeline', async ({ request }) => { const b = await request.json(); mock.saved.push([new URL(request.url).searchParams.get('name'), b]); return HttpResponse.json({ name: new URL(request.url).searchParams.get('name'), label: 'x', builtin: false, stages: BEST }) }),
  http.post('/api/project/generate/check', async ({ request }) => { const b = await request.json() as GenerateBody; mock.checks.push(b); const st = stagesOf(b); return HttpResponse.json({ key: 'abc123', exists: mock.exists, variant: mock.exists ? mock.existingVariant : null, stages: st, summary: summaryOf(st) }) }),
  http.post('/api/project/generate', async ({ request }) => { mock.generates.push(await request.json() as GenerateBody); return HttpResponse.json(mock.exists ? { state: 'exists', variant: mock.existingVariant } : { state: 'running', text: 'developing the full-size picture' }) }),
  http.get('/api/project/generate', () => HttpResponse.json({ state: 'done', result: generatedRecord })),
  http.get('/api/project/rules', () => HttpResponse.json({ rules: mock.rules, pending: mock.rules.length ? 3 : 0 })),
  http.put('/api/project/rules', async ({ request }) => { mock.rules = ((await request.json()) as { rules: Rule[] }).rules; return HttpResponse.json({ rules: mock.rules, pending: mock.rules.length ? 3 : 0 }) }),
  http.post('/api/project/apply', () => { mock.applies++; return HttpResponse.json({ state: 'running', done: 0, total: 3, text: 'starting', failed: [] }) }),
  http.get('/api/project/apply', () => { mock.statusCalls++; return HttpResponse.json({ state: 'done', done: 3, total: 3, text: '', failed: [] }) }),
  http.delete('/api/project/apply', () => HttpResponse.json({ state: 'stopped', done: 1, total: 3, failed: [] })),
)

export const server = setupServer(...handlers)
