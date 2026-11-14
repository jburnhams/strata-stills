import { http, HttpResponse } from 'msw'
import { setupServer } from 'msw/node'
import type { ArenaAlert, BrowseAdd, BrowsePicture, FlaggedPicture, GenerateBody, GenerateRecord, Item, Preset, Project, Rule, SourceSummary, Stage } from '../../src/api'

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
  http.get('/api/report', () => HttpResponse.json({ available: false, built: null })),
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
const fresh = () => ({ fullres: { state: 'ready', text: '' } as { state: string; text: string }, exists: false, existingVariant: 'gen-abc123', checks: [] as GenerateBody[], generates: [] as GenerateBody[], saved: [] as unknown[], rules: [] as Rule[], applies: 0, statusCalls: 0,
  pairs: 0, votes: [] as unknown[], flags: [] as unknown[], unflags: [] as unknown[], undos: 0, statuses: [] as unknown[], decisions: [] as unknown[], scans: [] as string[], removed: [] as string[] })
export const mock = fresh()
export const resetMock = () => { Object.assign(mock, fresh()) }

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

export const alerts: ArenaAlert[] = [
  { id: 'a2', token: 't2', pic: 'pic1', variant: 'auto1', other: 'auto2', side: 'right', note: 'banding in the sky', t: 1750000000, ms: 2500, votes_on_pic: 3, votes_on_variant: 2, status: 'fixed', resolved: true, pool: { file: 'img/1_auto1.jpg', hash: 'abc', key: 'k1' } },
  { id: 'a1', token: 't1', pic: 'pic2', variant: 'auto2', other: 'lightroom', side: 'left', t: 1700000000, ms: 1500, votes_on_pic: 2, votes_on_variant: 1, status: 'open', resolved: false, pool: null },
  { id: 'a0', pic: 'pic0', variant: 'auto3', other: 'lightroom', side: 'left', note: '', t: 1600000000, ms: 900, votes_on_pic: 1, votes_on_variant: 1, status: 'wontfix', resolved: true, pool: { file: 'img/0_auto3.jpg' } },
]
export const ratings = {
  votes: 6, superseded_votes: 1, pictures: 3, voted_pictures: 2, retired: [], left_odds: 1.02,
  variants: [{ name: 'lightroom', bt: 1620, elo: 1580, lo: 1550, hi: 1700, votes: 5 }, { name: 'auto1', bt: 1450, elo: 1470, lo: null, hi: null, votes: 6 }, { name: 'auto2', bt: null, elo: null, lo: null, hi: null, votes: 0 }],
  head_to_head: [{ a: 'lightroom', b: 'auto1', a_wins: 5, b_wins: 0, ties: 0 }],
}
export const flagged: FlaggedPicture[] = [{ path: 'C:\\shoot\\beach.JPG', cls: 'EXPOSED', score: 0.91, allowed: false }, { path: 'C:\\shoot\\pool.JPG', cls: 'COVERED', score: 0.62, allowed: true }]

handlers.push(
  http.get('/api/arena/next', () => { mock.pairs++; return HttpResponse.json({ token: 'tok' + mock.pairs, left: `img/tok${mock.pairs}/left`, right: `img/tok${mock.pairs}/right`, pool: 3, votes: 6 }) }),
  http.post('/api/arena/vote', async ({ request }) => { const b = await request.json() as { choice: string }; mock.votes.push(b); return HttpResponse.json(b.choice === 'skip' ? { ok: true, skipped: true } : { ok: true, votes: 6 + mock.votes.length }) }),
  http.post('/api/arena/undo', () => { mock.undos++; return HttpResponse.json({ undone: {}, votes: 5 }) }),
  http.post('/api/arena/alert', async ({ request }) => { mock.flags.push(await request.json()); return HttpResponse.json({ ok: true, id: 'new' }) }),
  http.post('/api/arena/alert/clear', async ({ request }) => { mock.unflags.push(await request.json()); return HttpResponse.json({ ok: true, cleared: 1 }) }),
  http.get('/api/arena/ratings', () => HttpResponse.json(ratings)),
  http.get('/api/alerts', () => HttpResponse.json({ alerts })),
  http.post('/api/alerts/status', async ({ request }) => { mock.statuses.push(await request.json()); return HttpResponse.json({ ok: true }) }),
  http.get('/api/flagged', () => HttpResponse.json({ threshold: 0.5, enabled: true, items: flagged })),
  http.post('/api/flagged/decide', async ({ request }) => { mock.decisions.push(await request.json()); return HttpResponse.json({ ok: true }) }),
  http.post('/api/flagged/scan', async ({ request }) => { const b = await request.json() as { folder: string }; mock.scans.push(b.folder); return b.folder === 'bad' ? HttpResponse.json({ detail: 'not a folder: bad' }, { status: 400 }) : HttpResponse.json({ kept: 12, flagged: 2 }) }),
)

/** The Browser's folders and pictures, and what was added (reset after each test). */
const browsePics: BrowsePicture[] = [
  { stem: 'DSC00001', kind: 'still', files: ['DSC00001.ARW', 'DSC00001.JPG'], taken: '2019-03-02T10:00:00Z', bytes: 1, projects: ['trip'], sources: [{ id: 'lightroom', label: 'Lightroom export' }, { id: 'embedded', label: 'Camera preview' }] },
  { stem: 'DSC00002', kind: 'still', files: ['DSC00002.ARW'], taken: '2019-03-02T10:01:00Z', bytes: 1, projects: [], sources: [{ id: 'embedded', label: 'Camera preview' }] },
]
export const browse = { added: [] as BrowseAdd[] }
export const resetBrowse = () => { browse.added = [] }
handlers.push(
  http.get('/api/browse/folders', ({ request }) => {
    const f = new URL(request.url).searchParams.get('folder')
    return HttpResponse.json(!f ? [{ key: 'raw', name: 'raw', folders: 1, pictures: 0 }] : f === 'raw' ? [{ key: 'raw/2019-03-02', name: '2019-03-02', folders: 0, pictures: 2 }] : [])
  }),
  http.get('/api/browse/pictures', () => HttpResponse.json({ pictures: browsePics, checking: 0 })),
  http.get('/api/browse/thumb', () => new HttpResponse(null, { status: 404 })),
  http.post('/api/browse/add', async ({ request }) => { const b = await request.json() as BrowseAdd; browse.added.push(b); return HttpResponse.json({ project: b.project ?? 'picks', sources: b.folders?.length ?? 0, pictures: 1, unchanged: 0 }) }),
)

export const server = setupServer(...handlers)

