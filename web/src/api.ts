import { fetchJson } from 'strata-react'
// Typed client for the FastAPI server (src/stratastills/server/app.py). Same-origin. Project ids contain slashes, so they travel as query parameters.
export interface Source { path: string; added?: string }
export interface Project { id: string; name: string; created: string | null; sources: Source[]; n_sources: number }
export interface SourceSummary { id: string; path: string; added?: string; reachable: boolean; stills: number; panos: number; videos: number; picked?: number | null }
export type ItemKind = 'still' | 'pano' | 'video'
export type Stage = { stage: string } & Record<string, string | number | boolean>
/** A variant of an item: a file of the source (raw, camera JPEG, a derived one) or a picture the app generated with a pipeline (kind 'generated', with the settings it was made with). */
export interface Variant {
  id: string; label: string; path: string; kind: 'raw' | 'jpeg' | 'derived' | 'generated'; ext: string
  key?: string; preset?: string | null; stages?: Stage[]; summary?: string; created?: string; output?: [number, number]; bytes?: number; quality?: number | null
}
export interface Item { id: string; stem: string; kind: ItemKind; parent: string | null; primary: string | null; chosen: boolean; variants: Variant[]; n_sidecars: number; video: string | null }
export interface TransformInfo { applied: boolean; angle?: number; confidence?: number; lines?: number; max_deg?: number; reason?: string | null; nadir?: boolean; pitch?: number | null }
export interface Detail { ok: boolean; reason?: string; native?: [number, number]; recommended?: number | null; f50?: number; f30?: number; f20?: number; scale50?: number; scale30?: number; scale20?: number; alpha?: number }
export interface Preset { name: string; label: string; builtin: boolean; stages: Stage[] }
/** The Advanced panel's flat settings, laid over a preset (only those changed). */
export interface Options { look?: string; size?: string; format?: 'jpeg' | 'tiff'; quality?: number | 'auto'; level?: boolean; sharpen?: boolean; sharpen_amount?: number; highlight_contrast?: number; exposure?: 'auto' | 'camera' | 'match' | number }
export interface GenerateBody { variant?: string | null; preset?: string | null; options?: Options | null; stages?: Stage[] | null }
export interface GenerateCheck { key: string; exists: boolean; variant: string | null; stages: Stage[]; summary: string }
export interface GenerateRecord { key: string; file: string; output: [number, number]; bytes: number; format: string; quality: number | null; summary: string; steps: string[]; existing?: boolean }
export interface GenerateJob { state: 'idle' | 'running' | 'done' | 'failed' | 'exists'; text?: string; result?: GenerateRecord | null; error?: string | null; variant?: string }
export interface FullRes { state: 'ready' | 'running' | 'failed' | 'idle'; text: string }
export interface Rule { source: string; pipeline: string; variant?: string }
export interface Rules { rules: Rule[]; pending: number }
export interface ApplyJob { state: 'idle' | 'running' | 'done' | 'stopped' | 'failed'; done?: number; total?: number; text?: string; failed?: [string, string][] }
export interface Transform { id: string; label: string }
/** The look's versions with their arena rating (Bradley-Terry, null before any votes), and which one 'best' is now. */
export interface LookVersions { best: string; versions: { name: string; rating: number | null; votes: number }[] }

/** A pair of the arena: two versions of one picture under neutral addresses (relative to /arena/), so the page cannot tell which is which. */
export interface ArenaPair { token: string; left: string; right: string; pool: number; votes: number }
export type ArenaChoice = 'left' | 'right' | 'tie' | 'skip'
export type ArenaSide = 'left' | 'right'
export interface VariantRating { name: string; bt: number | null; elo: number | null; lo: number | null; hi: number | null; votes: number }
export interface HeadToHead { a: string; b: string; a_wins: number; b_wins: number; ties: number }
export interface Ratings { votes: number; superseded_votes: number; pictures: number; voted_pictures: number; retired: string[]; left_odds: number; variants: VariantRating[]; head_to_head: HeadToHead[] }
/** The fix queue of the arena's flags (arena.STATUSES). */
export const ALERT_STATUSES = ['open', 'investigating', 'fixed', 'confirmed', 'wontfix'] as const
export type AlertStatus = (typeof ALERT_STATUSES)[number]
/** A flag raised in the arena on one version of one picture. */
export interface ArenaAlert {
  id: string; token?: string; pic: string; variant: string; other: string; side: ArenaSide; note?: string; t: number; ms: number
  votes_on_pic: number; votes_on_variant: number; status: AlertStatus; resolved: boolean; pool: { file: string; hash?: string; key?: string } | null
}
/** A picture the nudity filter flagged (nsfw.review). */
export interface FlaggedPicture { path: string; cls: string; score: number; allowed: boolean }
export interface FlaggedList { threshold: number; enabled: boolean; items: FlaggedPicture[] }

/** The Browser (browser.py). A folder key is `<root label>/<path below it>` (`raw/2019/2019-03-02`). */
export interface BrowseFolder { key: string; name: string; folders: number; pictures: number }
/** Where a thumbnail can come from, best first: a develop (`gen:<project>:<key>`), `lightroom`, `embedded` (the raw's preview), `jpeg`, `derived:<tag>`, `raw`. */
export interface ThumbSource { id: string; label: string }
export interface BrowsePicture { stem: string; kind: ItemKind; files: string[]; taken: string | null; bytes: number | null; sources: ThumbSource[]; projects: string[] }
/** A folder's pictures; `checking` are still with the nudity filter and are listed once it has looked at them. */
export interface BrowseListing { pictures: BrowsePicture[]; checking: number }
export interface BrowseAdd { project?: string | null; name?: string | null; folders?: string[]; subfolders?: boolean; pictures?: Record<string, string[]> }
export interface BrowseAdded { project: string; sources: number; pictures: number; unchanged: number }

/** The transforms the server offers on a picture, in order (transforms.py ORDER). */
export const TRANSFORMS: Transform[] = [{ id: 'original', label: 'Original' }, { id: 'level', label: 'Levelled' }]

// Where the app is served from: '' at the server's own root, '/strata-stills' behind a reverse proxy at that path (Caddy on the home server). Taken from the page's own URL (index.html has relative asset paths: vite.config.ts base './'), so one build works in both places.
export const BASE = typeof document === 'undefined' ? '' : new URL('.', document.baseURI).pathname.replace(/\/$/, '')      // no page (the unit tests run in node): the root
export const withBase = (u: string) => (u.startsWith('/') ? BASE + u : u)

export const q = (id: string) => '?id=' + encodeURIComponent(id)
const qi = (id: string, item: string) => q(id) + '&item=' + encodeURIComponent(item)

// strata-react's fetchJson: a non-2xx answer throws an HttpError carrying the server's own reason (FastAPI's `detail`), which useLoad and useAsyncAction put into words.
const req = <T,>(url: string, init?: RequestInit): Promise<T> => fetchJson<T>(withBase(url), init)
const send = (method: string, body: unknown): RequestInit => ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

export const api = {
  arenaNext: (query: string) => req<ArenaPair>('/api/arena/next' + query),
  vote: (token: string, choice: ArenaChoice, ms: number) => req<{ ok: boolean; votes?: number; skipped?: boolean }>('/api/arena/vote', send('POST', { token, choice, ms })),
  undoVote: () => req<{ undone: unknown; votes: number }>('/api/arena/undo', { method: 'POST' }),
  flag: (token: string, side: ArenaSide, ms: number) => req<{ ok: boolean; id: string }>('/api/arena/alert', send('POST', { token, side, note: '', ms })),
  unflag: (token: string, side: ArenaSide) => req<{ ok: boolean; cleared: number }>('/api/arena/alert/clear', send('POST', { token, side })),
  ratings: () => req<Ratings>('/api/arena/ratings'),
  alerts: () => req<{ alerts: ArenaAlert[] }>('/api/alerts'),
  setAlert: (id: string, status: AlertStatus, note: string) => req<{ ok: boolean }>('/api/alerts/status', send('POST', { id, status, note })),
  flagged: () => req<FlaggedList>('/api/flagged'),
  decideFlagged: (path: string, allow: boolean) => req<{ ok: boolean }>('/api/flagged/decide', send('POST', { path, allow })),
  scanFlagged: (folder: string) => req<{ kept: number; flagged: number }>('/api/flagged/scan', send('POST', { folder })),
  report: () => req<{ available: boolean; built: number | null; neutral?: boolean; auto?: boolean }>('/api/report'),
  settings: () => req<{ projects_dir: string }>('/api/settings'),
  projects: () => req<Project[]>('/api/projects'),
  createProject: (name: string, sources: string[]) => req<Project>('/api/projects', send('POST', { name, sources })),
  project: (id: string) => req<Project>('/api/project' + q(id)),
  sources: (id: string) => req<SourceSummary[]>('/api/project/sources' + q(id)),
  addSource: (id: string, path: string) => req<Project>('/api/project/sources' + q(id), send('POST', { path })),
  removeSource: (id: string, path: string) => req<Project>('/api/project/sources' + q(id) + '&path=' + encodeURIComponent(path), { method: 'DELETE' }),
  items: (id: string, source: string) => req<Item[]>('/api/project/items' + q(id) + '&source=' + encodeURIComponent(source)),
  item: (id: string, item: string) => req<Item>('/api/project/item' + qi(id, item)),
  setPrimary: (id: string, item: string, variant: string) => req<Item>('/api/project/primary' + qi(id, item), send('PUT', { variant })),
  detail: (id: string, item: string, variant: string) => req<Detail>('/api/project/detail' + qi(id, item) + '&variant=' + encodeURIComponent(variant)),
  transform: (id: string, item: string, variant: string, transform: string) => req<TransformInfo>('/api/project/transform' + qi(id, item) + '&variant=' + encodeURIComponent(variant) + '&transform=' + transform),
  pipelines: () => req<Preset[]>('/api/pipelines'),
  looks: () => req<LookVersions>('/api/look'),
  savePipeline: (name: string, label: string, stages: Stage[]) => req<Preset>('/api/pipeline?name=' + encodeURIComponent(name), send('PUT', { label, stages })),
  deletePipeline: (name: string) => req<Preset[]>('/api/pipeline?name=' + encodeURIComponent(name), { method: 'DELETE' }),
  checkGenerate: (id: string, item: string, b: GenerateBody) => req<GenerateCheck>('/api/project/generate/check' + qi(id, item), send('POST', b)),
  generate: (id: string, item: string, b: GenerateBody) => req<GenerateJob>('/api/project/generate' + qi(id, item), send('POST', b)),
  generateStatus: (id: string, item: string) => req<GenerateJob>('/api/project/generate' + qi(id, item)),
  fullres: (id: string, item: string, variant: string) => req<FullRes>('/api/project/fullres' + qi(id, item) + '&variant=' + encodeURIComponent(variant)),
  rules: (id: string) => req<Rules>('/api/project/rules' + q(id)),
  setRules: (id: string, rules: Rule[]) => req<Rules>('/api/project/rules' + q(id), send('PUT', { rules })),
  applyRules: (id: string) => req<ApplyJob>('/api/project/apply' + q(id), { method: 'POST' }),
  applyStatus: (id: string) => req<ApplyJob>('/api/project/apply' + q(id)),
  browseFolders: (folder: string | null) => req<BrowseFolder[]>('/api/browse/folders' + (folder ? '?folder=' + encodeURIComponent(folder) : '')),
  browsePictures: (folder: string) => req<BrowseListing>('/api/browse/pictures?folder=' + encodeURIComponent(folder)),
  browseAdd: (b: BrowseAdd) => req<BrowseAdded>('/api/browse/add', send('POST', b)),
  stopApply: (id: string) => req<ApplyJob>('/api/project/apply' + q(id), { method: 'DELETE' }),
}

/** URL of a picture: an item's variant with a transform, at most `px` on the long side. */
export function imageUrl(id: string, item: string, variant: string | null, transform = 'original', px = 1600) {
  return withBase('/api/project/image' + qi(id, item) + (variant ? '&variant=' + encodeURIComponent(variant) : '') + '&transform=' + transform + '&px=' + px)
}

/** URL of the loupe's crop: the `size` px square around the point (x, y in 0..1 of the picture) of a variant with a transform, enlarged `zoom` times. */
export function cropUrl(id: string, item: string, variant: string, transform: string, x: number, y: number, size: number, zoom: number) {
  return withBase('/api/project/crop' + qi(id, item) + '&variant=' + encodeURIComponent(variant) + '&transform=' + transform + `&x=${x.toFixed(4)}&y=${y.toFixed(4)}&size=${size}&zoom=${zoom}`)
}

/** A project item's grid thumbnail: its primary variant (a raw's embedded preview), from the cache the Browser fills too. */
export const itemThumbUrl = (id: string, item: string, variant: string, px = 320) => withBase('/api/project/thumb' + qi(id, item) + '&variant=' + encodeURIComponent(variant) + '&px=' + px)  // the variant is part of the address, so a new primary is a new picture to the browser's cache

/** A cached thumbnail of a picture in the Browser, from one of its sources (the server rounds `px` up to 160, 320 or 640). */
export const thumbUrl = (folder: string, stem: string, source: string, px: number) =>
  withBase('/api/browse/thumb?folder=' + encodeURIComponent(folder) + '&stem=' + encodeURIComponent(stem) + '&source=' + encodeURIComponent(source) + '&px=' + px)

/** Address of an arena picture: the pair's `left`/`right` are relative to the server's /arena/. */
export const arenaImageUrl = (rel: string) => withBase('/arena/' + rel)
/** The newest picture of a flagged version, or with `variant` the same picture's other version (the Lightroom final). */
export const alertImageUrl = (id: string, variant?: string) => withBase('/api/alerts/image/' + encodeURIComponent(id) + (variant ? '?variant=' + encodeURIComponent(variant) : ''))
/** A flagged picture, made small by the server. */
export const flaggedImageUrl = (path: string, px = 600) => withBase('/api/flagged/image?path=' + encodeURIComponent(path) + '&px=' + px)

/** Download address of a generated variant's file. */
export const generatedFileUrl = (id: string, item: string, variant: string) => withBase('/api/project/generated/file' + qi(id, item) + '&variant=' + encodeURIComponent(variant))

/** The value of one setting of a stage list (what the Advanced panel shows for the pipeline that will run). */
export const stageValue = (stages: Stage[] | undefined, stage: string, key: string) => stages?.find(s => s.stage === stage)?.[key]
export const hasStage = (stages: Stage[] | undefined, stage: string) => !!stages?.some(s => s.stage === stage)
