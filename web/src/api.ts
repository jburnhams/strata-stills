// Typed client for the FastAPI server (src/stratastills/server/app.py). Same-origin. Project ids contain slashes, so they travel as query parameters.
export interface Source { path: string; added?: string }
export interface Project { id: string; name: string; created: string | null; sources: Source[]; n_sources: number }
export interface SourceSummary { id: string; path: string; added?: string; reachable: boolean; stills: number; panos: number; videos: number }
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
export interface Options { size?: string; format?: 'jpeg' | 'tiff'; quality?: number | 'auto'; level?: boolean; sharpen?: boolean; sharpen_amount?: number; highlight_contrast?: number }
export interface GenerateBody { variant?: string | null; preset?: string | null; options?: Options | null; stages?: Stage[] | null }
export interface GenerateCheck { key: string; exists: boolean; variant: string | null; stages: Stage[]; summary: string }
export interface GenerateRecord { key: string; file: string; output: [number, number]; bytes: number; format: string; quality: number | null; summary: string; steps: string[]; existing?: boolean }
export interface GenerateJob { state: 'idle' | 'running' | 'done' | 'failed' | 'exists'; text?: string; result?: GenerateRecord | null; error?: string | null; variant?: string }
export interface FullRes { state: 'ready' | 'running' | 'failed' | 'idle'; text: string }
export interface Rule { source: string; pipeline: string; variant?: string }
export interface Rules { rules: Rule[]; pending: number }
export interface ApplyJob { state: 'idle' | 'running' | 'done' | 'stopped' | 'failed'; done?: number; total?: number; text?: string; failed?: [string, string][] }
export interface Transform { id: string; label: string }

/** The transforms the server offers on a picture, in order (transforms.py ORDER). */
export const TRANSFORMS: Transform[] = [{ id: 'original', label: 'Original' }, { id: 'level', label: 'Levelled' }]

export const q = (id: string) => '?id=' + encodeURIComponent(id)
const qi = (id: string, item: string) => q(id) + '&item=' + encodeURIComponent(item)

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init)
  const d = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error((d as { detail?: string }).detail || r.statusText)
  return d as T
}
const send = (method: string, body: unknown): RequestInit => ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

export const api = {
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
  stopApply: (id: string) => req<ApplyJob>('/api/project/apply' + q(id), { method: 'DELETE' }),
}

/** URL of a picture: an item's variant with a transform, at most `px` on the long side. */
export function imageUrl(id: string, item: string, variant: string | null, transform = 'original', px = 1600) {
  return '/api/project/image' + qi(id, item) + (variant ? '&variant=' + encodeURIComponent(variant) : '') + '&transform=' + transform + '&px=' + px
}

/** URL of the loupe's crop: the `size` px square around the point (x, y in 0..1 of the picture) of a variant with a transform, enlarged `zoom` times. */
export function cropUrl(id: string, item: string, variant: string, transform: string, x: number, y: number, size: number, zoom: number) {
  return '/api/project/crop' + qi(id, item) + '&variant=' + encodeURIComponent(variant) + '&transform=' + transform + `&x=${x.toFixed(4)}&y=${y.toFixed(4)}&size=${size}&zoom=${zoom}`
}

/** Download address of a generated variant's file. */
export const generatedFileUrl = (id: string, item: string, variant: string) => '/api/project/generated/file' + qi(id, item) + '&variant=' + encodeURIComponent(variant)

/** The value of one setting of a stage list (what the Advanced panel shows for the pipeline that will run). */
export const stageValue = (stages: Stage[] | undefined, stage: string, key: string) => stages?.find(s => s.stage === stage)?.[key]
export const hasStage = (stages: Stage[] | undefined, stage: string) => !!stages?.some(s => s.stage === stage)
