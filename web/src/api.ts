// Typed client for the FastAPI server (src/stratastills/server/app.py). Same-origin. Project ids contain slashes, so they travel as query parameters.
export interface Source { path: string; added?: string }
export interface Project { id: string; name: string; created: string | null; sources: Source[]; n_sources: number }
export interface SourceSummary { id: string; path: string; added?: string; reachable: boolean; stills: number; panos: number; videos: number }
export type ItemKind = 'still' | 'pano' | 'video'
export interface Variant { id: string; label: string; path: string; kind: 'raw' | 'jpeg' | 'derived'; ext: string }
export interface Item { id: string; stem: string; kind: ItemKind; parent: string | null; primary: string | null; chosen: boolean; variants: Variant[]; n_sidecars: number; video: string | null }
export interface TransformInfo { applied: boolean; angle?: number; confidence?: number; lines?: number; max_deg?: number; reason?: string | null; nadir?: boolean; pitch?: number | null }
export interface Detail { ok: boolean; reason?: string; native?: [number, number]; recommended?: number | null; f50?: number; f30?: number; f20?: number; scale50?: number; scale30?: number; scale20?: number; alpha?: number }
export interface ExportOptions { variant?: string | null; size: string; format: 'jpeg' | 'tiff'; level: boolean; sharpen: boolean; sharpen_amount?: number; highlight_contrast?: number; quality?: number }
export interface ExportRecord { file: string; name: string; bytes: number; output: [number, number]; native: [number, number]; format: string; created: string; stem: string; variant: string; steps: string[]; size: { long_side: number; why: string }; level?: { applied: boolean; angle?: number; reason?: string } }
export interface ExportJob { state: 'idle' | 'running' | 'done' | 'failed'; text?: string; result?: ExportRecord | null; error?: string | null }
export interface Transform { id: string; label: string }

/** The transforms the server offers on a picture, in order (transforms.py ORDER). */
export const TRANSFORMS: Transform[] = [{ id: 'original', label: 'Original' }, { id: 'level', label: 'Levelled' }]

export const q = (id: string) => '?id=' + encodeURIComponent(id)

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
  item: (id: string, item: string) => req<Item>('/api/project/item' + q(id) + '&item=' + encodeURIComponent(item)),
  setPrimary: (id: string, item: string, variant: string) => req<Item>('/api/project/primary' + q(id) + '&item=' + encodeURIComponent(item), send('PUT', { variant })),
  detail: (id: string, item: string, variant: string) => req<Detail>('/api/project/detail' + q(id) + '&item=' + encodeURIComponent(item) + '&variant=' + encodeURIComponent(variant)),
  startExport: (id: string, item: string, o: ExportOptions) => req<ExportJob>('/api/project/export' + q(id) + '&item=' + encodeURIComponent(item), send('POST', o)),
  exportStatus: (id: string, item: string) => req<ExportJob>('/api/project/export' + q(id) + '&item=' + encodeURIComponent(item)),
  exports: (id: string, stem: string) => req<ExportRecord[]>('/api/project/exports' + q(id) + '&stem=' + encodeURIComponent(stem)),
  transform: (id: string, item: string, variant: string, transform: string) => req<TransformInfo>('/api/project/transform' + q(id) + '&item=' + encodeURIComponent(item) + '&variant=' + encodeURIComponent(variant) + '&transform=' + transform),
}

/** URL of a picture: an item's variant with a transform, at most `px` on the long side. */
export function imageUrl(id: string, item: string, variant: string | null, transform = 'original', px = 1600) {
  return '/api/project/image' + q(id) + '&item=' + encodeURIComponent(item) + (variant ? '&variant=' + encodeURIComponent(variant) : '') + '&transform=' + transform + '&px=' + px
}

/** URL of the loupe's crop: the `size` px square around the point (x, y in 0..1 of the picture) of a variant with a transform, enlarged `zoom` times. */
export function cropUrl(id: string, item: string, variant: string, transform: string, x: number, y: number, size: number, zoom: number) {
  return '/api/project/crop' + q(id) + '&item=' + encodeURIComponent(item) + '&variant=' + encodeURIComponent(variant) + '&transform=' + transform + `&x=${x.toFixed(4)}&y=${y.toFixed(4)}&size=${size}&zoom=${zoom}`
}

export const exportFileUrl =(id: string, name: string) => '/api/project/export/file' + q(id) + '&name=' + encodeURIComponent(name)
