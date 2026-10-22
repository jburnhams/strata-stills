// Typed client for the FastAPI server (src/stratastills/server/app.py). Same-origin. Project ids contain slashes, so they travel as query parameters.
export interface Source { path: string; added?: string }
export interface Project { id: string; name: string; created: string | null; sources: Source[]; n_sources: number }
export interface SourceSummary { id: string; path: string; added?: string; reachable: boolean; stills: number; panos: number; videos: number }
export type ItemKind = 'still' | 'pano' | 'video'
export interface Variant { id: string; label: string; path: string; kind: 'raw' | 'jpeg' | 'derived'; ext: string }
export interface Item { id: string; stem: string; kind: ItemKind; parent: string | null; primary: string | null; chosen: boolean; variants: Variant[]; n_sidecars: number; video: string | null }
export interface TransformInfo { applied: boolean; angle?: number; confidence?: number; lines?: number; max_deg?: number; reason?: string | null; nadir?: boolean; pitch?: number | null }
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
  transform: (id: string, item: string, variant: string, transform: string) => req<TransformInfo>('/api/project/transform' + q(id) + '&item=' + encodeURIComponent(item) + '&variant=' + encodeURIComponent(variant) + '&transform=' + transform),
}

/** URL of a picture: an item's variant with a transform, at most `px` on the long side. */
export function imageUrl(id: string, item: string, variant: string | null, transform = 'original', px = 1600) {
  return '/api/project/image' + q(id) + '&item=' + encodeURIComponent(item) + (variant ? '&variant=' + encodeURIComponent(variant) : '') + '&transform=' + transform + '&px=' + px
}
