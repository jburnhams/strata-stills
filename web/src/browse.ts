// The Browser's view logic, apart from React so the unit tests run it in node: the order, the groups and the thumbnail sources of a folder's pictures.
import type { BrowsePicture } from './api'

export type SortKey = 'name' | 'name-desc' | 'time' | 'time-desc'
export type GroupKey = 'none' | 'hour' | 'day' | 'type' | 'project'
export type ShowKey = 'best' | 'camera'
export const SORTS: { value: SortKey; label: string }[] = [
  { value: 'name', label: 'Name, A to Z' }, { value: 'name-desc', label: 'Name, Z to A' }, { value: 'time', label: 'Oldest first' }, { value: 'time-desc', label: 'Newest first' },
]
export const GROUPS: { value: GroupKey; label: string }[] = [
  { value: 'none', label: 'No groups' }, { value: 'hour', label: 'By hour' }, { value: 'day', label: 'By day' }, { value: 'type', label: 'By type' }, { value: 'project', label: 'By project' },
]

const natural = new Intl.Collator('en', { numeric: true, sensitivity: 'base' })

/** The pictures in the order asked for; names compare naturally (DSC9 before DSC10), and a picture with no time goes last. */
export function sortPictures(rows: readonly BrowsePicture[], sort: SortKey): BrowsePicture[] {
  const byName = (a: BrowsePicture, b: BrowsePicture) => natural.compare(a.stem, b.stem)
  const out = [...rows]
  if (sort === 'name' || sort === 'name-desc') return out.sort((a, b) => (sort === 'name' ? byName(a, b) : byName(b, a)))
  const sign = sort === 'time' ? 1 : -1
  return out.sort((a, b) => {
    if (!a.taken || !b.taken) return Number(!a.taken) - Number(!b.taken) || byName(a, b)
    return sign * (Date.parse(a.taken) - Date.parse(b.taken)) || byName(a, b)
  })
}

/** What kind of files a picture has, in a few words: "Raw + JPEG", "Raw", "JPEG", "Panorama". */
export function typeOf(p: BrowsePicture): string {
  if (p.kind === 'pano') return 'Panorama'
  const ext = new Set(p.files.map(f => f.slice(f.lastIndexOf('.')).toLowerCase()))
  const raw = [...ext].some(e => ['.dng', '.arw', '.cr2', '.cr3', '.nef', '.orf', '.raf', '.rw2'].includes(e))
  const jpg = ext.has('.jpg') || ext.has('.jpeg')
  return raw && jpg ? 'Raw + JPEG' : raw ? 'Raw' : jpg ? 'JPEG' : 'Other'
}

const pad = (n: number) => String(n).padStart(2, '0')

/** The groups of a sorted list, in the order their first picture comes: [{key, label, rows}]; one group of everything for 'none'. Times are the file times, in the viewer's time zone. */
export function groupPictures(rows: readonly BrowsePicture[], group: GroupKey): { key: string; label: string; rows: BrowsePicture[] }[] {
  if (group === 'none') return [{ key: 'all', label: 'All pictures', rows: [...rows] }]
  const of = (p: BrowsePicture): [string, string] => {
    if (group === 'type') return [typeOf(p), typeOf(p)]
    if (group === 'project') return p.projects.length ? [p.projects.join(','), 'In ' + p.projects.join(', ')] : ['', 'In no project']
    if (!p.taken) return ['?', 'No time']
    const d = new Date(p.taken), day = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
    const dayLabel = d.toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' })
    return group === 'day' ? [day, dayLabel] : [`${day} ${pad(d.getHours())}`, `${dayLabel}, ${pad(d.getHours())}:00 to ${pad((d.getHours() + 1) % 24)}:00`]
  }
  const out = new Map<string, { key: string; label: string; rows: BrowsePicture[] }>()
  for (const p of rows) {
    const [key, label] = of(p)
    if (!out.has(key)) out.set(key, { key, label, rows: [] })
    out.get(key)!.rows.push(p)
  }
  return [...out.values()]
}

/** The sources to try for a thumbnail, in order: every one the server named (best first), or with 'camera' the camera's own picture first (the raw's preview, the JPEG), develops after. */
export function thumbSources(p: BrowsePicture, show: ShowKey): string[] {
  const ids = p.sources.map(s => s.id)
  if (show === 'best') return ids
  const camera: string[] = ids.filter(i => i === 'embedded' || i === 'jpeg')
  return [...camera, ...ids.filter(i => !camera.includes(i))]
}

/** The first source's label: what the thumbnail shows when it loads ("Lightroom export", "Camera preview"). */
export const shownFrom = (p: BrowsePicture, show: ShowKey) => p.sources.find(s => s.id === thumbSources(p, show)[0])?.label ?? 'No picture'

/** The parent keys of a folder key, top first, which the tree must have open to show it: raw/2019/2019-03-02 -> [raw, raw/2019]. */
export const ancestorKeys = (key: string) => key.split('/').slice(0, -1).map((_, i, a) => a.slice(0, i + 1).join('/'))
