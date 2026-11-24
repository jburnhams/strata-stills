// The Browser's view logic, apart from React so the unit tests run it in node: the groups and the thumbnail sources of a folder's pictures (strata-react's useViewOptions sorts and groups by them).
import type { BrowsePicture } from './api'

export type GroupKey = 'hour' | 'day' | 'type' | 'project'
export type ShowKey = 'best' | 'camera'
export const GROUPS: { key: GroupKey; label: string }[] = [{ key: 'hour', label: 'Hour' }, { key: 'day', label: 'Day' }, { key: 'type', label: 'Type' }, { key: 'project', label: 'Project' }]

/** When a picture was taken, for the "Date taken" sort (a picture with no time sorts last either way). */
export const takenAt = (p: BrowsePicture) => (p.taken ? new Date(p.taken) : null)

/** What kind of files a picture has, in a few words: "Raw + JPEG", "Raw", "JPEG", "Panorama". */
export function typeOf(p: BrowsePicture): string {
  if (p.kind === 'pano') return 'Panorama'
  const ext = new Set(p.files.map(f => f.slice(f.lastIndexOf('.')).toLowerCase()))
  const raw = [...ext].some(e => ['.dng', '.arw', '.cr2', '.cr3', '.nef', '.orf', '.raf', '.rw2'].includes(e))
  const jpg = ext.has('.jpg') || ext.has('.jpeg')
  return raw && jpg ? 'Raw + JPEG' : raw ? 'Raw' : jpg ? 'JPEG' : 'Other'
}

const pad = (n: number) => String(n).padStart(2, '0')

/** The group a picture is in, in words (the heading of its group): the hour or day it was taken (file times, in the viewer's time zone), its type, or its projects. */
export function groupOf(p: BrowsePicture, group: GroupKey): string {
  if (group === 'type') return typeOf(p)
  if (group === 'project') return p.projects.length ? 'In ' + p.projects.join(', ') : 'In no project'
  if (!p.taken) return 'No time'
  const d = new Date(p.taken), day = d.toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' })
  return group === 'day' ? day : `${day}, ${pad(d.getHours())}:00 to ${pad((d.getHours() + 1) % 24)}:00`
}

/** The server's thumbnail size for a tile this wide: twice its width for a 2x screen, rounded up to the sizes the cache keeps (160, 320, 640). */
export const thumbPx = (tile: number) => (tile * 2 <= 160 ? 160 : tile * 2 <= 320 ? 320 : 640)

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
