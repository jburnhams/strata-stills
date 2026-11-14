// The app's addresses. Hash routing (React Router's createHashRouter): the server serves one page at its root, behind a reverse proxy at a path too, so the route lives after the '#':
// #/ (projects), #/p/<project> (its sources), #/p/<project>/s/<source> (contents of a source), #/p/<project>/i/<item> (one item), #/arena, #/alerts, #/flagged, #/browse?folder=<key> (the Browser).
// Every id is URI-encoded (project ids contain slashes); these are the paths given to React Router's Link and navigate, which add the '#'.
import { useParams } from 'react-router'

const enc = encodeURIComponent

export const href = {
  projects: () => '/',
  project: (p: string) => `/p/${enc(p)}`,
  source: (p: string, s: string) => `/p/${enc(p)}/s/${enc(s)}`,
  item: (p: string, i: string) => `/p/${enc(p)}/i/${enc(i)}`,
  arena: () => '/arena',
  alerts: () => '/alerts',
  flagged: () => '/flagged',
  browse: (folder?: string) => '/browse' + (folder ? '?folder=' + enc(folder) : ''),
}

/** The source folder an item belongs to: its id is `<source>~<stem>`. */
export const sourceOf = (item: string) => item.split('~')[0]

// React Router keeps an encoded slash encoded inside one segment (its decodePath), so a project id with slashes arrives as 'a%2Fb': put the slashes back.
const dec = (s: string | undefined) => (s ?? '').replace(/%2F/gi, '/')

/** The ids in the current route, decoded. */
export function useIds() {
  const p = useParams()
  return { project: dec(p.project), source: dec(p.source), item: dec(p.item) }
}
