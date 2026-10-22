// Hash routing: #/ (projects), #/p/<project> (its sources), #/p/<project>/s/<source> (contents of a source), #/p/<project>/i/<item> (one item). Every id is URI-encoded (project ids contain slashes).
import { useEffect, useState } from 'react'

export type Route =
  | { page: 'projects' }
  | { page: 'project'; project: string }
  | { page: 'source'; project: string; source: string }
  | { page: 'item'; project: string; item: string; source: string }

const enc = encodeURIComponent

export function parse(hash: string): Route {
  const parts = hash.replace(/^#\/?/, '').split('/').filter(Boolean).map(decodeURIComponent)
  if (parts[0] === 'p' && parts[1]) {
    if (parts[2] === 's' && parts[3]) return { page: 'source', project: parts[1], source: parts[3] }
    if (parts[2] === 'i' && parts[3]) return { page: 'item', project: parts[1], item: parts[3], source: parts[3].split('~')[0] }
    return { page: 'project', project: parts[1] }
  }
  return { page: 'projects' }
}

export const href = {
  projects: () => '#/',
  project: (p: string) => `#/p/${enc(p)}`,
  source: (p: string, s: string) => `#/p/${enc(p)}/s/${enc(s)}`,
  item: (p: string, i: string) => `#/p/${enc(p)}/i/${enc(i)}`,
}

export function useRoute(): Route {
  const [r, setR] = useState(() => parse(window.location.hash))
  useEffect(() => { const f = () => setR(parse(window.location.hash)); window.addEventListener('hashchange', f); return () => window.removeEventListener('hashchange', f) }, [])
  return r
}

/** Run an async loader when `deps` change; gives {data, error, reload}. A newer call wins over an older one still in flight. */
export function useLoad<T>(load: () => Promise<T>, deps: unknown[]) {
  const [data, setData] = useState<T | null>(null), [error, setError] = useState<string | null>(null), [n, setN] = useState(0)
  useEffect(() => {
    let live = true; setError(null)
    load().then(d => { if (live) setData(d) }, e => { if (live) setError(String(e.message || e)) })
    return () => { live = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, n])
  return { data, error, reload: () => setN(x => x + 1) }
}
