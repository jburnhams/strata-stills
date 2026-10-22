import { useEffect, useState } from 'react'
import { api, generatedFileUrl, imageUrl, TRANSFORMS, type TransformInfo, type Variant } from '../api'
import { href, useLoad } from '../route'
import { Crumbs, Err, btnQuiet, projectCrumbs } from './ui'
import GeneratePanel, { mb } from './GeneratePanel'
import Loupe, { pointIn, type Point } from './Loupe'

/** One sentence on what a transform did to the picture (or why it did nothing). */
export function describe(id: string, i: TransformInfo | null): string {
  if (!i) return 'Working…'
  if (id === 'original') return 'The picture as it comes from the file.'
  if (id === 'level') {
    if (!i.applied) return `Left as it is: ${i.reason ?? 'no change needed'}.` + (i.lines != null ? ` (${i.lines} lines, ${i.nadir ? 'straight-down shot' : 'oblique shot'})` : '')
    const a = i.angle ?? 0
    return `Turned ${Math.abs(a).toFixed(2)}° ${a > 0 ? 'anticlockwise' : 'clockwise'} (confidence ${(i.confidence ?? 0).toFixed(2)}, ${i.nadir ? 'straight-down shot: aligned to the dominant edges' : 'oblique shot'}).`
  }
  return ''
}

interface PictureProps { project: string; item: string; variant: string; transform: string; label: string; info: TransformInfo | null; point: Point | null; onPoint: (p: Point) => void; onToggle: () => void }

function Picture({ project, item, variant, transform, label, info, point, onPoint, onToggle }: PictureProps) {
  return (
    <figure className="flex-1 min-w-0">
      <div className="relative cursor-crosshair" data-testid={`pic-${transform}`} onMouseMove={e => { const p = pointIn(e, e.currentTarget); if (p) onPoint(p) }} onClick={onToggle}>
        <img alt={`${label} — ${item}`} draggable={false} className="w-full rounded-lg border border-stone-300 dark:border-stone-700 bg-stone-200 dark:bg-stone-800 select-none" src={imageUrl(project, item, variant, transform, 1600)} />
        {point && <span aria-hidden className="pointer-events-none absolute w-6 h-6 -ml-3 -mt-3 rounded-full border-2 border-white shadow-[0_0_0_1px_rgba(0,0,0,0.6)]" style={{ left: `${point.x * 100}%`, top: `${point.y * 100}%` }} />}
      </div>
      <figcaption className="text-xs text-stone-500 mt-1"><b>{label}.</b> {describe(transform, info)}</figcaption>
    </figure>
  )
}

export default function ItemPage({ project, item, source }: { project: string; item: string; source: string }) {
  const proj = useLoad(() => api.project(project), [project])
  const it = useLoad(() => api.item(project, item), [project, item])
  const siblings = useLoad(() => api.items(project, source), [project, source])
  const [variant, setVariant] = useState<string | null>(null), [transform, setTransform] = useState('level'), [compare, setCompare] = useState(true)
  const [infos, setInfos] = useState<Record<string, TransformInfo | null>>({}), [err, setErr] = useState<string | null>(null)
  const [point, setPoint] = useState<Point | null>(null), [locked, setLocked] = useState(false)       // the loupe's point: follows the pointer over the picture until a click holds it
  const v = variant ?? it.data?.primary ?? null
  const pic = { point, onPoint: (p: Point) => { if (!locked) setPoint(p) }, onToggle: () => setLocked(l => !l) }

  useEffect(() => { setVariant(null); setPoint(null); setLocked(false) }, [item])
  useEffect(() => {
    if (!v) return
    let live = true; setInfos({})
    for (const t of TRANSFORMS) api.transform(project, item, v, t.id).then(i => live && setInfos(o => ({ ...o, [t.id]: i })), e => live && setErr(e.message))
    return () => { live = false }
  }, [project, item, v])

  const list = siblings.data?.filter(i => i.kind !== 'video') ?? []
  const idx = list.findIndex(i => i.id === item), prev = idx > 0 ? list[idx - 1] : null, next = idx >= 0 && idx < list.length - 1 ? list[idx + 1] : null
  useEffect(() => {
    const k = (e: KeyboardEvent) => { if ((e.target as HTMLElement)?.tagName === 'INPUT') return; if (e.key === 'ArrowLeft' && prev) window.location.hash = href.item(project, prev.id); if (e.key === 'ArrowRight' && next) window.location.hash = href.item(project, next.id) }
    window.addEventListener('keydown', k); return () => window.removeEventListener('keydown', k)
  }, [prev, next, project])

  const choose = async (x: Variant) => { setErr(null); try { await api.setPrimary(project, item, x.id); it.reload() } catch (e) { setErr((e as Error).message) } }
  const data = it.data
  return (
    <section>
      <Crumbs items={[...projectCrumbs(project, proj.data?.name), { label: 'Contents', to: href.source(project, source) }, { label: data?.stem ?? item }]} />
      <Err msg={it.error || err} />
      {data && (
        <>
          <div className="flex items-baseline gap-3 flex-wrap mb-3">
            <h1 className="text-lg font-semibold font-mono">{data.stem}</h1>
            <span className="text-xs text-stone-500">{data.kind}{data.parent ? ` (from ${data.parent})` : ''}</span>
            <span className="ml-auto flex gap-2">
              {prev ? <a className={btnQuiet} href={href.item(project, prev.id)}>← Previous</a> : null}
              {next ? <a className={btnQuiet} href={href.item(project, next.id)}>Next →</a> : null}
            </span>
          </div>

          <h2 className="text-sm font-medium mb-1">Variants <span className="text-stone-500 font-normal">(the primary is what the processing starts from)</span></h2>
          <div role="radiogroup" aria-label="Variants" className="flex flex-wrap gap-2 mb-4">
            {data.variants.map(x => (
              <div key={x.id} className={`rounded-lg border px-3 py-2 text-sm ${x.id === v ? 'border-blue-500 bg-blue-50 dark:bg-blue-950' : 'border-stone-300 dark:border-stone-700'}`}>
                <button role="radio" aria-checked={x.id === v} className="font-medium" onClick={() => setVariant(x.id)}>{x.label}</button>
                {x.kind === 'generated'
                  ? <div className="text-xs text-stone-500 max-w-64">{x.summary}{x.quality ? ` (q${x.quality})` : ''}{x.bytes ? ` · ${mb(x.bytes)}` : ''}<br /><a className="text-blue-600 hover:underline" href={generatedFileUrl(project, item, x.id)}>Download</a></div>
                  : <div className="font-mono text-xs text-stone-500">{x.path.split(/[\\/]/).pop()}</div>}
                {x.id === data.primary
                  ? <span className="text-xs text-blue-600">★ primary</span>
                  : <button className={btnQuiet + ' mt-1'} onClick={() => choose(x)}>Make primary</button>}
              </div>
            ))}
            {data.variants.length === 0 && <p className="text-stone-500 text-sm">This item has no still picture (a video).</p>}
          </div>

          {v && (
            <>
              <div className="flex items-center gap-2 mb-3 flex-wrap" role="tablist" aria-label="Transforms">
                {TRANSFORMS.map(t => (
                  <button key={t.id} role="tab" aria-selected={t.id === transform} onClick={() => setTransform(t.id)}
                    className={`rounded-md px-3 py-1 text-sm border ${t.id === transform ? 'bg-blue-600 text-white border-blue-600' : 'border-stone-300 dark:border-stone-700'}`}>{t.label}{infos[t.id]?.applied ? ' ✓' : ''}</button>
                ))}
                <label className="ml-auto text-sm text-stone-500 flex items-center gap-2"><input type="checkbox" checked={compare} onChange={e => setCompare(e.target.checked)} />Side by side with original</label>
              </div>
              <div className="flex gap-3 flex-col lg:flex-row">
                {compare && transform !== 'original' && <Picture project={project} item={item} variant={v} transform="original" label="Original" info={infos.original ?? null} {...pic} />}
                <Picture project={project} item={item} variant={v} transform={transform} label={TRANSFORMS.find(t => t.id === transform)?.label ?? transform} info={infos[transform] ?? null} {...pic} />
              </div>
              <Loupe project={project} item={item} variants={data.variants} point={point} locked={locked} />
              <GeneratePanel project={project} item={item} variant={data.variants.find(x => x.id === v)?.kind === 'generated' ? null : v} onGenerated={id => { it.reload(); setVariant(id) }} onShow={id => setVariant(id)} />
            </>
          )}
        </>
      )}
    </section>
  )
}
