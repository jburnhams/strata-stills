import { useEffect, useState } from 'react'
import { Field, NativeSelect } from 'strata-react'
import { api, cropUrl, TRANSFORMS, type FullRes, type Variant } from '../api'

export interface Point { x: number; y: number }

/** The normalised position (0..1) of a mouse event inside an element, or null when the element has no size. */
export function pointIn(e: { clientX: number; clientY: number }, el: Element): Point | null {
  const r = el.getBoundingClientRect()
  if (!r.width || !r.height) return null
  return { x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)) }
}

/** The point is held back a moment, so sweeping the mouse over the picture asks the server for a crop about every tenth of a second, not for every pixel moved. */
function useSettled(p: Point | null, ms = 120): Point | null {
  const [s, setS] = useState(p)
  useEffect(() => { const t = window.setTimeout(() => setS(p), ms); return () => window.clearTimeout(t) }, [p?.x, p?.y]) // eslint-disable-line react-hooks/exhaustive-deps
  return s
}

/** Where a variant's full-resolution copy stands (it is made once, in the background: seconds for a JPEG, about a minute and a half for a raw): asks the server, which starts it, and asks again until it is ready. */
export function useFullRes(project: string, item: string, variant: string): FullRes {
  const [s, setS] = useState<FullRes>({ state: 'idle', text: '' })
  useEffect(() => {
    let live = true
    let t: number | undefined
    const ask = () => api.fullres(project, item, variant).then(r => { if (!live) return; setS(r); if (r.state === 'running' || r.state === 'idle') t = window.setTimeout(ask, 1500) }, e => { if (live) setS({ state: 'failed', text: String(e.message || e) }) })
    setS({ state: 'idle', text: '' }); ask()
    return () => { live = false; if (t) window.clearTimeout(t) }
  }, [project, item, variant])
  return s
}

function Cell({ project, item, variant, transform, label, point, size, zoom }: { project: string; item: string; variant: Variant; transform: string; label: string; point: Point; size: number; zoom: number }) {
  const fr = useFullRes(project, item, variant.id), side = size * zoom
  return (
    <figure className="grid shrink-0 gap-0.5">
      {fr.state === 'ready'
        ? <img alt={`${variant.label} ${label}`} width={side} height={side} className="rounded-control border border-border bg-muted" style={{ width: side, height: side }} src={cropUrl(project, item, variant.id, transform, point.x, point.y, size, zoom)} />
        : <div role="status" className="flex items-center justify-center rounded-control border border-dashed border-border p-2 text-center type-caption text-muted-foreground" style={{ width: side, height: side }}>
            {fr.state === 'failed' ? `Cannot make the full-size copy: ${fr.text}` : `Preparing the full-resolution view of ${variant.label}: ${fr.text || 'starting'}`}
          </div>}
      <figcaption className="type-caption text-muted-foreground">{variant.label} · {label}</figcaption>
    </figure>
  )
}

/**
 * The loupe: the same scene area, cut at FULL RESOLUTION from every variant with every transform, side by side: for judging sharpness, grain, colour, compression artefacts and what levelling or another variant does. At anything less than full resolution it would show nothing worth comparing.
 * The area is in pixels of the biggest variant (the raw), so a smaller variant (a 6016 px generated picture) shows the same part of the scene enlarged to match. It sticks to the bottom of the window so it stays in view while the picture above is hovered.
 */
export default function Loupe({ project, item, variants, point, locked }: { project: string; item: string; variants: Variant[]; point: Point | null; locked: boolean }) {
  const [size, setSize] = useState(160)
  const [zoom, setZoom] = useState(2)
  const p = useSettled(point)
  if (variants.length === 0) return null
  return (
    <section aria-label="Loupe" className="sticky bottom-0 z-10 grid gap-2 border-t border-border bg-background py-2">
      <div className="flex flex-wrap items-end gap-3">
        <div className="grid gap-0.5">
          <h2 className="type-label">Loupe (full resolution)</h2>
          <p className="type-caption text-muted-foreground">{p ? (locked ? 'Held: click the picture again to release.' : 'Following the pointer: click the picture to hold it.') : 'Move the pointer over the picture.'}</p>
        </div>
        <Field label="Loupe area" className="ml-auto"><NativeSelect size="sm" value={size} onChange={e => setSize(Number(e.target.value))}>{[100, 160, 240, 320].map(n => <option key={n} value={n}>{n} px</option>)}</NativeSelect></Field>
        <Field label="Loupe zoom"><NativeSelect size="sm" value={zoom} onChange={e => setZoom(Number(e.target.value))}>{[1, 2, 3, 4].map(n => <option key={n} value={n}>{n}×</option>)}</NativeSelect></Field>
      </div>
      <div className="flex gap-2 overflow-x-auto pb-1">
        {p && variants.flatMap(v => TRANSFORMS.map(t => <Cell key={v.id + t.id} project={project} item={item} variant={v} transform={t.id} label={t.label} point={p} size={size} zoom={zoom} />))}
      </div>
    </section>
  )
}
