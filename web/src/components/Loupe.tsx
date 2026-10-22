import { useEffect, useState } from 'react'
import { cropUrl, TRANSFORMS, type Variant } from '../api'

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

const sel = 'rounded-md border border-stone-300 dark:border-stone-700 bg-transparent px-1 py-0.5 text-xs'

/** The loupe: the same place in the picture, cut from every variant with every transform, side by side at one scale: for judging sharpness, noise, colour and what levelling or another variant does.
 *  It sticks to the bottom of the window so it stays in view while the picture above is hovered. */
export default function Loupe({ project, item, variants, point, locked }: { project: string; item: string; variants: Variant[]; point: Point | null; locked: boolean }) {
  const [size, setSize] = useState(100), [zoom, setZoom] = useState(3), p = useSettled(point)
  if (variants.length === 0) return null
  return (
    <section aria-label="Loupe" className="sticky bottom-0 z-10 mt-4 -mx-4 px-4 py-2 bg-stone-100/95 dark:bg-stone-950/95 border-t border-stone-300 dark:border-stone-700 backdrop-blur">
      <div className="flex items-center gap-3 text-xs text-stone-500 mb-1 flex-wrap">
        <b className="text-stone-800 dark:text-stone-200 text-sm">Loupe</b>
        <span>{p ? (locked ? 'Held: click the picture again to release.' : 'Following the pointer: click the picture to hold it.') : 'Move the pointer over the picture.'}</span>
        <label className="ml-auto flex items-center gap-1">Area
          <select aria-label="Loupe area" className={sel} value={size} onChange={e => setSize(Number(e.target.value))}>{[64, 100, 160, 240].map(n => <option key={n} value={n}>{n} px</option>)}</select>
        </label>
        <label className="flex items-center gap-1">Zoom
          <select aria-label="Loupe zoom" className={sel} value={zoom} onChange={e => setZoom(Number(e.target.value))}>{[2, 3, 4, 5].map(n => <option key={n} value={n}>{n}×</option>)}</select>
        </label>
      </div>
      <div className="flex gap-2 overflow-x-auto pb-1">
        {p && variants.flatMap(v => TRANSFORMS.map(t => (
          <figure key={v.id + t.id} className="shrink-0">
            <img alt={`${v.label} ${t.label}`} width={size * zoom} height={size * zoom} className="rounded border border-stone-300 dark:border-stone-700 bg-stone-200 dark:bg-stone-800" style={{ width: size * zoom, height: size * zoom }} src={cropUrl(project, item, v.id, t.id, p.x, p.y, size, zoom)} />
            <figcaption className="text-xs text-stone-500 mt-0.5">{v.label} · {t.label}</figcaption>
          </figure>
        )))}
      </div>
    </section>
  )
}
