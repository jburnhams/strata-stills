// One question: "Is this picture right, and which version of it is best?" The picture (original beside a transform), the loupe at full resolution under it,
// every variant in a table (pick what the processing starts from), a close comparison of the variants with one zoom (strata-react CompareGrid), and Generate.
// Previous and next are ← and →, from one shortcut list shown in the legend (useHotkeys + KeysLegend: never while typing).
import { useCallback, useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router'
import { useQueryClient } from '@tanstack/react-query'
import { Alert, Breadcrumb, CompareGrid, DataTable, DataView, Disclosure, EmptyState, Field, KeysLegend, NativeSelect, PageHeader, RowAction, Skeleton, Status, Switch, ToggleGroup, buttonVariants, useAsyncAction, useHotkeys, useLoad, type Column, type Shortcut } from 'strata-react'
import { api, generatedFileUrl, imageUrl, TRANSFORMS, type Item, type TransformInfo, type Variant } from '../api'
import { href, sourceOf, useIds } from '../route'
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

interface PictureProps { project: string; item: string; variant: string; transform: string; label: string; point: Point | null; onPoint: (p: Point) => void; onToggle: () => void }

/** The picture the loupe follows: the pointer's place over it is the loupe's point, a click holds it. */
function Picture({ project, item, variant, transform, label, point, onPoint, onToggle }: PictureProps) {
  const info = useLoad({ queryKey: ['transform', project, item, variant, transform], queryFn: () => api.transform(project, item, variant, transform) })
  return (
    <figure className="grid min-w-0 flex-1 content-start gap-1.5">
      <div className="relative cursor-crosshair" data-testid={`pic-${transform}`} onMouseMove={e => { const p = pointIn(e, e.currentTarget); if (p) onPoint(p) }} onClick={onToggle}>
        <img alt={`${label} — ${item}`} draggable={false} className="w-full select-none rounded-surface border border-border bg-muted" src={imageUrl(project, item, variant, transform, 1600)} />
        {point && <span aria-hidden className="pointer-events-none absolute -mt-3 -ml-3 size-6 rounded-pill border-2 border-background outline outline-foreground" style={{ left: `${point.x * 100}%`, top: `${point.y * 100}%` }} />}
      </div>
      <figcaption className="type-caption text-muted-foreground"><span className="type-label text-foreground">{label}.</span> {describe(transform, info.status === 'ready' ? info.data : null)}</figcaption>
    </figure>
  )
}

function VariantTable({ project, item, data, onShow }: { project: string; item: string; data: Item; onShow: (id: string) => void }) {
  const client = useQueryClient()
  const choose = useAsyncAction(async (v: Variant) => { client.setQueryData(['item', project, item], await api.setPrimary(project, item, v.id)); await client.invalidateQueries({ queryKey: ['items', project, sourceOf(item)] }) })
  const columns: Column<Variant>[] = [
    { key: 'label', header: 'Variant', value: v => v.label, cell: v => <span className="type-label">{v.label}</span> },
    { key: 'what', header: 'File or settings', truncate: true, sort: false, cell: v => (v.kind === 'generated' ? `${v.summary ?? ''}${v.quality ? ` (q${v.quality})` : ''}` : <span className="font-mono">{v.path.split(/[\\/]/).pop()}</span>) },
    { key: 'bytes', header: 'Size', numeric: true, value: v => v.bytes ?? null, cell: v => (v.bytes ? mb(v.bytes) : null) },
    { key: 'primary', header: 'Primary', sort: false, cell: v => (v.id === data.primary ? <Status tone="info">Primary</Status> : <RowAction label="Make primary" subject={v.label} onPress={() => void choose.run(v)} />) },
    { key: 'actions', header: 'Actions', hideHeader: true, sort: false, cell: v => <span className="flex gap-1"><RowAction label="Show" subject={v.label} onPress={() => onShow(v.id)} />{v.kind === 'generated' && <a className={buttonVariants({ variant: 'ghost', size: 'sm' })} href={generatedFileUrl(project, item, v.id)} aria-label={`Download ${v.label}`}>Download</a>}</span> },
  ]
  return (
    <div className="grid gap-2">
      <DataTable caption="Variants: the primary is what the processing starts from" columns={columns} rows={data.variants} rowKey={v => v.id} />
      {choose.error && <Alert tone="danger" announce>{choose.error}</Alert>}
    </div>
  )
}

export default function ItemPage() {
  const { project, item } = useIds(), source = sourceOf(item)
  const navigate = useNavigate()
  const proj = useLoad({ queryKey: ['project', project], queryFn: () => api.project(project) })
  const it = useLoad({ queryKey: ['item', project, item], queryFn: () => api.item(project, item) })
  const siblings = useLoad({ queryKey: ['items', project, source], queryFn: () => api.items(project, source) })
  const [variant, setVariant] = useState<string | null>(null), [transform, setTransform] = useState('level'), [compare, setCompare] = useState(true)
  const [point, setPoint] = useState<Point | null>(null), [locked, setLocked] = useState(false) // the loupe's point: follows the pointer over the picture until a click holds it
  useEffect(() => { setVariant(null); setPoint(null); setLocked(false) }, [item])
  const data = it.status === 'ready' ? it.data : null
  const v = variant ?? data?.primary ?? null
  const pic = { point, onPoint: (p: Point) => { if (!locked) setPoint(p) }, onToggle: () => setLocked(l => !l) }

  const list = siblings.status === 'ready' ? siblings.data.filter(i => i.kind !== 'video') : []
  const idx = list.findIndex(i => i.id === item), prev = idx > 0 ? list[idx - 1] : null, next = idx >= 0 && idx < list.length - 1 ? list[idx + 1] : null
  const shortcuts: Shortcut[] = [
    { keys: ['ArrowLeft'], label: 'Previous picture', run: () => prev && navigate(href.item(project, prev.id)), disabled: !prev },
    { keys: ['ArrowRight'], label: 'Next picture', run: () => next && navigate(href.item(project, next.id)), disabled: !next },
  ]
  useHotkeys(shortcuts)
  const onGenerated = useCallback((id: string) => setVariant(id), [])
  const name = proj.status === 'ready' ? proj.data.name : project
  const label = TRANSFORMS.find(t => t.id === transform)?.label ?? transform

  return (
    <section className="grid gap-6">
      <Breadcrumb items={[{ label: 'Projects', to: href.projects() }, { label: name, to: href.project(project) }, { label: 'Contents', to: href.source(project, source) }, { label: data?.stem ?? item }]} />
      <DataView state={it} isEmpty={() => false} skeleton={<Skeleton className="h-96" />} empty={null}>
        {d => (
          <div className="grid gap-6">
            <PageHeader title={d.stem} scope={`${d.kind === 'pano' ? 'Panorama' : d.kind === 'video' ? 'Video' : 'Still'}${d.parent ? ` (from ${d.parent})` : ''}`}
              actions={<>
                {prev && <Link className={buttonVariants({ size: 'sm' })} to={href.item(project, prev.id)}>← Previous</Link>}
                {next && <Link className={buttonVariants({ size: 'sm' })} to={href.item(project, next.id)}>Next →</Link>}
              </>} />
            {d.variants.length === 0
              ? <EmptyState title="No still picture">This item is a video; videos are not processed yet.</EmptyState>
              : <VariantTable project={project} item={item} data={d} onShow={setVariant} />}
            {v && (
              <>
                <div className="flex flex-wrap items-end gap-4">
                  <Field label="Variant shown"><NativeSelect value={v} onChange={e => setVariant(e.target.value)}>{d.variants.map(x => <option key={x.id} value={x.id}>{x.label}</option>)}</NativeSelect></Field>
                  <ToggleGroup label="Transform" options={TRANSFORMS.map(t => ({ value: t.id, label: t.label }))} value={transform} onChange={setTransform} />
                  <Switch label="Side by side with the original" on={compare} onChange={setCompare} className="ml-auto" />
                </div>
                <div className="flex flex-col gap-3 lg:flex-row">
                  {compare && transform !== 'original' && <Picture project={project} item={item} variant={v} transform="original" label="Original" {...pic} />}
                  <Picture project={project} item={item} variant={v} transform={transform} label={label} {...pic} />
                </div>
                <Loupe project={project} item={item} variants={d.variants} point={point} locked={locked} />
                {d.variants.length > 1 && (
                  <Disclosure summary={`Compare all ${d.variants.length} variants (${label.toLowerCase()}), zoomed together`}>
                    <CompareGrid label={`Variants of ${d.stem}`} blind={false} missing="This variant could not be drawn" columns={d.variants.length > 3 ? 4 : d.variants.length === 3 ? 3 : 2} keys={false}
                      items={d.variants.map(x => ({ id: x.id, label: x.label, note: x.kind === 'generated' ? x.summary : x.path.split(/[\\/]/).pop(), media: { kind: 'image' as const, src: imageUrl(project, item, x.id, transform, 1600), alt: `${x.label} of ${d.stem}` } }))} />
                  </Disclosure>
                )}
                <GeneratePanel project={project} item={item} variant={d.variants.find(x => x.id === v)?.kind === 'generated' ? null : v} onGenerated={onGenerated} onShow={setVariant} />
              </>
            )}
            <KeysLegend shortcuts={shortcuts} />
          </div>
        )}
      </DataView>
    </section>
  )
}
