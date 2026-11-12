// One question: "Did the nudity filter hide a picture it should not have?" Every picture at or above the threshold, blurred until asked (a Toggle per picture, so nothing is shown by surprise),
// with one answer each: allow it back into every output, or hide it again. A folder can be scanned so its pictures are judged and show up here.
import { useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { Alert, AsyncButton, DataView, EmptyState, Field, Input, PageHeader, Status, Thumbnail, ThumbnailSkeleton, Toggle, cn, format, useAsyncAction, useLoad } from 'strata-react'
import { api, flaggedImageUrl, type FlaggedPicture } from '../api'

function Card({ p }: { p: FlaggedPicture }) {
  const client = useQueryClient()
  const [shown, setShown] = useState(false)
  const name = p.path.split(/[\\/]/).pop() ?? p.path
  const decide = useAsyncAction(async () => { await api.decideFlagged(p.path, !p.allowed); await client.invalidateQueries({ queryKey: ['flagged'] }) })
  return (
    <li className="grid content-start gap-2 rounded-surface border border-border p-2">
      <div className={cn('overflow-hidden rounded-control', !shown && '[&_img]:blur-xl')}><Thumbnail src={flaggedImageUrl(p.path)} alt={shown ? name : `${name}, blurred`} aspect="4/3" fit="contain" missing="Picture not found" className="w-full" /></div>
      <div className="flex flex-wrap items-center gap-2"><span className="type-label">{p.cls}</span><span className="type-caption tabular-nums text-muted-foreground">{p.score}</span>{p.allowed && <Status tone="success">Allowed</Status>}</div>
      <span className="break-all font-mono type-caption text-muted-foreground">{p.path}</span>
      <div className="flex flex-wrap gap-2">
        <Toggle size="sm" label="Show" pressed={shown} onChange={setShown} />
        <AsyncButton size="sm" phase={decide.phase} error={decide.error} labels={p.allowed ? { idle: 'Hide again', pending: 'Hiding…', done: 'Hidden', error: 'Try again' } : { idle: 'Not nudity: allow', pending: 'Allowing…', done: 'Allowed', error: 'Try again' }} onPress={() => void decide.run()} />
      </div>
      {decide.error && <Alert tone="danger" announce>{decide.error}</Alert>}
    </li>
  )
}

export default function FlaggedPage() {
  const client = useQueryClient()
  const list = useLoad({ queryKey: ['flagged'], queryFn: api.flagged })
  const [folder, setFolder] = useState(''), [scanned, setScanned] = useState<string | null>(null)
  const scan = useAsyncAction(async () => { const r = await api.scanFlagged(folder.trim()); setScanned(`${format.count(r.kept, 'picture')} kept, ${r.flagged} flagged in all.`); await client.invalidateQueries({ queryKey: ['flagged'] }) },
    { validate: () => ({ folder: folder.trim() ? undefined : 'Type the folder to scan.' }) })
  const hidden = list.status === 'ready' ? list.data.items.filter(i => !i.allowed).length : 0
  const grid = 'grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4'
  return (
    <section className="grid gap-6">
      <PageHeader title="Flagged pictures" scope="Pictures the nudity filter keeps out of every output. They are blurred: press Show to look."
        finding={list.status === 'ready' ? { text: `${list.data.enabled ? '' : 'The filter is off. '}${format.count(list.data.items.length, 'picture')} at or above threshold ${list.data.threshold}; ${hidden} hidden from every output.` } : undefined} />
      <form className="max-w-prose" aria-label="Scan a folder" onSubmit={e => { e.preventDefault(); void scan.run() }}>
        <Field label="Scan a folder (and below) for more" hint={scanned ?? 'Every picture in it is judged, so the ones over the threshold show up here.'} error={scan.fieldErrors.folder ?? scan.error}
          action={<AsyncButton type="submit" phase={scan.phase} error={scan.error} labels={{ idle: 'Scan', pending: 'Scanning…', done: 'Scanned', error: 'Try again' }} />}>
          <Input value={folder} onChange={e => { setFolder(e.target.value); scan.clearFieldError('folder') }} />
        </Field>
      </form>
      <DataView state={list} isEmpty={d => d.items.length === 0} skeleton={<div className={grid}>{[0, 1, 2].map(i => <ThumbnailSkeleton key={i} aspect="4/3" />)}</div>}
        empty={<EmptyState title="Nothing flagged">No picture scanned so far is over the threshold.</EmptyState>}>
        {d => <ul aria-label="Flagged pictures" className={grid}>{d.items.map(p => <Card key={p.path} p={p} />)}</ul>}
      </DataView>
    </section>
  )
}
