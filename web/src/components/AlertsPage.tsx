// One question: "Which flagged pictures still need fixing, and is the fix right?" The arena's flags are a fix queue: open → investigating → fixed (we think so) → confirmed (Jonathan agrees),
// or wontfix. The tabs split the queue by what to do next and live in the URL (?show=), as does the open comparison (?open=), so a link lands on the same flag.
// Each flag shows the newest picture of the flagged version beside the Lightroom final, and opens into strata-react's Compare (not blind: here knowing which is which is the point).
import { useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { AsyncButton, Compare, DataView, Disclosure, EmptyState, Field, NativeSelect, PageHeader, PropertyList, Skeleton, Status, Tabs, Textarea, Thumbnail, format, useAsyncAction, useExpanded, useLoad, type CompareMode, type Tone } from 'strata-react'
import { ALERT_STATUSES, alertImageUrl, api, type AlertStatus, type ArenaAlert } from '../api'

export const STATUS: Record<AlertStatus, { word: string; tone: Tone }> = {
  open: { word: 'Open', tone: 'danger' },
  investigating: { word: 'Investigating', tone: 'warning' },
  fixed: { word: 'Fixed, to confirm', tone: 'info' },
  confirmed: { word: 'Confirmed', tone: 'success' },
  wontfix: { word: "Won't fix", tone: 'neutral' },
}

/** The tabs: what is left to do on a flag decides where it is listed. */
export const VIEWS = [
  { value: 'todo', label: 'To fix', has: (s: AlertStatus) => s === 'open' || s === 'investigating' },
  { value: 'check', label: 'To confirm', has: (s: AlertStatus) => s === 'fixed' },
  { value: 'closed', label: 'Closed', has: (s: AlertStatus) => s === 'confirmed' || s === 'wontfix' },
  { value: 'all', label: 'All', has: () => true },
] as const

function AlertCard({ a, expanded }: { a: ArenaAlert; expanded: ReturnType<typeof useExpanded> }) {
  const client = useQueryClient()
  const [note, setNote] = useState(a.note ?? ''), [status, setStatus] = useState<AlertStatus>(a.status)
  const [mode, setMode] = useState<CompareMode>('side')
  const save = useAsyncAction(async () => { await api.setAlert(a.id, status, note); await client.invalidateQueries({ queryKey: ['alerts'] }) })
  const newest = alertImageUrl(a.id), lightroom = alertImageUrl(a.id, 'lightroom')
  return (
    <article aria-labelledby={`alert-${a.id}`} className="grid gap-4 rounded-surface border border-border p-4 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
      <div className="grid content-start gap-2">
        <div className="grid grid-cols-2 gap-2">
          <figure className="grid gap-1"><Thumbnail src={newest} alt={`${a.variant} of ${a.pic}, newest`} aspect="3/2" fit="contain" missing="Gone from the pool" className="w-full rounded-control" /><figcaption className="type-caption text-muted-foreground">{a.variant} (newest)</figcaption></figure>
          <figure className="grid gap-1"><Thumbnail src={lightroom} alt={`Lightroom final of ${a.pic}`} aspect="3/2" fit="contain" missing="No Lightroom final" className="w-full rounded-control" /><figcaption className="type-caption text-muted-foreground">Lightroom final</figcaption></figure>
        </div>
        <Disclosure summary="Compare closely" open={expanded.isOpen(a.id)} onOpenChange={o => expanded.setOpen(a.id, o)}>
          <Compare label={`${a.variant} against the Lightroom final of ${a.pic}`} blind={false} missing="This picture could not be loaded" mode={mode} onModeChange={setMode} keys={false} className="pt-2"
            a={{ id: a.variant, label: a.variant, note: 'newest', media: { kind: 'image', src: newest, alt: `${a.variant} of ${a.pic}, newest` } }}
            b={{ id: 'lightroom', label: 'Lightroom final', media: { kind: 'image', src: lightroom, alt: `Lightroom final of ${a.pic}` } }} />
        </Disclosure>
      </div>
      <div className="grid content-start gap-3">
        <div className="flex flex-wrap items-center gap-2"><h2 id={`alert-${a.id}`} className="type-heading break-all">{a.pic} / {a.variant}</h2><Status tone={STATUS[a.status].tone}>{STATUS[a.status].word}</Status></div>
        <PropertyList label="About this flag" items={[
          { term: 'When', value: format.dateTime(a.t * 1000) },
          { term: 'Shown', value: `${a.side}, against ${a.other}` },
          { term: 'Time on pair', value: (a.ms / 1000).toFixed(1), unit: 's' },
          { term: 'Votes', value: `${a.votes_on_pic} on the picture, ${a.votes_on_variant} involving this version` },
          { term: 'File', value: a.pool ? a.pool.file : 'Gone from the pool' },
          { term: 'Hash / key', value: a.pool ? `${a.pool.hash ?? ''} / ${a.pool.key ?? ''}` : null },
        ]} />
        <form className="grid gap-3" onSubmit={e => { e.preventDefault(); void save.run() }} aria-label={`Status of ${a.pic} / ${a.variant}`}>
          <Field label="What is wrong"><Textarea rows={2} value={note} placeholder="The reason" onChange={e => setNote(e.target.value)} /></Field>
          <Field label="Status" error={save.error} action={<AsyncButton type="submit" phase={save.phase} error={save.error} labels={{ idle: 'Save', pending: 'Saving…', done: 'Saved', error: 'Try again' }} />}>
            <NativeSelect value={status} onChange={e => setStatus(e.target.value as AlertStatus)}>{ALERT_STATUSES.map(s => <option key={s} value={s}>{STATUS[s].word}</option>)}</NativeSelect>
          </Field>
        </form>
      </div>
    </article>
  )
}

export default function AlertsPage() {
  const alerts = useLoad({ queryKey: ['alerts'], queryFn: api.alerts })
  const expanded = useExpanded({ param: 'open' })
  const all = alerts.status === 'ready' ? alerts.data.alerts : []
  const todo = all.filter(a => VIEWS[0].has(a.status)).length, check = all.filter(a => VIEWS[1].has(a.status)).length
  return (
    <section className="grid gap-6">
      <PageHeader title="Arena flags" scope="Problems with our versions, newest first. An open or investigating flag keeps that version of that picture out of the arena."
        finding={alerts.status === 'ready' ? { text: todo || check ? `${format.count(todo, 'flag')} to fix, ${format.count(check, 'fix', 'fixes')} to confirm: look at the newest picture, then set confirmed or reopen.` : 'Nothing to fix or confirm.' } : undefined} />
      <Tabs param="show" label="Flags by what is left to do" items={VIEWS.map(v => ({ value: v.value, label: v.label }))}>
        {view => (
          <DataView state={alerts} isEmpty={d => d.alerts.every(a => !VIEWS.find(v => v.value === view)!.has(a.status))} skeleton={<div className="grid gap-3"><Skeleton className="h-48" /><Skeleton className="h-48" /></div>}
            empty={<EmptyState title="No flags here">Flags are raised in the arena with the flag button (or 1 and 2) when something is wrong with a picture.</EmptyState>}>
            {d => <div className="grid gap-4">{d.alerts.filter(a => VIEWS.find(v => v.value === view)!.has(a.status)).map(a => <AlertCard key={a.id} a={a} expanded={expanded} />)}</div>}
          </DataView>
        )}
      </Tabs>
    </section>
  )
}
