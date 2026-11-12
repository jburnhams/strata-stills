// One question, asked hundreds of times a day: "Which of these two is better?" The arena's votes are the only ratings that train the model, so this screen is built for speed and for not knowing
// which version is which: strata-react's Compare with `blind` (the pictures are called Left and Right, and are served under neutral addresses, img/<token>/left), one zoom and pan for both,
// swipe, blink and full screen for looking closely. Answers are single keys from one list that also draws the legend (← → ↓ S Z, and 1 2 to flag a side), and nothing moves when you answer
// (ux-research ux-character.md: done 100+ times a day, so no motion). The next pair is fetched, and its pictures loaded, while this one is judged.
// A flag says something is wrong with one picture: the pair is not counted, and the flag goes to the fix queue on the Arena flags screen.
import { useEffect, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router'
import { useQueryClient } from '@tanstack/react-query'
import { Alert, Button, ColorSchemeChoice, Compare, DataTable, DataView, Disclosure, EmptyState, HttpError, KeysLegend, PageHeader, Skeleton, TableSkeleton, Toggle, format, toast, useHotkeys, useLoad, type Column, type CompareMode, type Shortcut } from 'strata-react'
import { api, arenaImageUrl, type ArenaChoice, type ArenaPair, type ArenaSide, type HeadToHead, type VariantRating } from '../api'
import { href } from '../route'

/** A pair to judge, or the server's reason there is none (no pictures built yet, an unknown set). */
export type Next = { pair: ArenaPair } | { none: string }

/** The arena's own query (`set`, `want`) passed on to the server; anything else in the address is not its business. */
export function arenaQuery(params: URLSearchParams): string {
  const o = new URLSearchParams()
  for (const k of ['set', 'want']) { const v = params.get(k); if (v) o.set(k, v) }
  const q = o.toString()
  return q ? '?' + q : ''
}

/** Asks the browser for both pictures, so the next pair shows at once. */
function preload(p: ArenaPair) {
  for (const s of [p.left, p.right]) { const i = new Image(); i.src = arenaImageUrl(s) }
}

async function fetchNext(query: string): Promise<Next> {
  try { return { pair: await api.arenaNext(query) } }
  catch (e) { if (e instanceof HttpError && e.status === 404) return { none: e.reason ?? 'There is nothing to rate yet.' }; throw e }
}

function Ratings() {
  const r = useLoad({ queryKey: ['arena', 'ratings'], queryFn: api.ratings })
  const variants: Column<VariantRating>[] = [
    { key: 'name', header: 'Version', value: v => v.name },
    { key: 'bt', header: 'Rating', numeric: true, value: v => v.bt },
    { key: 'range', header: '5–95% range', numeric: true, sort: false, cell: v => (v.lo != null && v.hi != null ? `${format.integer(v.lo)} to ${format.integer(v.hi)}` : format.missing) },
    { key: 'elo', header: 'Elo', numeric: true, value: v => v.elo },
    { key: 'votes', header: 'Votes', numeric: true, value: v => v.votes },
  ]
  const pairs: Column<HeadToHead>[] = [
    { key: 'pair', header: 'Pair', value: h => `${h.a} vs ${h.b}` },
    { key: 'a', header: 'First wins', numeric: true, value: h => h.a_wins },
    { key: 'b', header: 'Second wins', numeric: true, value: h => h.b_wins },
    { key: 't', header: 'Equal', numeric: true, value: h => h.ties },
  ]
  return (
    <DataView state={r} isEmpty={() => false} skeleton={<TableSkeleton rows={4} columns={5} />} empty={null}>
      {j => (
        <div className="grid gap-4">
          <p className="max-w-prose type-body-sm text-muted-foreground">
            {format.count(j.votes, 'vote')} on {j.voted_pictures} of {format.count(j.pictures, 'picture')}{j.superseded_votes ? ` (${format.count(j.superseded_votes, 'older vote')} on remade pictures left out)` : ''}.
            Rating is Bradley-Terry on the Elo scale (mean 1500), with the 5–95% range from resampling the votes; Elo is played in time order.
          </p>
          <DataTable caption="Ratings of each version" columns={variants} rows={j.variants} rowKey={v => v.name} initialSort={{ key: 'bt', dir: 'descending' }} />
          <DataTable caption="Head to head" columns={pairs} rows={j.head_to_head} rowKey={h => `${h.a}|${h.b}`} />
        </div>
      )}
    </DataView>
  )
}

export default function ArenaPage() {
  const [params] = useSearchParams()
  const query = arenaQuery(params)
  const client = useQueryClient()
  const [n, setN] = useState(0)
  const key = (i: number) => ['arena', 'pair', query, i]
  const next = useLoad({ queryKey: key(n), queryFn: () => fetchNext(query), staleTime: Infinity, retry: false })
  const pair = next.status === 'ready' && 'pair' in next.data ? next.data.pair : null
  const [mode, setMode] = useState<CompareMode>('side')
  const [flags, setFlags] = useState<Record<ArenaSide, boolean>>({ left: false, right: false })
  const [votes, setVotes] = useState<number | null>(null), [failure, setFailure] = useState<string | null>(null)
  const busy = useRef(false), t0 = useRef(Date.now())

  // A new pair: its clock starts, its flags are clear, and the one after it is fetched now.
  useEffect(() => {
    if (!pair) return
    t0.current = Date.now(); setFlags({ left: false, right: false })
    void client.fetchQuery({ queryKey: key(n + 1), queryFn: () => fetchNext(query), staleTime: Infinity }).then(x => { if ('pair' in x) preload(x.pair) }, () => {})
  }, [pair?.token]) // eslint-disable-line react-hooks/exhaustive-deps

  const advance = () => { client.removeQueries({ queryKey: key(n) }); setN(i => i + 1) }
  const say = (e: unknown) => setFailure(e instanceof Error ? (e instanceof HttpError && e.reason ? e.reason : e.message) : String(e))
  const act = async (f: () => Promise<void>) => {
    if (busy.current) return
    busy.current = true; setFailure(null)
    try { await f() } catch (e) { say(e) } finally { busy.current = false }
  }
  const vote = (c: ArenaChoice) => pair && act(async () => {
    const r = await api.vote(pair.token, c, Date.now() - t0.current)
    if (r.votes != null) setVotes(r.votes)
    void client.invalidateQueries({ queryKey: ['arena', 'ratings'] })
    advance()
  })
  const undo = () => act(async () => {
    const r = await api.undoVote()
    setVotes(r.votes); client.removeQueries({ queryKey: ['arena', 'pair'] }); setN(i => i + 1)
    void client.invalidateQueries({ queryKey: ['arena', 'ratings'] })
    toast.show({ title: 'Vote undone', description: `${format.count(r.votes, 'vote')} now.` })
  })
  const flag = (side: ArenaSide, on: boolean) => pair && act(async () => {
    if (on) await api.flag(pair.token, side, Date.now() - t0.current)
    else await api.unflag(pair.token, side)
    setFlags(f => ({ ...f, [side]: on }))
  })

  const shortcuts: Shortcut[] = [
    { keys: ['ArrowLeft'], label: 'Left is better', run: () => void vote('left'), disabled: !pair },
    { keys: ['ArrowRight'], label: 'Right is better', run: () => void vote('right'), disabled: !pair },
    { keys: ['ArrowDown', 'e'], label: 'Equal', run: () => void vote('tie'), disabled: !pair },
    { keys: ['s'], label: 'Skip', run: () => void vote('skip'), disabled: !pair },
    { keys: ['z'], label: 'Undo last vote', run: () => void undo() },
    { keys: ['1'], label: 'Flag left', run: () => void flag('left', !flags.left), disabled: !pair },
    { keys: ['2'], label: 'Flag right', run: () => void flag('right', !flags.right), disabled: !pair },
  ]
  useHotkeys(shortcuts)
  const count = pair ? `${format.count(votes ?? pair.votes, 'vote')}, ${format.count(pair.pool, 'picture')}` : undefined

  return (
    <main className="mx-auto grid max-w-screen-2xl gap-4 px-4 py-4">
      <PageHeader title="Which is better?" scope={count} actions={<>
        <Link to={href.projects()} className="type-body-sm">Projects</Link>
        <Link to={href.alerts()} className="type-body-sm">Arena flags</Link>
        <ColorSchemeChoice compact />
      </>} />
      <DataView state={next} isEmpty={x => 'none' in x} skeleton={<Skeleton className="aspect-3/1 w-full" />}
        empty={<EmptyState title="Nothing to rate">{next.status === 'ready' && 'none' in next.data ? next.data.none : ''}</EmptyState>}>
        {() => pair && (
          <div className="grid gap-3">
            <Compare label="Two versions of one picture" blind missing="This picture could not be loaded" mode={mode} onModeChange={setMode} modes={['side', 'swipe', 'blink', 'difference']} keys={false}
              a={{ id: 'left', label: 'Left', media: { kind: 'image', src: arenaImageUrl(pair.left), alt: 'Left' } }}
              b={{ id: 'right', label: 'Right', media: { kind: 'image', src: arenaImageUrl(pair.right), alt: 'Right' } }} />
            <div role="group" aria-label="Your answer" className="flex flex-wrap items-center gap-2">
              <Button onPress={() => void vote('left')}>Left is better</Button>
              <Button onPress={() => void vote('tie')}>Equal</Button>
              <Button onPress={() => void vote('right')}>Right is better</Button>
              <Button variant="ghost" onPress={() => void vote('skip')}>Skip</Button>
              <Button variant="ghost" onPress={() => void undo()}>Undo last vote</Button>
              <span className="ml-auto flex gap-2">
                <Toggle label="Flag left" pressed={flags.left} onChange={on => void flag('left', on)} />
                <Toggle label="Flag right" pressed={flags.right} onChange={on => void flag('right', on)} />
              </span>
            </div>
          </div>
        )}
      </DataView>
      {failure && <Alert tone="danger" announce>{failure}</Alert>}
      <KeysLegend shortcuts={shortcuts} />
      <Disclosure summary="Ratings"><div className="pt-3"><Ratings /></div></Disclosure>
    </main>
  )
}
