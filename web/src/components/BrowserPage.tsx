// The Browser: "Which of my pictures go into a project?" The source roots as a lazy tree on the left (strata-react Tree: a folder's children are read as it opens), the pictures of the
// folder in view as cached thumbnails on the right (SelectableGrid of Thumbnails, which try each source the server names in turn), and adding the chosen pictures, the folder,
// or the folder with every folder below it, to a project or to a new one.
// Interim pieces until strata-react has them (issues #245 Tree current + ticked, #246 grid sizing/sorting/grouping/selection behaviours, #247 view options): the tree browses one
// folder (single selection) and the folder is added with buttons; groups are one SelectableGrid each over the same selection; sort, group, size and source are a plain Toolbar.
import { useEffect, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router'
import { useQueryClient } from '@tanstack/react-query'
import {
  AsyncButton, BulkBar, Button, DataView, Dialog, EmptyState, Field, Input, NativeSelect, PageHeader, SelectableGrid, Status, Thumbnail, ThumbnailSkeleton, ToggleGroup, Toolbar,
  Tree, format, toast, useAsyncAction, useExpanded, useLoad, type TileSize,
} from 'strata-react'
import { api, thumbUrl, type BrowseAdd, type BrowseFolder, type BrowsePicture } from '../api'
import { GROUPS, SORTS, ancestorKeys, groupPictures, shownFrom, sortPictures, thumbSources, typeOf, type GroupKey, type ShowKey, type SortKey } from '../browse'
import { href } from '../route'

const PX: Record<TileSize, number> = { sm: 160, md: 320, lg: 640 }  // the server's thumbnail sizes, enough for a tile on a 2x screen
const SIZES = [{ value: 'sm', label: 'Small' }, { value: 'md', label: 'Medium' }, { value: 'lg', label: 'Large' }]
const SHOWS = [{ value: 'best', label: 'Best develop' }, { value: 'camera', label: 'Camera preview' }]
const pick = <T extends string>(v: string | null, ok: readonly { value: T }[], d: T): T => (ok.some(o => o.value === v) ? (v as T) : d)

/** One folder row in the tree: its name and how many pictures it holds. */
function FolderRow({ node }: { node: BrowseFolder }) {
  return (
    <span className="flex min-w-0 items-baseline gap-2">
      <span className="truncate" title={node.name}>{node.name}</span>
      {node.pictures > 0 && <span className="ml-auto shrink-0 type-caption tabular-nums text-muted-foreground">{format.integer.format(node.pictures)}</span>}
    </span>
  )
}

/** The name a new project gets unless the person types another: the folder's own name (`2026-04-06`, `20250402 - Hazelmere`). */
export const folderName = (key: string) => key.split('/').filter(Boolean).at(-1) ?? ''

const useChildren = (node: BrowseFolder) => useLoad({ queryKey: ['browse', 'folders', node.key], queryFn: () => api.browseFolders(node.key) })

/** What to add, and where: an existing project or a new one. Opened for chosen pictures, the folder in view, or the folder with its sub-folders. */
function AddDialog({ what, body, suggested, open, onOpenChange, onAdded }: { what: string; body: BrowseAdd; suggested: string; open: boolean; onOpenChange: (o: boolean) => void; onAdded: () => void }) {
  const projects = useLoad({ queryKey: ['projects'], queryFn: api.projects })
  const [target, setTarget] = useState(''), [name, setName] = useState(suggested)
  const client = useQueryClient()
  const add = useAsyncAction(async () => {
    const r = await api.browseAdd({ ...body, ...(target ? { project: target } : { name: name.trim() }) })
    await Promise.all([client.invalidateQueries({ queryKey: ['projects'] }), client.invalidateQueries({ queryKey: ['browse', 'pictures'] })])
    const parts = [r.sources ? format.count(r.sources, 'folder') : null, r.pictures ? format.count(r.pictures, 'picture') : null].filter(Boolean)
    toast.show({ title: parts.length ? `Added ${parts.join(' and ')} to ${r.project}` : `Nothing new: ${r.project} already has ${r.unchanged === 1 ? 'it' : 'them'}`, action: { label: 'Open project', onPress: () => { location.hash = '#' + href.project(r.project) } } })
    onAdded()
    onOpenChange(false)
  }, { validate: () => ({ name: target || name.trim() ? undefined : 'Give the new project a name.' }) })
  return (
    <Dialog title={`Add ${what} to a project`} open={open} onOpenChange={onOpenChange} actions={close => (
      <>
        <Button onPress={close}>Cancel</Button>
        <AsyncButton variant="default" phase={add.phase} error={add.error} onPress={() => void add.run()} labels={{ idle: target ? 'Add to project' : 'Create project and add', pending: 'Adding…', done: 'Added', error: 'Try again' }} />
      </>
    )}>
      <div className="grid gap-3">
        <Field label="Project" hint={projects.status === 'error' ? 'The projects could not be read: you can still make a new one.' : undefined}>
          <NativeSelect value={target} onChange={e => setTarget(e.target.value)}>
            <option value="">New project…</option>
            {projects.status === 'ready' && projects.data.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
          </NativeSelect>
        </Field>
        {!target && <Field label="New project name" hint="Suggested from the folder's name." error={add.fieldErrors.name}><Input value={name} onChange={e => { setName(e.target.value); add.clearFieldError('name') }} autoFocus onFocus={e => e.currentTarget.select()} /></Field>}
        {add.error && <p role="alert" className="type-body-sm text-danger">{add.error}</p>}
      </div>
    </Dialog>
  )
}

function Tile({ folder, p, show, px }: { folder: string; p: BrowsePicture; show: ShowKey; px: number }) {
  const srcs = thumbSources(p, show).map(s => thumbUrl(folder, p.stem, s, px))
  return (
    <Thumbnail src={srcs} alt={p.stem} missing="No picture" aspect="3/2" fit="cover" className="w-full">
      {p.projects.length > 0 && <span className="absolute bottom-1 left-1"><Status tone="info">{`In ${p.projects.length === 1 ? p.projects[0] : format.count(p.projects.length, 'project')}`}</Status></span>}
    </Thumbnail>
  )
}


export default function BrowserPage() {
  const [params, setParams] = useSearchParams()
  const folder = params.get('folder') ?? ''
  const sort = pick<SortKey>(params.get('sort'), SORTS, 'name')
  const group = pick<GroupKey>(params.get('group'), GROUPS, 'none')
  const size = pick<TileSize>(params.get('size'), SIZES as { value: TileSize }[], 'md')
  const show = pick<ShowKey>(params.get('show'), SHOWS as { value: ShowKey }[], 'best')
  const set = (k: string, v: string | null) => setParams(prev => { const n = new URLSearchParams(prev); if (v) n.set(k, v); else n.delete(k); return n }, { replace: k !== 'folder' })

  const roots = useLoad({ queryKey: ['browse', 'folders', ''], queryFn: () => api.browseFolders(null) })
  const expanded = useExpanded({ initial: ancestorKeys(folder) })
  // The listing of the folder in view; while the nudity filter is still looking at some of its pictures, it is read again every few seconds so they appear as they are cleared.
  const listing = useLoad({ queryKey: ['browse', 'pictures', folder], queryFn: () => api.browsePictures(folder), enabled: !!folder, refetchInterval: q => ((q.state.data?.checking ?? 0) > 0 ? 3000 : false) })
  const parent = useLoad({ queryKey: ['browse', 'folders', ancestorKeys(folder).at(-1) ?? ''], queryFn: () => api.browseFolders(ancestorKeys(folder).at(-1) ?? null), enabled: !!folder })
  const here = parent.status === 'ready' ? parent.data.find(f => f.key === folder) : undefined

  const [chosen, setChosen] = useState<Set<string>>(new Set())
  useEffect(() => setChosen(new Set()), [folder])
  const [adding, setAdding] = useState<{ what: string; body: BrowseAdd } | null>(null)

  const rows = useMemo(() => (listing.status === 'ready' ? sortPictures(listing.data.pictures, sort) : []), [listing, sort])
  const groups = useMemo(() => groupPictures(rows, group), [rows, group])
  const checking = listing.status === 'ready' ? listing.data.checking : 0

  return (
    <section className="grid gap-4">
      <PageHeader title="Browser" scope={folder ? <span className="font-mono break-all">{folder}</span> : 'Choose a folder of the archive'} actions={folder ? (
        <>
          <Button onPress={() => setAdding({ what: 'this folder', body: { folders: [folder] } })}>Add folder…</Button>
          {here && here.folders > 0 && <Button onPress={() => setAdding({ what: 'this folder and the folders in it', body: { folders: [folder], subfolders: true } })}>Add with sub-folders…</Button>}
        </>
      ) : undefined} />
      <div className="grid items-start gap-4 md:grid-cols-[minmax(14rem,18rem)_1fr]">
        <nav aria-label="Folders" className="rounded-surface border border-border p-2">
          <Tree<BrowseFolder>
            label="Folders"
            roots={roots}
            useChildren={useChildren}
            nodeKey={n => n.key}
            nodeLabel={n => n.name}
            isBranch={n => n.folders > 0}
            renderNode={n => <FolderRow node={n} />}
            selected={folder || null}
            onSelect={k => set('folder', k)}
            expanded={expanded}
            emptyRow="No folders"
            height="calc(100dvh - 14rem)"
          />
        </nav>
        <div className="grid min-w-0 gap-3">
          <Toolbar label="View">
            <label className="flex items-center gap-2 type-body-sm">Sort <NativeSelect size="sm" value={sort} onChange={e => set('sort', e.target.value)}>{SORTS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}</NativeSelect></label>
            <label className="flex items-center gap-2 type-body-sm">Group <NativeSelect size="sm" value={group} onChange={e => set('group', e.target.value)}>{GROUPS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}</NativeSelect></label>
            <ToggleGroup label="Tile size" size="sm" options={SIZES} value={size} onChange={v => set('size', v)} />
            <ToggleGroup label="Thumbnails from" size="sm" options={SHOWS} value={show} onChange={v => set('show', v)} />
          </Toolbar>
          {!folder ? (
            <EmptyState title="No folder chosen">Open a year in the tree and choose a folder to see its pictures.</EmptyState>
          ) : (
            <DataView
              state={listing}
              isEmpty={l => l.pictures.length === 0 && l.checking === 0}
              skeleton={<div className="grid grid-cols-2 gap-3 sm:grid-cols-4">{[0, 1, 2, 3].map(i => <ThumbnailSkeleton key={i} aspect="3/2" />)}</div>}
              empty={<EmptyState title="No pictures here">This folder holds no raws or JPEGs; its sub-folders may. Open it in the tree.</EmptyState>}
            >
              {() => (
                <div className="grid gap-4">
                  {checking > 0 && <p role="status" className="type-body-sm text-muted-foreground">{format.count(checking, 'picture')} still being checked by the nudity filter; they appear here once cleared.</p>}
                  <BulkBar count={chosen.size} noun="picture" onClear={() => setChosen(new Set())}>
                    <Button variant="default" onPress={() => setAdding({ what: format.count(chosen.size, 'picture'), body: { pictures: { [folder]: [...chosen] } } })}>Add {format.count(chosen.size, 'picture')} to a project…</Button>
                  </BulkBar>
                  {groups.map(g => (
                    <section key={g.key} aria-label={g.label} className="grid gap-2">
                      {group !== 'none' && <h2 className="type-heading">{g.label} <span className="type-body-sm text-muted-foreground">{format.count(g.rows.length, 'picture')}</span></h2>}
                      <SelectableGrid<BrowsePicture>
                        label={group === 'none' ? 'Pictures' : `Pictures, ${g.label}`}
                        rows={g.rows}
                        rowKey={p => p.stem}
                        renderImage={p => <Tile folder={folder} p={p} show={show} px={PX[size]} />}
                        detail={p => <span title={`${p.files.join(', ')} · thumbnail: ${shownFrom(p, show)}`}><span className="font-mono">{p.stem}</span> <span className="text-muted-foreground">{typeOf(p)}</span></span>}
                        selection={{ selected: chosen, onChange: setChosen, rowLabel: p => p.stem }}
                        aspect="landscape"
                        tileSize={size}
                        maxHeight={720}
                      />
                    </section>
                  ))}
                </div>
              )}
            </DataView>
          )}
        </div>
      </div>
      {adding && <AddDialog what={adding.what} body={adding.body} suggested={folderName(folder)} open onOpenChange={o => { if (!o) setAdding(null) }} onAdded={() => setChosen(new Set())} />}
      <p className="type-caption text-muted-foreground">Pictures are only read: adding them to a project records where they are. <Link to={href.projects()}>Projects</Link></p>
    </section>
  )
}
