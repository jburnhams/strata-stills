// The Browser: "Which of my pictures go into a project?" The source roots as a lazy tree on the left (strata-react Tree: a folder's children are read as it opens; a press on a
// name shows that folder, the boxes tick folders to add, and ticking one with sub-folders takes them all without opening it), the pictures of the folder in view as cached
// thumbnails on the right (one SelectableGrid of Thumbnails, which try each source the server names in turn; grouped, zoomed with the slider, Ctrl+wheel or + and -), how they
// are looked at in one ViewOptions bar (sort, group, size, filter: kept per browser and in the address), and adding the chosen pictures, the ticked folders, the folder in view,
// or it with every folder below it, to a project or to a new one.
import { useState } from 'react'
import { Link } from 'react-router'
import { useQueryClient, type QueryClient } from '@tanstack/react-query'
import {
  AsyncButton, BulkBar, Button, DataView, Dialog, EmptyState, Field, Input, NativeSelect, PageHeader, SelectableGrid, SelectableGridSkeleton, Status, Thumbnail, ToggleGroup,
  Tree, ViewOptions, format, toast, useAsyncAction, useChoices, useExpanded, useLoad, useSearchParam, useSelection, useViewOptions,
} from 'strata-react'
import { api, thumbUrl, type BrowseAdd, type BrowseFolder, type BrowsePicture } from '../api'
import { GROUPS, ancestorKeys, groupOf, shownFrom, takenAt, thumbPx, thumbSources, typeOf, type ShowKey } from '../browse'
import { href } from '../route'

const SHOWS = [{ value: 'best', label: 'Best develop' }, { value: 'camera', label: 'Camera preview' }]
const isShow = (v: string): v is ShowKey => v === 'best' || v === 'camera'

/** The name a new project gets unless the person types another: the folder's own name (`2026-04-06`, `20250402 - Hazelmere`). */
export const folderName = (key: string) => key.split('/').filter(Boolean).at(-1) ?? ''

const folderQuery = (key: string) => ({ queryKey: ['browse', 'folders', key], queryFn: () => api.browseFolders(key || null) })
const useChildren = (node: BrowseFolder) => useLoad(folderQuery(node.key))

/** Every folder below a ticked one, read level by level from the same cache the tree fills (only folders that have sub-folders are asked for theirs). */
export async function foldersBelow(client: QueryClient, node: BrowseFolder): Promise<string[]> {
  const kids = await client.fetchQuery(folderQuery(node.key))
  const deeper = await Promise.all(kids.filter(k => k.folders > 0).map(k => foldersBelow(client, k)))
  return [...kids.map(k => k.key), ...deeper.flat()]
}

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
  return <Thumbnail src={thumbSources(p, show).map(s => thumbUrl(folder, p.stem, s, px))} alt={p.stem} missing="No picture" aspect="3/2" fit="cover" className="w-full" />
}

/** The pictures of one folder (mounted per folder, so a choice never carries over to the next one). */
function FolderPictures({ folder, onAdd }: { folder: string; onAdd: (what: string, body: BrowseAdd, done: () => void) => void }) {
  const [show] = useSearchParam<ShowKey>('show', { default: 'best', parse: isShow })
  const view = useViewOptions<BrowsePicture>({
    id: 'browser',
    sorts: [
      { key: 'name', label: 'Name', value: p => p.stem, directions: ['A to Z', 'Z to A'] },
      { key: 'taken', label: 'Date taken', value: takenAt, directions: ['Oldest first', 'Newest first'] },
    ],
    groups: GROUPS.map(g => ({ key: g.key, label: g.label, of: (p: BrowsePicture) => groupOf(p, g.key) })),
    size: { min: 96, max: 480, step: 16, default: 200 },
    filterText: p => `${p.stem} ${p.files.join(' ')} ${p.projects.join(' ')}`,
    url: true,
  })
  // The listing; while the nudity filter is still looking at some of its pictures, it is read again every few seconds so they appear as they are cleared.
  const listing = useLoad({ queryKey: ['browse', 'pictures', folder], queryFn: () => api.browsePictures(folder), refetchInterval: q => ((q.state.data?.checking ?? 0) > 0 ? 3000 : false) })
  const all = listing.status === 'ready' ? listing.data.pictures : []
  const rows = view.apply(all)
  const chosen = useSelection(rows, p => p.stem)
  const checking = listing.status === 'ready' ? listing.data.checking : 0
  const group = GROUPS.find(g => g.key === view.group)?.key
  return (
    <div className="grid min-w-0 gap-3">
      <div className="flex flex-wrap items-center gap-3">
        <ViewOptions view={view} shown={rows.length} total={all.length} label="View" />
        <ToggleGroup label="Thumbnails from" size="sm" options={SHOWS} param="show" />
      </div>
      <DataView
        state={listing}
        isEmpty={l => l.pictures.length === 0 && l.checking === 0}
        skeleton={<SelectableGridSkeleton count={8} tileSize={view.size} />}
        empty={<EmptyState title="No pictures here">This folder holds no raws or JPEGs; its sub-folders may. Open it in the tree.</EmptyState>}
      >
        {() => (
          <div className="grid gap-4">
            {checking > 0 && <p role="status" className="type-body-sm text-muted-foreground">{format.count(checking, 'picture')} still being checked by the nudity filter; they appear here once cleared.</p>}
            <BulkBar count={chosen.count} noun="picture" onClear={chosen.clear}>
              <Button variant="default" onPress={() => onAdd(format.count(chosen.count, 'picture'), { pictures: { [folder]: [...chosen.selected] } }, chosen.clear)}>Add {format.count(chosen.count, 'picture')} to a project…</Button>
            </BulkBar>
            <SelectableGrid<BrowsePicture>
              label="Pictures"
              rows={rows}
              rowKey={p => p.stem}
              renderImage={p => <Tile folder={folder} p={p} show={show} px={thumbPx(view.size)} />}
              detail={p => <span title={`${p.files.join(', ')} · thumbnail: ${shownFrom(p, show)}`}><span className="font-mono">{p.stem}</span> <span className="text-muted-foreground">{typeOf(p)}</span></span>}
              badge={p => (p.projects.length > 0 ? <Status tone="info">{`In ${p.projects.length === 1 ? p.projects[0] : format.count(p.projects.length, 'project')}`}</Status> : null)}
              selection={{ ...chosen, rowLabel: p => p.stem }}
              aspect="landscape"
              tileSize={view.size}
              onTileSizeChange={view.setSize}
              groupBy={group ? p => groupOf(p, group) : undefined}
              maxHeight={720}
            />
          </div>
        )}
      </DataView>
    </div>
  )
}

export default function BrowserPage() {
  const [folder, setFolder] = useSearchParam<string>('folder', { default: '', replace: false })
  const client = useQueryClient()
  const roots = useLoad(folderQuery(''))
  const expanded = useExpanded({ initial: ancestorKeys(folder) })
  const ticked = useChoices()
  const parent = useLoad({ ...folderQuery(ancestorKeys(folder).at(-1) ?? ''), enabled: !!folder })
  const here = parent.status === 'ready' ? parent.data.find(f => f.key === folder) : undefined
  const [adding, setAdding] = useState<{ what: string; body: BrowseAdd; done?: () => void } | null>(null)

  return (
    <section className="grid gap-4">
      <PageHeader title="Browser" scope={folder ? <span className="font-mono break-all">{folder}</span> : 'Choose a folder of the archive'} actions={
        <>
          {ticked.count > 0 && <Button onPress={() => setAdding({ what: format.count(ticked.count, 'folder'), body: { folders: [...ticked.selected] }, done: ticked.clear })}>Add {format.count(ticked.count, 'folder')}…</Button>}
          {folder && <Button onPress={() => setAdding({ what: 'this folder', body: { folders: [folder] } })}>Add folder…</Button>}
          {folder && here && here.folders > 0 && <Button onPress={() => setAdding({ what: 'this folder and the folders in it', body: { folders: [folder], subfolders: true } })}>Add with sub-folders…</Button>}
        </>
      } />
      <div className="grid items-start gap-4 md:grid-cols-[minmax(14rem,18rem)_1fr]">
        <nav aria-label="Folders" className="rounded-surface border border-border p-2">
          <Tree<BrowseFolder>
            label="Folders"
            roots={roots}
            useChildren={useChildren}
            nodeKey={n => n.key}
            nodeLabel={n => n.name}
            isBranch={n => n.folders > 0}
            choices={ticked}
            current={folder || null}
            onCurrentChange={k => setFolder(k)}
            count={n => (n.pictures > 0 ? format.integer.format(n.pictures) : null)}
            resolveBranch={n => foldersBelow(client, n)}
            coverNoun={['folder', 'folders']}
            expanded={expanded}
            emptyRow="No folders"
            height="calc(100dvh - 14rem)"
          />
        </nav>
        {folder ? <FolderPictures key={folder} folder={folder} onAdd={(what, body, done) => setAdding({ what, body, done })} />
          : <EmptyState title="No folder chosen">Open a year in the tree and choose a folder to see its pictures, or tick folders to add them to a project.</EmptyState>}
      </div>
      {adding && <AddDialog what={adding.what} body={adding.body} suggested={folderName(folder)} open onOpenChange={o => { if (!o) setAdding(null) }} onAdded={() => adding.done?.()} />}
      <p className="type-caption text-muted-foreground">Pictures are only read: adding them to a project records where they are. <Link to={href.projects()}>Projects</Link></p>
    </section>
  )
}
