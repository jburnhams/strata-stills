// One question: "Which folder of this project do I open?" A table, because each folder has figures to compare (stills, panoramas, videos) and an action (remove);
// removing is reversible (the folder can be added again, nothing on disk changes), so it is an undo window (useUndoable + UndoBar), never a confirm dialog.
import { useState } from 'react'
import { Link } from 'react-router'
import { useQueryClient } from '@tanstack/react-query'
import { AsyncButton, Breadcrumb, DataTable, DataView, EmptyState, Field, Input, PageHeader, RowAction, Status, TableSkeleton, UndoBar, useAsyncAction, useLoad, useUndoable, type Column } from 'strata-react'
import { api, type SourceSummary } from '../api'
import { href, useIds } from '../route'

/** What a reachable folder holds, in words. */
export const holds = (s: SourceSummary) => [`${s.stills} stills`, s.panos ? `${s.panos} panoramas` : '', s.videos ? `${s.videos} videos` : ''].filter(Boolean).join(', ')

export default function ProjectPage() {
  const { project } = useIds()
  const proj = useLoad({ queryKey: ['project', project], queryFn: () => api.project(project) })
  const sources = useLoad({ queryKey: ['sources', project], queryFn: () => api.sources(project) })
  const client = useQueryClient(), undoable = useUndoable()
  const [path, setPath] = useState('')
  const refresh = () => Promise.all([client.invalidateQueries({ queryKey: ['sources', project] }), client.invalidateQueries({ queryKey: ['project', project] }), client.invalidateQueries({ queryKey: ['projects'] })])
  const add = useAsyncAction(async () => { await api.addSource(project, path.trim()); setPath(''); await refresh() }, { validate: () => ({ path: path.trim() ? undefined : 'Type the folder to add.' }) })
  const name = proj.status === 'ready' ? proj.data.name : project

  const columns: Column<SourceSummary>[] = [
    { key: 'path', header: 'Folder', value: s => s.path, cell: s => (s.reachable ? <Link to={href.source(project, s.id)} className="font-mono break-all">{s.path}</Link> : <span className="font-mono break-all">{s.path}</span>) },
    { key: 'holds', header: 'Contents', sort: false, cell: s => (s.reachable ? holds(s) : <Status tone="danger">Not reachable: is the share on?</Status>) },
    { key: 'stills', header: 'Stills', numeric: true, value: s => (s.reachable ? s.stills : null) },
    { key: 'remove', header: 'Actions', hideHeader: true, sort: false, cell: s => <RowAction label="Remove" subject={s.path} onPress={() => undoable.schedule({ id: s.id, message: `Removed ${s.path} from ${name}.`, failure: `${s.path} is back in ${name}: removing it did not go through.`, commit: async () => { await api.removeSource(project, s.path); await refresh() } })} /> },
  ]

  return (
    <section className="grid gap-6">
      <Breadcrumb items={[{ label: 'Projects', to: href.projects() }, { label: name }]} />
      <PageHeader title={name} scope="Source folders" />
      <DataView
        state={sources}
        isEmpty={rows => rows.filter(s => !undoable.isRemoved(s.id)).length === 0}
        skeleton={<TableSkeleton rows={2} columns={4} />}
        empty={<EmptyState title="No source folders yet">Add the folder the photos are in below; nothing in it is ever changed.</EmptyState>}
      >
        {rows => <DataTable caption="Source folders" columns={columns} rows={rows.filter(s => !undoable.isRemoved(s.id))} rowKey={s => s.id} />}
      </DataView>
      <form onSubmit={e => { e.preventDefault(); void add.run() }} className="max-w-prose" aria-label="Add a source folder">
        <Field label="Source folder" hint="For example \\Server\Drone\20250402." error={add.fieldErrors.path ?? add.error}
          action={<AsyncButton type="submit" phase={add.phase} error={add.error} labels={{ idle: 'Add source', pending: 'Adding…', done: 'Added', error: 'Try again' }} />}>
          <Input value={path} onChange={e => { setPath(e.target.value); add.clearFieldError('path') }} />
        </Field>
      </form>
      <UndoBar undoable={undoable} />
    </section>
  )
}
