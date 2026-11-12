// One question: "Which project am I working on?" Each project is a destination, so it is a LinkCard (one link, the whole card the target); a new one is made with the form under them.
import { useState } from 'react'
import { useNavigate } from 'react-router'
import { useQueryClient } from '@tanstack/react-query'
import { AsyncButton, DataView, EmptyState, Field, Input, LinkCard, LinkCardList, PageHeader, Skeleton, format, useAsyncAction, useLoad } from 'strata-react'
import { api } from '../api'
import { href } from '../route'

export default function ProjectList() {
  const projects = useLoad({ queryKey: ['projects'], queryFn: api.projects })
  const settings = useLoad({ queryKey: ['settings'], queryFn: api.settings })
  const [name, setName] = useState(''), [src, setSrc] = useState('')
  const navigate = useNavigate(), client = useQueryClient()
  const create = useAsyncAction(async () => {
    const p = await api.createProject(name.trim(), src.trim() ? [src.trim()] : [])
    await client.invalidateQueries({ queryKey: ['projects'] })
    navigate(href.project(p.id))
  }, { validate: () => ({ name: name.trim() ? undefined : 'Give the project a name.' }) })

  return (
    <section className="grid gap-6">
      <PageHeader title="Projects" scope={settings.status === 'ready' ? <>Projects folder: <span className="font-mono">{settings.data.projects_dir}</span></> : undefined} />
      <DataView
        state={projects}
        isEmpty={ps => ps.length === 0}
        skeleton={<div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{[0, 1, 2].map(i => <Skeleton key={i} className="h-24" />)}</div>}
        empty={<EmptyState title="No projects yet">Create one below with a name and the folder its photos are in.</EmptyState>}
      >
        {ps => (
          <LinkCardList label="Projects">
            {ps.map(p => <LinkCard key={p.id} to={href.project(p.id)} title={p.name} description={[format.count(p.n_sources, 'source folder'), ...p.sources.map(s => s.path)].join(' · ')} />)}
          </LinkCardList>
        )}
      </DataView>
      <form className="grid gap-3 max-w-prose" onSubmit={e => { e.preventDefault(); void create.run() }} aria-label="New project">
        <h2 className="type-heading">New project</h2>
        <Field label="Project name" error={create.fieldErrors.name}><Input value={name} onChange={e => { setName(e.target.value); create.clearFieldError('name') }} /></Field>
        <Field label="First source folder" hint="Optional: you can add folders later."><Input value={src} onChange={e => setSrc(e.target.value)} /></Field>
        <AsyncButton type="submit" variant="default" className="justify-self-start" phase={create.phase} error={create.error} labels={{ idle: 'Create project', pending: 'Creating…', done: 'Created', error: 'Try again' }} />
        {create.error && <p role="alert" className="type-body-sm text-danger">{create.error}</p>}
      </form>
    </section>
  )
}
