import { describe, expect, it } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import ProjectList from '../../src/components/ProjectList'
import { project, server } from '../utils/server'

describe('ProjectList', () => {
  it('lists projects', async () => {
    render(<ProjectList />)
    expect(await screen.findByText('Demo')).toBeInTheDocument()
    expect(screen.getByText('C:\\projects')).toBeInTheDocument()
    expect(screen.getByText('C:\\shoot')).toBeInTheDocument()
  })

  it('creates a project and navigates to it', async () => {
    render(<ProjectList />)
    await screen.findByText('Demo')
    await userEvent.type(screen.getByLabelText('Project name'), 'Fresh')
    await userEvent.click(screen.getByRole('button', { name: 'Create project' }))
    await waitFor(() => expect(window.location.hash).toBe('#/p/fresh'))
  })

  it('shows an empty state when there are no projects', async () => {
    server.use(http.get('/api/projects', () => HttpResponse.json([])))
    render(<ProjectList />)
    expect(await screen.findByText(/No projects yet/)).toBeInTheDocument()
  })

  it('shows an API error if loading fails', async () => {
    server.use(http.get('/api/projects', () => HttpResponse.json({ detail: 'Failed to fetch projects' }, { status: 500 })))
    render(<ProjectList />)
    expect(await screen.findByRole('alert')).toHaveTextContent('Failed to fetch projects')
  })

  it('shows an API error if creating project fails', async () => {
    render(<ProjectList />)
    await screen.findByText('Demo')
    server.use(http.post('/api/projects', () => HttpResponse.json({ detail: 'Failed to create' }, { status: 500 })))
    await userEvent.type(screen.getByLabelText('Project name'), 'Fresh')
    await userEvent.click(screen.getByRole('button', { name: 'Create project' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Failed to create')
  })

  it('sends the first source folder with the new project, trimmed, or none when the field is left empty', async () => {
    const bodies: { name: string; sources: string[] }[] = []
    server.use(http.post('/api/projects', async ({ request }) => { const b = await request.json() as { name: string; sources: string[] }; bodies.push(b); return HttpResponse.json({ ...project, id: b.name.toLowerCase(), name: b.name }) }))
    render(<ProjectList />); await screen.findByText('Demo')
    await userEvent.type(screen.getByLabelText('Project name'), 'One'); await userEvent.type(screen.getByLabelText('First source folder'), '  D:\\raw  ')
    await userEvent.click(screen.getByRole('button', { name: 'Create project' })); await waitFor(() => expect(bodies).toHaveLength(1))
    expect(bodies[0]).toEqual({ name: 'One', sources: ['D:\\raw'] })
    await waitFor(() => expect(screen.getByLabelText('Project name')).toHaveValue(''))                         // the form is cleared for the next one
    await userEvent.type(screen.getByLabelText('Project name'), 'Two'); await userEvent.click(screen.getByRole('button', { name: 'Create project' }))
    await waitFor(() => expect(bodies).toHaveLength(2)); expect(bodies[1]).toEqual({ name: 'Two', sources: [] })
  })

  it('says "source folder" for one and "source folders" for several', async () => {
    server.use(http.get('/api/projects', () => HttpResponse.json([project, { ...project, id: 'two', name: 'Two', n_sources: 2, sources: [{ path: 'C:\\a' }, { path: 'C:\\b' }] }])))
    render(<ProjectList />)
    expect(await screen.findByText('1 source folder')).toBeInTheDocument(); expect(screen.getByText('2 source folders')).toBeInTheDocument()
  })
})
