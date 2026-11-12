import { describe, expect, it } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse, delay } from 'msw'
import { format } from 'strata-react'
import { open } from '../utils/render'
import { server } from '../utils/server'

describe('the project list', () => {
  it('lists each project as one link with its folders, and says where projects are kept', async () => {
    open('/')
    const link = await screen.findByRole('link', { name: 'Demo' })
    expect(link).toHaveAccessibleDescription(`${format.count(1, 'source folder')} · C:\\shoot`)
    expect(await screen.findByText('C:\\projects')).toBeInTheDocument()
  })

  it('teaches what to do when there are no projects', async () => {
    server.use(http.get('/api/projects', () => HttpResponse.json([])))
    open('/')
    expect(await screen.findByText('No projects yet')).toBeInTheDocument()
  })

  it('shows a skeleton while loading and the reason when the list fails', async () => {
    server.use(http.get('/api/projects', async () => { await delay(50); return HttpResponse.json({ detail: 'projects folder is gone' }, { status: 500 }) }))
    open('/')
    expect(screen.queryByRole('link', { name: 'Demo' })).toBeNull()
    expect(await screen.findByRole('alert')).toBeInTheDocument()
  })

  it('creates a project with its first folder and opens it', async () => {
    let body: unknown
    server.use(http.post('/api/projects', async ({ request }) => { body = await request.json(); return HttpResponse.json({ id: 'trip', name: 'Trip', created: null, n_sources: 1, sources: [] }) }))
    const { router } = open('/')
    await userEvent.type(screen.getByLabelText('Project name'), ' Trip ')
    await userEvent.type(screen.getByLabelText('First source folder'), 'D:\\trip')
    await userEvent.click(screen.getByRole('button', { name: 'Create project' }))
    await waitFor(() => expect(router.state.location.pathname).toBe('/p/trip'))
    expect(body).toEqual({ name: 'Trip', sources: ['D:\\trip'] })
  })

  it('creates a project without a folder', async () => {
    let body: unknown
    server.use(http.post('/api/projects', async ({ request }) => { body = await request.json(); return HttpResponse.json({ id: 'x', name: 'X', created: null, n_sources: 0, sources: [] }) }))
    open('/')
    await userEvent.type(screen.getByLabelText('Project name'), 'X')
    await userEvent.click(screen.getByRole('button', { name: 'Create project' }))
    await waitFor(() => expect(body).toEqual({ name: 'X', sources: [] }))
  })

  it('asks for a name before sending anything, and clears the message as one is typed', async () => {
    open('/')
    await userEvent.click(screen.getByRole('button', { name: 'Create project' }))
    expect(await screen.findByText('Give the project a name.')).toBeInTheDocument()
    expect(screen.getByLabelText('Project name')).toHaveAttribute('aria-invalid', 'true')
    await userEvent.type(screen.getByLabelText('Project name'), 'a')
    expect(screen.queryByText('Give the project a name.')).toBeNull()
  })

  it('says why the server refused a project', async () => {
    server.use(http.post('/api/projects', () => HttpResponse.json({ detail: 'a project of that name exists' }, { status: 400 })))
    open('/')
    await userEvent.type(screen.getByLabelText('Project name'), 'Demo')
    await userEvent.click(screen.getByRole('button', { name: 'Create project' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('a project of that name exists')
  })
})
