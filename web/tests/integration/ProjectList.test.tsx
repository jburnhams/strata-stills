import { describe, expect, it } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import ProjectList from '../../src/components/ProjectList'
import { server } from '../utils/server'

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
})
