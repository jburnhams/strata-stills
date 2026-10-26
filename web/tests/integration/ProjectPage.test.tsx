import { describe, expect, it } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import ProjectPage from '../../src/components/ProjectPage'
import { server } from '../utils/server'

describe('ProjectPage', () => {
  it('lists the source folders with what is in them and links into the contents', async () => {
    render(<ProjectPage project="demo" />)
    expect(await screen.findByText('2 stills, 1 panoramas, 1 videos')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /C:\\shoot/ })).toHaveAttribute('href', '#/p/demo/s/abcd1234')
  })

  it('shows an empty state when there are no source folders', async () => {
    server.use(http.get('/api/project/sources', () => HttpResponse.json([])))
    render(<ProjectPage project="demo" />)
    expect(await screen.findByText('This project has no source folders yet. Add one below.')).toBeInTheDocument()
  })

  it('marks an unreachable folder', async () => {
    server.use(http.get('/api/project/sources', () => HttpResponse.json([{ id: 'x', path: '\\\\Server\\off', reachable: false, stills: 0, panos: 0, videos: 0 }])))
    render(<ProjectPage project="demo" />)
    expect(await screen.findByText(/Not reachable/)).toBeInTheDocument()
  })

  it('adds a source folder', async () => {
    let requestMade = false
    server.use(http.post('/api/project/sources', async ({ request }) => {
        const body = await request.json()
        if (body.path === 'new_folder') { requestMade = true }
        return HttpResponse.json([{ id: 'y', path: 'new_folder', reachable: true, stills: 1, panos: 0, videos: 0 }])
    }))
    render(<ProjectPage project="demo" />)
    await screen.findByText('2 stills, 1 panoramas, 1 videos')
    await userEvent.type(screen.getByLabelText('Source folder'), 'new_folder')
    await userEvent.click(screen.getByRole('button', { name: 'Add source' }))
    expect(screen.getByLabelText('Source folder')).toHaveValue('')
    await waitFor(() => expect(requestMade).toBe(true))
  })

  it('reports a refused folder when adding', async () => {
    render(<ProjectPage project="demo" />)
    await screen.findByText('2 stills, 1 panoramas, 1 videos')
    await userEvent.type(screen.getByLabelText('Source folder'), 'bad')
    await userEvent.click(screen.getByRole('button', { name: 'Add source' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('not a folder: bad')
  })

  it('removes a source folder', async () => {
    let requestMade = false
    server.use(http.delete('/api/project/sources', ({ request }) => {
        const url = new URL(request.url)
        if (url.searchParams.get('path') === 'C:\\shoot') requestMade = true
        return HttpResponse.json([{ id: 'z', path: 'deleted', reachable: false, stills: 0, panos: 0, videos: 0 }])
    }))
    render(<ProjectPage project="demo" />)
    await screen.findByText('2 stills, 1 panoramas, 1 videos')
    await userEvent.click(screen.getByRole('button', { name: 'Remove from project' }))
    await waitFor(() => expect(screen.queryByRole('alert')).toBeNull())
    await waitFor(() => expect(requestMade).toBe(true))
  })
})
