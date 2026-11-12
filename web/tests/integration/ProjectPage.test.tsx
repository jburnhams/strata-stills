import { describe, expect, it } from 'vitest'
import { act, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { open } from '../utils/render'
import { mock, server } from '../utils/server'

describe('a project', () => {
  it('lists the source folders with what is in them, each linking into its contents', async () => {
    open('/p/demo')
    expect(await screen.findByRole('link', { name: 'C:\\shoot' })).toHaveAttribute('href', '/p/demo/s/abcd1234')
    expect(screen.getByText('2 stills, 1 panoramas, 1 videos')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Demo', level: 1 })).toBeInTheDocument()
    expect(within(screen.getByRole('navigation', { name: 'Breadcrumb' })).getByRole('link', { name: 'Projects' })).toHaveAttribute('href', '/')
  })

  it('takes a project id with slashes from the address', async () => {
    let asked = ''
    server.use(http.get('/api/project', ({ request }) => { asked = new URL(request.url).searchParams.get('id')!; return HttpResponse.json({ id: asked, name: 'Two', created: null, sources: [], n_sources: 0 }) }))
    open('/p/2025%2Fdrone%2Ftwo')
    expect(await screen.findByRole('heading', { name: 'Two' })).toBeInTheDocument()
    expect(asked).toBe('2025/drone/two')
  })

  it('teaches what to do when there are no folders', async () => {
    server.use(http.get('/api/project/sources', () => HttpResponse.json([])))
    open('/p/demo')
    expect(await screen.findByText('No source folders yet')).toBeInTheDocument()
  })

  it('marks an unreachable folder, which is not a link', async () => {
    server.use(http.get('/api/project/sources', () => HttpResponse.json([{ id: 'x', path: '\\\\Server\\off', reachable: false, stills: 0, panos: 0, videos: 0 }])))
    open('/p/demo')
    expect(await screen.findByText('Not reachable: is the share on?')).toBeInTheDocument()
    expect(screen.queryByRole('link', { name: '\\\\Server\\off' })).toBeNull()
  })

  it('adds a folder and reads the list again', async () => {
    let added = ''
    server.use(http.post('/api/project/sources', async ({ request }) => { added = ((await request.json()) as { path: string }).path; return HttpResponse.json({}) }))
    open('/p/demo')
    await screen.findByRole('link', { name: 'C:\\shoot' })
    await userEvent.type(screen.getByLabelText('Source folder'), ' D:\\new ')
    await userEvent.click(screen.getByRole('button', { name: 'Add source' }))
    await waitFor(() => expect(added).toBe('D:\\new'))
    await waitFor(() => expect(screen.getByLabelText('Source folder')).toHaveValue(''))
  })

  it('asks for a folder before sending anything', async () => {
    open('/p/demo')
    await userEvent.click(screen.getByRole('button', { name: 'Add source' }))
    expect(await screen.findByText('Type the folder to add.')).toBeInTheDocument()
    await userEvent.type(screen.getByLabelText('Source folder'), 'x')
    expect(screen.queryByText('Type the folder to add.')).toBeNull()
  })

  it('says why a folder was refused, under the field', async () => {
    open('/p/demo')
    await userEvent.type(screen.getByLabelText('Source folder'), 'bad')
    await userEvent.click(screen.getByRole('button', { name: 'Add source' }))
    await waitFor(() => expect(screen.getByLabelText('Source folder')).toHaveAccessibleDescription(/not a folder: bad/))
  })

  it('removes a folder at once with an undo, and tells the server when the window closes', async () => {
    server.use(http.delete('/api/project/sources', ({ request }) => { mock.removed.push(new URL(request.url).searchParams.get('path')!); return HttpResponse.json({}) }))
    open('/p/demo')
    await userEvent.click(await screen.findByRole('button', { name: 'Remove C:\\shoot' }))
    expect(screen.queryByRole('link', { name: 'C:\\shoot' })).toBeNull()
    expect(screen.getByText('Removed C:\\shoot from Demo.')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Undo' }))
    expect(await screen.findByRole('link', { name: 'C:\\shoot' })).toBeInTheDocument()
    expect(mock.removed).toEqual([])
    await userEvent.click(screen.getByRole('button', { name: 'Remove C:\\shoot' }))
    await act(async () => { await new Promise(r => setTimeout(r, 0)) })
    await waitFor(() => expect(mock.removed).toEqual(['C:\\shoot']), { timeout: 10000 })
  }, 15000)
})
