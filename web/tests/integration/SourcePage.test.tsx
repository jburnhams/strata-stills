import { describe, expect, it } from 'vitest'
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { format } from 'strata-react'
import { open } from '../utils/render'
import { mock, server } from '../utils/server'

const pictures = async () => within(await screen.findByRole('list', { name: 'Pictures' }))

describe('the contents of a folder', () => {
  it('shows each still as a thumbnail linking to it, and keeps videos behind a switch', async () => {
    open('/p/demo/s/abcd1234')
    const grid = await pictures()
    expect(grid.getByRole('link', { name: /A_1/ })).toHaveAttribute('href', '/p/demo/i/abcd1234~A_1')
    expect(grid.getByRole('img', { name: 'A_1' })).toHaveAttribute('src', '/api/project/image?id=demo&item=abcd1234~A_1&variant=raw&transform=original&px=360')
    expect(grid.queryByRole('link', { name: /A_3/ })).toBeNull()
    expect(screen.getAllByText('C:\\shoot').length).toBeGreaterThan(0)
    await userEvent.click(screen.getByRole('switch', { name: new RegExp(`^Show ${format.count(1, 'video')}`) }))
    expect((await pictures()).getByRole('img', { name: 'A_3: Video' })).toBeInTheDocument()
  })

  it('teaches what to do with an empty folder', async () => {
    server.use(http.get('/api/project/items', () => HttpResponse.json([])))
    open('/p/demo/s/abcd1234')
    expect(await screen.findByText('No pictures in this folder')).toBeInTheDocument()
    expect(screen.queryByRole('switch')).toBeNull()
  })

  it('says why the folder could not be read', async () => {
    server.use(http.get('/api/project/items', () => HttpResponse.json({ detail: 'folder is offline' }, { status: 400 })))
    open('/p/demo/s/abcd1234')
    expect(await screen.findByText(/folder is offline/)).toBeInTheDocument()
  })
})

describe('auto-generate', () => {
  it('has nothing to do until a pipeline is chosen', async () => {
    open('/p/demo/s/abcd1234')
    expect(await screen.findByRole('button', { name: 'Nothing chosen' })).toBeDisabled()
  })

  it('saves the chosen pipelines as the folder\'s rule, and generates what is missing as a job', async () => {
    open('/p/demo/s/abcd1234')
    await userEvent.click(await screen.findByRole('checkbox', { name: 'Best quality (automatic)' }))
    await waitFor(() => expect(mock.rules).toEqual([{ source: 'abcd1234', pipeline: 'best' }]))
    const go = await screen.findByRole('button', { name: 'Generate 3 missing' })
    await userEvent.click(go)
    await waitFor(() => expect(mock.applies).toBe(1))
    expect(await screen.findByText('Made 3 of 3.')).toBeInTheDocument()
    expect(screen.getByText('Generated')).toBeInTheDocument()
  })

  it('leaves the rules of other folders as they are, and takes a pipeline off again', async () => {
    mock.rules = [{ source: 'other', pipeline: 'archive' }, { source: 'abcd1234', pipeline: 'best' }]
    open('/p/demo/s/abcd1234')
    const best = await screen.findByRole('checkbox', { name: 'Best quality (automatic)' })
    await waitFor(() => expect(best).toBeChecked())
    await userEvent.click(best)
    await waitFor(() => expect(mock.rules).toEqual([{ source: 'other', pipeline: 'archive' }]))
    await waitFor(() => expect(best).not.toBeChecked())
  })

  it('says everything is generated when a rule has nothing missing', async () => {
    server.use(http.get('/api/project/rules', () => HttpResponse.json({ rules: [{ source: '*', pipeline: 'best' }], pending: 0 })))
    open('/p/demo/s/abcd1234')
    expect(await screen.findByRole('button', { name: 'Everything is generated' })).toBeDisabled()
  })

  it('shows a running job with its progress, and stops it', async () => {
    let state = 'running'
    server.use(
      http.get('/api/project/apply', () => HttpResponse.json(state === 'running' ? { state, done: 1, total: 3, text: 'A_2', failed: [] } : { state, done: 1, total: 3, failed: [] })),
      http.delete('/api/project/apply', () => { state = 'stopped'; return HttpResponse.json({ state: 'stopped' }) }),
    )
    open('/p/demo/s/abcd1234')
    expect(await screen.findByText('1 of 3: A_2')).toBeInTheDocument()
    expect(screen.getByRole('progressbar', { name: 'Auto-generate' })).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Stop after this picture' }))
    expect(await screen.findByText('Stopped after 1 of 3.')).toBeInTheDocument()
  })

  it('names each picture that failed, with its reason', async () => {
    server.use(http.get('/api/project/apply', () => HttpResponse.json({ state: 'done', done: 2, total: 3, failed: [['abcd1234~A_2', 'raw is truncated']] })))
    open('/p/demo/s/abcd1234')
    expect(await screen.findByText('Made 2 of 3; 1 failed.')).toBeInTheDocument()
    expect(within(screen.getByRole('list', { name: 'Failures' })).getByText(/A_2/)).toBeInTheDocument()
    expect(within(screen.getByRole('list', { name: 'Failures' })).getByText(/raw is truncated/)).toBeInTheDocument()
  })

  it('says why saving the rule failed', async () => {
    server.use(http.put('/api/project/rules', () => HttpResponse.json({ detail: 'project is read-only' }, { status: 400 })))
    open('/p/demo/s/abcd1234')
    await userEvent.click(await screen.findByRole('checkbox', { name: 'Best quality (automatic)' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('project is read-only')
  })
})
