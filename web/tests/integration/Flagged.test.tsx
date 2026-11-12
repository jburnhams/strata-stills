import { describe, expect, it } from 'vitest'
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { open } from '../utils/render'
import { mock, server } from '../utils/server'

const list = async () => within(await screen.findByRole('list', { name: 'Flagged pictures' }))

describe('the flagged pictures', () => {
  it('lists every flagged picture blurred, with its class, score and whether it is allowed', async () => {
    open('/flagged')
    const l = await list()
    expect(l.getByRole('img', { name: 'beach.JPG, blurred' })).toHaveAttribute('src', '/api/flagged/image?path=C%3A%5Cshoot%5Cbeach.JPG&px=600')
    expect(l.getByText('EXPOSED')).toBeInTheDocument()
    expect(l.getByText('Allowed', { ignore: 'button *' })).toBeInTheDocument()
    expect(screen.getByText(/2\spictures at or above threshold 0\.5; 1 hidden from every output\./)).toBeInTheDocument()
  })

  it('shows a picture only when asked', async () => {
    open('/flagged')
    const l = await list()
    await userEvent.click(l.getAllByRole('button', { name: 'Show' })[0])
    expect(l.getByRole('img', { name: 'beach.JPG' })).toBeInTheDocument()
  })

  it('allows a picture back, and hides an allowed one again', async () => {
    open('/flagged')
    const l = await list()
    await userEvent.click(l.getByRole('button', { name: 'Not nudity: allow' }))
    await userEvent.click(l.getByRole('button', { name: 'Hide again' }))
    await waitFor(() => expect(mock.decisions).toEqual([{ path: 'C:\\shoot\\beach.JPG', allow: true }, { path: 'C:\\shoot\\pool.JPG', allow: false }]))
  })

  it('says why a decision failed', async () => {
    server.use(http.post('/api/flagged/decide', () => HttpResponse.json({ detail: 'not a flagged picture' }, { status: 404 })))
    open('/flagged')
    await userEvent.click((await list()).getByRole('button', { name: 'Not nudity: allow' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('not a flagged picture')
  })

  it('scans a folder and says what it found', async () => {
    open('/flagged')
    await list()
    await userEvent.type(screen.getByLabelText('Scan a folder (and below) for more'), 'D:\\raw')
    await userEvent.click(screen.getByRole('button', { name: 'Scan' }))
    await waitFor(() => expect(mock.scans).toEqual(['D:\\raw']))
    expect(await screen.findByText(/12\spictures kept, 2 flagged in all\./)).toBeInTheDocument()
  })

  it('asks for a folder, and says why a scan was refused', async () => {
    open('/flagged')
    await userEvent.click(screen.getByRole('button', { name: 'Scan' }))
    expect(await screen.findByText('Type the folder to scan.')).toBeInTheDocument()
    await userEvent.type(screen.getByLabelText('Scan a folder (and below) for more'), 'bad')
    expect(screen.queryByText('Type the folder to scan.')).toBeNull()
    await userEvent.click(screen.getByRole('button', { name: 'Scan' }))
    await waitFor(() => expect(screen.getByLabelText('Scan a folder (and below) for more')).toHaveAccessibleDescription(/not a folder: bad/))
  })

  it('says when the filter is off, and when nothing is flagged', async () => {
    server.use(http.get('/api/flagged', () => HttpResponse.json({ threshold: 0.5, enabled: false, items: [] })))
    open('/flagged')
    expect(await screen.findByText('Nothing flagged')).toBeInTheDocument()
    expect(screen.getByText(/The filter is off\./)).toBeInTheDocument()
  })
})
