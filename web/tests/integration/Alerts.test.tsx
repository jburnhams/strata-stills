import { describe, expect, it } from 'vitest'
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { open } from '../utils/render'
import { mock, server } from '../utils/server'

const card = (name: string) => within(screen.getByRole('article', { name }))

describe('the arena flags', () => {
  it('lists what is left to fix first, and counts what is to fix and to confirm', async () => {
    open('/alerts')
    expect(await screen.findByRole('article', { name: 'pic2 / auto2' })).toBeInTheDocument()
    expect(screen.queryByRole('article', { name: 'pic1 / auto1' })).toBeNull()
    expect(screen.getByText(/1\sflag to fix, 1\sfix to confirm/)).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'To fix' })).toHaveAttribute('aria-selected', 'true')
  })

  it('shows a flag with its facts, the newest picture and the Lightroom final', async () => {
    open('/alerts?show=check')
    const c = card(await screen.findByRole('article', { name: 'pic1 / auto1' }).then(e => e.getAttribute('aria-labelledby') && 'pic1 / auto1') as string)
    expect(c.getByRole('img', { name: 'auto1 of pic1, newest' })).toHaveAttribute('src', '/api/alerts/image/a2')
    expect(c.getByRole('img', { name: 'Lightroom final of pic1' })).toHaveAttribute('src', '/api/alerts/image/a2?variant=lightroom')
    expect(c.getByText('Fixed, to confirm', { ignore: 'option' })).toBeInTheDocument()
    expect(c.getByText('right, against auto2')).toBeInTheDocument()
    expect(c.getByText('3 on the picture, 2 involving this version')).toBeInTheDocument()
    expect(c.getByText('img/1_auto1.jpg')).toBeInTheDocument()
    expect(c.getByText('abc / k1')).toBeInTheDocument()
    expect(c.getByLabelText('What is wrong')).toHaveValue('banding in the sky')
  })

  it('says when a flagged version is gone from the pool', async () => {
    open('/alerts')
    const c = card((await screen.findByRole('article', { name: 'pic2 / auto2' })) && 'pic2 / auto2')
    expect(c.getByText('Gone from the pool')).toBeInTheDocument()
    expect(c.getByLabelText('What is wrong')).toHaveValue('')
  })

  it('saves the reason and the status, and reads the list again', async () => {
    let reads = 0
    server.use(http.get('/api/alerts', async () => { reads++; return HttpResponse.json({ alerts: (await import('../utils/server')).alerts }) }))
    open('/alerts')
    await screen.findByRole('article', { name: 'pic2 / auto2' })
    const c = card('pic2 / auto2')
    await userEvent.type(c.getByLabelText('What is wrong'), 'halo round the roof')
    await userEvent.selectOptions(c.getByLabelText('Status'), 'investigating')
    await userEvent.click(c.getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(mock.statuses).toEqual([{ id: 'a1', status: 'investigating', note: 'halo round the roof' }]))
    await waitFor(() => expect(reads).toBe(2))
  })

  it('says why saving failed, under the status', async () => {
    server.use(http.post('/api/alerts/status', () => HttpResponse.json({ detail: 'unknown alert' }, { status: 404 })))
    open('/alerts')
    await screen.findByRole('article', { name: 'pic2 / auto2' })
    await userEvent.click(card('pic2 / auto2').getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(card('pic2 / auto2').getByLabelText('Status')).toHaveAccessibleDescription(/unknown alert/))
  })

  it('opens a close comparison, kept in the address', async () => {
    const { router } = open('/alerts')
    await screen.findByRole('article', { name: 'pic2 / auto2' })
    await userEvent.click(card('pic2 / auto2').getByRole('button', { name: 'Compare closely' }))
    expect(screen.getByRole('region', { name: 'auto2 against the Lightroom final of pic2' })).toBeInTheDocument()
    expect(router.state.location.search).toBe('?open=a1')
    await userEvent.click(card('pic2 / auto2').getByRole('button', { name: 'Compare closely' }))
    expect(screen.queryByRole('region', { name: 'auto2 against the Lightroom final of pic2' })).toBeNull()
  })

  it('splits the queue by what is left to do', async () => {
    open('/alerts')
    await screen.findByRole('article', { name: 'pic2 / auto2' })
    await userEvent.click(screen.getByRole('tab', { name: 'Closed' }))
    expect(await screen.findByRole('article', { name: 'pic0 / auto3' })).toBeInTheDocument()
    expect(screen.getByText("Won't fix", { ignore: 'option' })).toBeInTheDocument()
    await userEvent.click(screen.getByRole('tab', { name: 'All' }))
    expect(screen.getAllByRole('article')).toHaveLength(3)
  })

  it('teaches where flags come from when a tab is empty, and says when nothing is left', async () => {
    server.use(http.get('/api/alerts', () => HttpResponse.json({ alerts: [] })))
    open('/alerts')
    expect(await screen.findByText('No flags here')).toBeInTheDocument()
    expect(screen.getByText('Nothing to fix or confirm.')).toBeInTheDocument()
  })
})
