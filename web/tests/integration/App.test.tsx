import { describe, expect, it } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import App from '../../src/App'
import { server } from '../utils/server'

const open = (hash: string) => { window.location.hash = hash; return render(<App />) }

describe('App navigation', () => {
  it('navigates from project list to item page', async () => {
    open('#/')

    // Project list
    expect(await screen.findByText('Demo')).toBeInTheDocument()
    await userEvent.click(screen.getByText('Demo'))

    // Project page
    expect(await screen.findByText('2 stills, 1 panoramas, 1 videos')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('link', { name: /C:\\shoot/ }))

    // Source page
    const img = await screen.findByAltText('A_1')
    await userEvent.click(img)

    // Item page
    expect(await screen.findByText('★ primary')).toBeInTheDocument()
  })

  it('links to the reports only when the server has them', async () => {
    open('#/')
    await screen.findByText('Demo')
    expect(screen.queryByRole('link', { name: 'Neutral report' })).toBeNull()
  })

  it('shows the visual report link when there is a report', async () => {
    server.use(http.get('/api/report', () => HttpResponse.json({ available: true, built: 1, neutral: true, auto: true })))
    open('#/')
    const link = await screen.findByRole('link', { name: 'Neutral report' })
    expect(link.getAttribute('href')).toContain('report/neutral/')
    expect((await screen.findByRole('link', { name: 'Auto report' })).getAttribute('href')).toContain('report/auto/')
  })

  it('shows no report link when the report request fails', async () => {
    server.use(http.get('/api/report', () => new HttpResponse(null, { status: 500 })))
    open('#/')
    await screen.findByText('Demo')
    expect(screen.queryByRole('link', { name: 'Neutral report' })).toBeNull()
  })
})
