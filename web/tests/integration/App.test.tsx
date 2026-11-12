import { describe, expect, it } from 'vitest'
import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { open } from '../utils/render'
import { server } from '../utils/server'

describe('the app', () => {
  it('goes from the project list to a folder to a picture', async () => {
    const { router } = open('/')
    await userEvent.click(await screen.findByRole('link', { name: 'Demo' }))
    expect(router.state.location.pathname).toBe('/p/demo')
    await userEvent.click(await screen.findByRole('link', { name: 'C:\\shoot' }))
    expect(router.state.location.pathname).toBe('/p/demo/s/abcd1234')
    await userEvent.click(await screen.findByRole('link', { name: /A_1/ }))
    expect(router.state.location.pathname).toBe('/p/demo/i/abcd1234~A_1')
    expect(await screen.findByRole('heading', { name: 'A_1', level: 1 })).toBeInTheDocument()
  })

  it('names every screen in the side nav and marks the current one', async () => {
    open('/flagged')
    const nav = screen.getByRole('navigation', { name: 'Main' })
    expect(within(nav).getByRole('link', { name: 'Flagged pictures' })).toHaveAttribute('aria-current', 'page')
    for (const name of ['Projects', 'Arena', 'Arena flags']) expect(within(nav).getByRole('link', { name })).toBeInTheDocument()
  })

  it('sends an unknown address to the projects', async () => {
    const { router } = open('/nowhere')
    expect(await screen.findByRole('heading', { name: 'Projects' })).toBeInTheDocument()
    expect(router.state.location.pathname).toBe('/')
  })

  it('links to the reports only when the server has them', async () => {
    open('/')
    await screen.findByRole('link', { name: 'Demo' })
    expect(screen.queryByRole('link', { name: 'Neutral report' })).toBeNull()
  })

  it('links to each report the server has', async () => {
    server.use(http.get('/api/report', () => HttpResponse.json({ available: true, built: 1, neutral: true, auto: true })))
    open('/')
    expect(await screen.findByRole('link', { name: 'Neutral report' })).toHaveAttribute('href', '/report/neutral/')
    expect(screen.getByRole('link', { name: 'Auto report' })).toHaveAttribute('href', '/report/auto/')
  })

  it('offers the light and dark choice in the top bar', async () => {
    open('/')
    expect(screen.getByRole('radiogroup', { name: 'Appearance' })).toBeInTheDocument()
  })
})
