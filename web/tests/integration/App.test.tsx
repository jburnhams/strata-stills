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
})
