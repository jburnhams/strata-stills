import { describe, expect, it } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import SourcePage from '../../src/components/SourcePage'
import { server } from '../utils/server'

describe('SourcePage', () => {
  it('shows a thumbnail per picture, hides videos until asked, and links to the item', async () => {
    render(<SourcePage project="demo" source="abcd1234" />)
    const img = await screen.findByAltText('A_1')
    expect(img.getAttribute('src')).toContain('/api/project/image?id=demo&item=abcd1234~A_1&variant=raw')
    expect(screen.queryByText('video', { selector: 'span' })).toBeNull()

    await userEvent.click(screen.getByLabelText(/Show 1 videos/))
    expect(screen.getAllByText('video').length).toBe(2)

    expect(img.closest('a')).toHaveAttribute('href', '#/p/demo/i/abcd1234~A_1')
  })

  it('shows loading state initially', () => {
    render(<SourcePage project="demo" source="abcd1234" />)
    expect(screen.getByText('Loading…')).toBeInTheDocument()
  })

  it('shows an error state if fetching items fails', async () => {
    server.use(http.get('/api/project/items', () => HttpResponse.json({ detail: 'Failed to fetch items' }, { status: 500 })))
    render(<SourcePage project="demo" source="abcd1234" />)
    expect(await screen.findByRole('alert')).toHaveTextContent('Failed to fetch items')
  })

  it('shows empty state if there are no items', async () => {
    server.use(http.get('/api/project/items', () => HttpResponse.json([])))
    render(<SourcePage project="demo" source="abcd1234" />)
    await waitFor(() => expect(screen.queryByText('Loading…')).toBeNull())
    expect(screen.queryByAltText('A_1')).toBeNull()
  })
})
