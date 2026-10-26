import { describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import AutoGenerate from '../../src/components/AutoGenerate'
import { server } from '../utils/server'

describe('AutoGenerate', () => {
  it('shows presets and handles starting and stopping auto-generation', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    const onDone = vi.fn()
    render(<AutoGenerate project="demo" source="abcd1234" onDone={onDone} />)

    // Default mock.rules is empty. Best quality (automatic) should be present.
    expect(await screen.findByText('Best quality (automatic)')).toBeInTheDocument()

    // Choose a preset
    await userEvent.click(screen.getByLabelText('Best quality (automatic)'))

    // Status text should be "Nothing chosen" at first before save completes, but rules are mocked to have 3 pending when saved
    await screen.findByRole('button', { name: 'Generate 3 missing' })

    // Start generating
    await userEvent.click(screen.getByRole('button', { name: 'Generate 3 missing' }))
    expect(await screen.findByRole('button', { name: 'Generating…' })).toBeInTheDocument()

    // Check status
    expect(await screen.findByText(/starting/)).toBeInTheDocument()

    // Stop generating
    await userEvent.click(screen.getByRole('button', { name: 'Stop after this picture' }))

    // Wait for stop state and onDone
    vi.advanceTimersByTime(2000)
    await waitFor(() => expect(onDone).toHaveBeenCalled())

    vi.useRealTimers()
  })

  it('displays API errors on initial load', async () => {
    server.use(http.get('/api/project/rules', () => HttpResponse.json({ detail: 'Failed to fetch project rules' }, { status: 500 })))
    render(<AutoGenerate project="demo" source="abcd1234" onDone={vi.fn()} />)
    expect(await screen.findByRole('alert')).toHaveTextContent('Failed to fetch project rules')
  })

  it('handles failed pictures in apply job', async () => {
    server.use(http.post('/api/project/apply', () => HttpResponse.json({ state: 'done', done: 1, total: 2, failed: [['abcd1234~A_1', 'Failed to generate']] })))
    server.use(http.get('/api/project/rules', () => HttpResponse.json({ rules: [{ source: 'abcd1234', pipeline: 'best' }], pending: 2 })))

    const { unmount } = render(<AutoGenerate project="demo" source="abcd1234" onDone={vi.fn()} />)

    await screen.findByText('Best quality (automatic)')

    await userEvent.click(await screen.findByRole('button', { name: 'Generate 2 missing' }))

    expect(await screen.findByText('A_1: Failed to generate')).toBeInTheDocument()
    unmount()
  })
})
