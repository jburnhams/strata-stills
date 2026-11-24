import { describe, expect, it } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { open } from '../utils/render'
import { mock, task } from '../utils/server'

describe('the jobs page', () => {
  it('teaches how to start a job when there are none', async () => {
    open('/jobs')
    expect(await screen.findByText('No jobs yet')).toBeInTheDocument()
  })

  it('lists each job in words with its state, and cancels or runs one again', async () => {
    mock.jobs = [
      task('apply', 'running', { id: 2, domain: 'demo', progress: { done: 1, total: 3, text: 'A_2' } }),
      task('generate', 'failed', { id: 1, domain: 'demo/abcd1234~A_1', error: 'out of memory' }),
    ]
    open('/jobs')
    expect(await screen.findByText('Auto-generate demo')).toBeInTheDocument()
    expect(screen.getByText('Generate A_1 in demo')).toBeInTheDocument()
    expect(screen.getByText(/out of memory/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: /Cancel.*Auto-generate demo/ }))
    await waitFor(() => expect(mock.cancels).toEqual([2]))
    await userEvent.click(screen.getByRole('button', { name: /Run again.*Generate A_1 in demo/ }))
    await waitFor(() => expect(mock.retries).toEqual([1]))
  })
})
