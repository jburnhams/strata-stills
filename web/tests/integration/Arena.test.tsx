import { describe, expect, it } from 'vitest'
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { format } from 'strata-react'
import { open } from '../utils/render'
import { mock, server } from '../utils/server'
import { images } from '../utils/images'

const pair = async () => within(await screen.findByRole('region', { name: 'Two versions of one picture' }))

describe('the arena', () => {
  it('shows a pair blind, as Left and Right under neutral addresses, with the counts', async () => {
    open('/arena')
    const p = await pair()
    expect(p.getByRole('img', { name: 'Left' })).toHaveAttribute('src', '/arena/img/tok1/left')
    expect(p.getByRole('img', { name: 'Right' })).toHaveAttribute('src', '/arena/img/tok1/right')
    expect(screen.getByText(/^6\svotes,\s3\spictures$/)).toBeInTheDocument()
    await waitFor(() => expect(mock.pairs).toBe(2)) // the next pair is fetched while this one is judged
  })

  it('shows a pair, and takes answers, only once both its pictures have loaded', async () => {
    images.hold()
    open('/arena')
    await waitFor(() => expect(images.held()).toEqual(['/arena/img/tok1/left', '/arena/img/tok1/right']))
    await userEvent.keyboard('{ArrowLeft}')
    expect(screen.queryByRole('button', { name: 'Left is better' })).not.toBeInTheDocument()
    expect(mock.votes).toHaveLength(0)
    images.release()
    await pair()
    await userEvent.keyboard('{ArrowLeft}')
    await waitFor(() => expect(mock.votes[0]).toMatchObject({ token: 'tok1', choice: 'left' }))
  })

  it('records a vote from a button, with the time spent, and moves to the prefetched pair', async () => {
    open('/arena')
    await pair(); await waitFor(() => expect(mock.pairs).toBe(2))
    await userEvent.click(screen.getByRole('button', { name: 'Left is better' }))
    await waitFor(() => expect((mock.votes[0] as { token: string; choice: string; ms: number })).toMatchObject({ token: 'tok1', choice: 'left' }))
    expect(typeof (mock.votes[0] as { ms: number }).ms).toBe('number')
    expect((await pair()).getByRole('img', { name: 'Left' })).toHaveAttribute('src', '/arena/img/tok2/left')
    expect(await screen.findByText(/^7\svotes,\s3\spictures$/)).toBeInTheDocument()
  })

  it('answers from the keyboard: arrows, down or E for equal, S to skip', async () => {
    open('/arena')
    await pair()
    for (const [key, choice] of [['{ArrowRight}', 'right'], ['{ArrowDown}', 'tie'], ['e', 'tie'], ['s', 'skip'], ['{ArrowLeft}', 'left']] as const) {
      const n = mock.votes.length
      await userEvent.keyboard(key)
      await waitFor(() => expect(mock.votes).toHaveLength(n + 1))
      expect(mock.votes[n]).toMatchObject({ choice })
      await waitFor(() => expect(screen.getByRole('img', { name: 'Left' })).toHaveAttribute('src', `/arena/img/tok${n + 2}/left`))
    }
    expect(screen.getByLabelText('Keyboard shortcuts')).toHaveTextContent(/Left is better.*Right is better.*Equal.*Skip.*Undo last vote.*Flag left.*Flag right/)
  })

  it('answers the other buttons too', async () => {
    open('/arena')
    await pair()
    await userEvent.click(screen.getByRole('button', { name: 'Equal' })); await waitFor(() => expect(mock.votes).toHaveLength(1))
    await waitFor(() => expect(screen.getByRole('img', { name: 'Left' })).toHaveAttribute('src', '/arena/img/tok2/left'))
    await userEvent.click(screen.getByRole('button', { name: 'Right is better' })); await waitFor(() => expect(mock.votes).toHaveLength(2))
    await waitFor(() => expect(screen.getByRole('img', { name: 'Left' })).toHaveAttribute('src', '/arena/img/tok3/left'))
    await userEvent.click(screen.getByRole('button', { name: 'Skip' })); await waitFor(() => expect(mock.votes).toHaveLength(3))
    expect(mock.votes.map(v => (v as { choice: string }).choice)).toEqual(['tie', 'right', 'skip'])
  })

  it('undoes the last vote with Z, says so, and shows a fresh pair', async () => {
    open('/arena')
    await pair()
    await userEvent.keyboard('z')
    await waitFor(() => expect(mock.undos).toBe(1))
    expect(await screen.findByText('Vote undone')).toBeInTheDocument()
    expect(await screen.findByText(/^5\svotes,\s3\spictures$/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Undo last vote' }))
    await waitFor(() => expect(mock.undos).toBe(2))
  })

  it('says why an undo was refused', async () => {
    server.use(http.post('/api/arena/undo', () => HttpResponse.json({ detail: 'no vote to undo' }, { status: 404 })))
    open('/arena')
    await pair()
    await userEvent.keyboard('z')
    expect(await screen.findByRole('alert')).toHaveTextContent('no vote to undo')
  })

  it('flags a side and takes the flag back, by button and by 1 and 2', async () => {
    open('/arena')
    await pair()
    const left = screen.getByRole('button', { name: 'Flag left' })
    await userEvent.click(left)
    await waitFor(() => expect(left).toHaveAttribute('aria-pressed', 'true'))
    expect(mock.flags[0]).toMatchObject({ token: 'tok1', side: 'left', note: '' })
    await userEvent.keyboard('1')
    await waitFor(() => expect(left).toHaveAttribute('aria-pressed', 'false'))
    expect(mock.unflags[0]).toMatchObject({ token: 'tok1', side: 'left' })
    await userEvent.keyboard('2')
    await waitFor(() => expect(screen.getByRole('button', { name: 'Flag right' })).toHaveAttribute('aria-pressed', 'true'))
    await userEvent.click(screen.getByRole('button', { name: 'Flag right' }))
    await waitFor(() => expect(mock.unflags).toHaveLength(2))
  })

  it('clears the flags when the next pair comes', async () => {
    open('/arena')
    await pair()
    await userEvent.keyboard('2')
    await waitFor(() => expect(screen.getByRole('button', { name: 'Flag right' })).toHaveAttribute('aria-pressed', 'true'))
    await userEvent.keyboard('s')
    await waitFor(() => expect(screen.getByRole('img', { name: 'Left' })).toHaveAttribute('src', '/arena/img/tok2/left'))
    expect(screen.getByRole('button', { name: 'Flag right' })).toHaveAttribute('aria-pressed', 'false')
  })

  it('says why a vote was refused, and stays on the pair', async () => {
    server.use(http.post('/api/arena/vote', () => HttpResponse.json({ detail: 'unknown pair (already answered?)' }, { status: 404 })))
    open('/arena')
    await pair()
    await userEvent.keyboard('{ArrowLeft}')
    expect(await screen.findByRole('alert')).toHaveTextContent('unknown pair (already answered?)')
    expect(screen.getByRole('img', { name: 'Left' })).toHaveAttribute('src', '/arena/img/tok1/left')
  })

  it('passes the set and the wanted versions on to the server', async () => {
    const asked: string[] = []
    server.use(http.get('/api/arena/next', ({ request }) => { asked.push(new URL(request.url).search); return HttpResponse.json({ token: 't', left: 'img/t/left', right: 'img/t/right', pool: 1, votes: 0 }) }))
    open('/arena?set=hard&want=auto1&other=x')
    await pair()
    expect(asked[0]).toBe('?set=hard&want=auto1')
  })

  it('says there is nothing to rate, in the server\'s words, and binds no answers', async () => {
    server.use(http.get('/api/arena/next', () => HttpResponse.json({ detail: 'nothing to rate yet: make the pictures with scripts/build_arena.py' }, { status: 404 })))
    open('/arena')
    expect(await screen.findByText('Nothing to rate')).toBeInTheDocument()
    expect(screen.getByText(/scripts\/build_arena\.py/)).toBeInTheDocument()
    await userEvent.keyboard('{ArrowLeft}1')
    expect(mock.votes).toEqual([]); expect(mock.flags).toEqual([])
  })

  it('uses a plain sentence when the server gives no reason, and shows a failed read with Try again', async () => {
    server.use(http.get('/api/arena/next', () => new HttpResponse(null, { status: 404 })))
    const { unmount } = open('/arena')
    expect(await screen.findByText('There is nothing to rate yet.')).toBeInTheDocument()
    unmount()
    server.use(http.get('/api/arena/next', () => HttpResponse.json({ detail: 'database is down' }, { status: 500 })))
    open('/arena')
    expect(await screen.findByRole('button', { name: 'Try again' })).toBeInTheDocument()
  })

  it('opens the ratings: each version with its range, and head to head', async () => {
    open('/arena')
    await pair()
    await userEvent.click(screen.getByRole('button', { name: 'Ratings' }))
    const table = within(await screen.findByRole('grid', { name: 'Ratings of each version' }))
    expect(table.getByRole('row', { name: /lightroom/ })).toHaveTextContent(/1,?620.*1,?550 to 1,?700.*1,?580.*5/)
    expect(table.getByRole('row', { name: /auto1/ })).toHaveTextContent(format.missing)
    expect(screen.getByRole('grid', { name: 'Head to head' })).toHaveTextContent('lightroom vs auto1')
    expect(screen.getByText(/older vote on remade pictures left out/)).toBeInTheDocument()
  })

  it('reads the ratings again after a vote', async () => {
    let reads = 0
    server.use(http.get('/api/arena/ratings', () => { reads++; return HttpResponse.json({ votes: 0, superseded_votes: 0, pictures: 0, voted_pictures: 0, retired: [], left_odds: 1, variants: [], head_to_head: [] }) }))
    open('/arena')
    await pair()
    await userEvent.click(screen.getByRole('button', { name: 'Ratings' }))
    await waitFor(() => expect(reads).toBe(1))
    expect(screen.queryByText(/older vote/)).toBeNull()
    await userEvent.keyboard('s')
    await waitFor(() => expect(reads).toBe(2))
  })

  it('links back to the projects and to the flags', async () => {
    open('/arena')
    expect(screen.getByRole('link', { name: 'Projects' })).toHaveAttribute('href', '/')
    expect(screen.getByRole('link', { name: 'Arena flags' })).toHaveAttribute('href', '/alerts')
  })
})
