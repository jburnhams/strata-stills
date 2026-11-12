import { describe, expect, it } from 'vitest'
import { fireEvent, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { open } from '../utils/render'
import { items, mock, server } from '../utils/server'

const A1 = '/p/demo/i/abcd1234~A_1'
const gen = { id: 'gen-abc123', label: 'Generated 6016×4512', path: 'C:\\p\\generated\\abc123.jpg', kind: 'generated' as const, ext: '.jpg', key: 'abc123', summary: 'JPEG q90 · size auto · levelled · sharpened 3', quality: 90, bytes: 15728640, output: [6016, 4512] as [number, number] }

describe('a picture', () => {
  it('shows the original beside the levelled picture, each saying what was done', async () => {
    open(A1)
    expect(await screen.findByRole('heading', { name: 'A_1', level: 1 })).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'Original — abcd1234~A_1' })).toHaveAttribute('src', '/api/project/image?id=demo&item=abcd1234~A_1&variant=raw&transform=original&px=1600')
    expect(screen.getByRole('img', { name: 'Levelled — abcd1234~A_1' })).toHaveAttribute('src', expect.stringContaining('transform=level'))
    expect(await screen.findByText(/Turned 1\.37° clockwise/)).toBeInTheDocument()
    expect(await screen.findByText(/as it comes from the file/)).toBeInTheDocument()
  })

  it('switches the transform and the side-by-side view', async () => {
    open(A1)
    await screen.findByRole('img', { name: 'Levelled — abcd1234~A_1' })
    await userEvent.click(screen.getByRole('switch', { name: 'Side by side with the original' }))
    expect(screen.queryByRole('img', { name: 'Original — abcd1234~A_1' })).toBeNull()
    await userEvent.click(screen.getByRole('radio', { name: 'Original' }))
    expect(screen.getByRole('img', { name: 'Original — abcd1234~A_1' })).toBeInTheDocument()
  })

  it('lists the variants, marks the primary, and makes another one primary', async () => {
    let chosen = ''
    server.use(http.put('/api/project/primary', async ({ request }) => { chosen = ((await request.json()) as { variant: string }).variant; return HttpResponse.json({ ...items[0], primary: chosen }) }))
    open(A1)
    const table = within(await screen.findByRole('grid', { name: /Variants/ }))
    expect(table.getByRole('row', { name: /Raw/ })).toHaveTextContent('Primary')
    await userEvent.click(table.getByRole('button', { name: 'Make primary Camera JPEG' }))
    await waitFor(() => expect(chosen).toBe('jpeg'))
    await waitFor(() => expect(table.getByRole('row', { name: /Camera JPEG/ })).toHaveTextContent('Primary'))
  })

  it('says why a variant could not be made primary', async () => {
    server.use(http.put('/api/project/primary', () => HttpResponse.json({ detail: 'variant is gone' }, { status: 400 })))
    open(A1)
    await userEvent.click(await screen.findByRole('button', { name: 'Make primary Camera JPEG' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('variant is gone')
  })

  it('shows another variant, from the table or the picker', async () => {
    open(A1)
    await userEvent.click(await screen.findByRole('button', { name: 'Show Camera JPEG' }))
    expect(screen.getByRole('img', { name: 'Levelled — abcd1234~A_1' })).toHaveAttribute('src', expect.stringContaining('variant=jpeg'))
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Variant shown' }), 'raw')
    expect(screen.getByRole('img', { name: 'Levelled — abcd1234~A_1' })).toHaveAttribute('src', expect.stringContaining('variant=raw'))
  })

  it('lists a generated variant with its settings, size and a download link', async () => {
    server.use(http.get('/api/project/item', () => HttpResponse.json({ ...items[0], variants: [...items[0].variants, gen] })))
    open(A1)
    const row = within(await screen.findByRole('row', { name: /Generated 6016×4512/ }))
    expect(row.getByText('JPEG q90 · size auto · levelled · sharpened 3 (q90)')).toBeInTheDocument()
    expect(row.getByText('15.0 MB')).toBeInTheDocument()
    expect(row.getByRole('link', { name: 'Download Generated 6016×4512' })).toHaveAttribute('href', '/api/project/generated/file?id=demo&item=abcd1234~A_1&variant=gen-abc123')
  })

  it('compares every variant with one zoom, once asked', async () => {
    open(A1)
    await userEvent.click(await screen.findByRole('button', { name: /Compare all 2 variants/ }))
    const grid = within(screen.getByRole('region', { name: 'Variants of A_1' }))
    expect(grid.getByRole('img', { name: 'Raw of A_1' })).toBeInTheDocument()
    expect(grid.getByRole('img', { name: 'Camera JPEG of A_1' })).toHaveAttribute('src', expect.stringContaining('variant=jpeg&transform=level'))
  })

  it('goes to the next and previous picture of the folder, by link and by arrow key', async () => {
    const { router } = open(A1)
    expect(await screen.findByRole('link', { name: 'Next →' })).toHaveAttribute('href', '/p/demo/i/abcd1234~A_2')
    expect(screen.queryByRole('link', { name: '← Previous' })).toBeNull()
    await userEvent.keyboard('{ArrowLeft}')
    expect(router.state.location.pathname).toBe(A1)
    await userEvent.keyboard('{ArrowRight}')
    await waitFor(() => expect(router.state.location.pathname).toBe('/p/demo/i/abcd1234~A_2'))
    expect(await screen.findByRole('link', { name: '← Previous' })).toBeInTheDocument()
    await userEvent.keyboard('{ArrowLeft}')
    await waitFor(() => expect(router.state.location.pathname).toBe(A1))
    expect(screen.getByLabelText('Keyboard shortcuts')).toHaveTextContent(/Previous picture.*Next picture/)
  })

  it('says a video has no still picture', async () => {
    open('/p/demo/i/abcd1234~A_3')
    expect(await screen.findByText('No still picture')).toBeInTheDocument()
    expect(screen.queryByRole('img', { name: /Levelled/ })).toBeNull()
  })

  it('says why the picture could not be read', async () => {
    server.use(http.get('/api/project/item', () => HttpResponse.json({ detail: 'no such item' }, { status: 404 })))
    open(A1)
    expect(await screen.findByText(/no such item/)).toBeInTheDocument()
  })
})

describe('the loupe', () => {
  it('follows the pointer with full-resolution crops of every variant and transform, until a click holds it', async () => {
    open(A1)
    const pic = await screen.findByTestId('pic-level')
    expect(screen.getByText('Move the pointer over the picture.')).toBeInTheDocument()
    fireEvent.mouseMove(pic, { clientX: 320, clientY: 140 })
    const loupe = within(screen.getByRole('region', { name: 'Loupe' }))
    expect(await loupe.findByRole('img', { name: 'Raw Levelled' })).toHaveAttribute('src', '/api/project/crop?id=demo&item=abcd1234~A_1&variant=raw&transform=level&x=0.5000&y=0.5000&size=160&zoom=2')
    expect(loupe.getByRole('img', { name: 'Camera JPEG Original' })).toBeInTheDocument()
    fireEvent.click(pic)
    expect(loupe.getByText('Held: click the picture again to release.')).toBeInTheDocument()
    fireEvent.mouseMove(pic, { clientX: 0, clientY: 0 })
    await new Promise(r => setTimeout(r, 200))
    expect(loupe.getByRole('img', { name: 'Raw Levelled' })).toHaveAttribute('src', expect.stringContaining('x=0.5000'))
    fireEvent.click(pic)
    expect(loupe.getByText('Following the pointer: click the picture to hold it.')).toBeInTheDocument()
  })

  it('changes the area and the zoom', async () => {
    open(A1)
    fireEvent.mouseMove(await screen.findByTestId('pic-level'), { clientX: 320, clientY: 140 })
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Loupe area' }), '240')
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Loupe zoom' }), '3')
    await waitFor(() => expect(screen.getByRole('img', { name: 'Raw Levelled' })).toHaveAttribute('src', expect.stringContaining('size=240&zoom=3')))
  })

  it('says while the full-size copy is made, and when it cannot be', async () => {
    mock.fullres = { state: 'running', text: 'developing' }
    open(A1)
    fireEvent.mouseMove(await screen.findByTestId('pic-level'), { clientX: 320, clientY: 140 })
    expect((await screen.findAllByText('Preparing the full-resolution view of Raw: developing')).length).toBe(2)
    server.use(http.get('/api/project/fullres', () => HttpResponse.json({ detail: 'raw is truncated' }, { status: 500 })))
    expect((await screen.findAllByText(/Cannot make the full-size copy/, undefined, { timeout: 4000 })).length).toBeGreaterThan(0)
  })
})
