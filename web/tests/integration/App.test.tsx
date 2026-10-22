import { describe, expect, it } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import App from '../../src/App'
import { server } from '../utils/server'

const open = (hash: string) => { window.location.hash = hash; return render(<App />) }

describe('project list', () => {
  it('opens on the projects to choose from, and shows the projects folder', async () => {
    open('#/')
    expect(await screen.findByText('Demo')).toBeInTheDocument()
    expect(screen.getByText('C:\\projects')).toBeInTheDocument()
    expect(screen.getByText('C:\\shoot')).toBeInTheDocument()
  })
  it('creates a project and goes into it', async () => {
    open('#/'); await screen.findByText('Demo')
    await userEvent.type(screen.getByLabelText('Project name'), 'Fresh'); await userEvent.click(screen.getByRole('button', { name: 'Create project' }))
    await waitFor(() => expect(window.location.hash).toBe('#/p/fresh'))
  })
  it('shows an empty state and an error', async () => {
    server.use(http.get('/api/projects', () => HttpResponse.json([])))
    open('#/'); expect(await screen.findByText(/No projects yet/)).toBeInTheDocument()
  })
})

describe('sources', () => {
  it('lists the source folders with what is in them and links into the contents', async () => {
    open('#/p/demo')
    expect(await screen.findByText('2 stills, 1 panoramas, 1 videos')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /C:\\shoot/ })).toHaveAttribute('href', '#/p/demo/s/abcd1234')
  })
  it('reports a refused folder', async () => {
    open('#/p/demo'); await screen.findByText(/2 stills/)
    await userEvent.type(screen.getByLabelText('Source folder'), 'bad'); await userEvent.click(screen.getByRole('button', { name: 'Add source' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('not a folder: bad')
  })
  it('marks an unreachable folder', async () => {
    server.use(http.get('/api/project/sources', () => HttpResponse.json([{ id: 'x', path: '\\\\Server\\off', reachable: false, stills: 0, panos: 0, videos: 0 }])))
    open('#/p/demo'); expect(await screen.findByText(/Not reachable/)).toBeInTheDocument()
  })
})

describe('contents', () => {
  it('shows a thumbnail per picture, hides videos until asked, and links to the item', async () => {
    open('#/p/demo/s/abcd1234')
    const img = await screen.findByAltText('A_1'); expect(img.getAttribute('src')).toContain('/api/project/image?id=demo&item=abcd1234~A_1&variant=raw')
    expect(screen.queryByText('video')).toBeNull()
    await userEvent.click(screen.getByLabelText(/Show 1 videos/)); expect(screen.getAllByText('video').length).toBe(2)
    expect(img.closest('a')).toHaveAttribute('href', '#/p/demo/i/abcd1234~A_1')
  })
})

describe('item', () => {
  it('shows the variants with the primary marked, and the levelled picture beside the original', async () => {
    open('#/p/demo/i/abcd1234~A_1')
    expect(await screen.findByText('★ primary')).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Camera JPEG' })).toBeInTheDocument()
    expect(await screen.findByText(/Turned 1\.37° clockwise/)).toBeInTheDocument()
    expect(screen.getByAltText(/Original/)).toBeInTheDocument(); expect(screen.getByAltText(/Levelled/)).toBeInTheDocument()
  })
  it('makes another variant primary', async () => {
    open('#/p/demo/i/abcd1234~A_1'); await screen.findByText('★ primary')
    await userEvent.click(screen.getByRole('button', { name: 'Make primary' }))
    await waitFor(() => expect(screen.queryByRole('alert')).toBeNull())
  })
  it('switches to the original tab', async () => {
    open('#/p/demo/i/abcd1234~A_1'); await screen.findByText(/Turned 1\.37/)
    await userEvent.click(screen.getByRole('tab', { name: /Original/ })); expect(screen.getByRole('tab', { name: /Original/ })).toHaveAttribute('aria-selected', 'true')
    expect(screen.queryByAltText(/Levelled/)).toBeNull()
  })
})

describe('export', () => {
  it('offers the measured size, exports with the chosen options and lists the result', async () => {
    const { exportCalls } = await import('../utils/server') as { exportCalls: unknown[] }
    open('#/p/demo/i/abcd1234~A_1')
    expect(await screen.findByText(/Contrast is half gone at about 2822 px across/)).toBeInTheDocument()
    expect(screen.getByRole('option', { name: /Recommended: 6016 px \(27\.1 MP\)/ })).toBeInTheDocument()
    expect(await screen.findByText('A_1-6016x4512.jpg')).toHaveAttribute('href', expect.stringContaining('/api/project/export/file?id=demo&name=A_1-6016x4512.jpg'))
    await userEvent.selectOptions(screen.getByLabelText('Format'), 'tiff'); await userEvent.click(screen.getByLabelText('Level'))
    await userEvent.click(screen.getByRole('button', { name: 'Export' }))
    expect(exportCalls.at(-1)).toMatchObject({ variant: 'raw', size: 'auto', format: 'tiff', level: false, sharpen: true, sharpen_amount: 1.5, highlight_contrast: 0.8 })
    expect(await screen.findByText(/Done: A_1-6016x4512.jpg/, {}, { timeout: 4000 })).toBeInTheDocument()
  })
  it('sends the chosen sharpness', async () => {
    const { exportCalls } = await import('../utils/server') as { exportCalls: unknown[] }
    open('#/p/demo/i/abcd1234~A_1'); await screen.findByText(/Contrast is half gone/)
    await userEvent.selectOptions(screen.getByLabelText('Sharpness'), '2'); await userEvent.click(screen.getByRole('button', { name: 'Export' })); await waitFor(() => expect(exportCalls.at(-1)).toMatchObject({ sharpen_amount: 2 }))
  })
  it('sends a custom size', async () => {
    const { exportCalls } = await import('../utils/server') as { exportCalls: unknown[] }
    open('#/p/demo/i/abcd1234~A_1'); await screen.findByText(/Contrast is half gone/)
    await userEvent.selectOptions(screen.getByLabelText('Size'), 'custom'); const f = screen.getByLabelText('Long side in pixels'); await userEvent.clear(f); await userEvent.type(f, '3000')
    await userEvent.click(screen.getByRole('button', { name: 'Export' })); await waitFor(() => expect(exportCalls.at(-1)).toMatchObject({ size: '3000' }))
  })
})
