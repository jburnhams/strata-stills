import { describe, expect, it } from 'vitest'
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { open } from '../utils/render'
import { browse, server } from '../utils/server'

describe('the Browser', () => {
  it('opens folders in the tree and shows the pictures of the one chosen', async () => {
    open('/browse')
    expect(await screen.findByText('No folder chosen')).toBeInTheDocument()
    const tree = await screen.findByRole('treegrid', { name: 'Folders' })
    await userEvent.click(within(tree).getByRole('button', { name: /raw/i }))
    await userEvent.click(await within(tree).findByText('2019-03-02'))
    const grid = await screen.findByRole('grid', { name: 'Pictures' })
    expect(within(grid).getAllByRole('row')).toHaveLength(2)
    expect(within(grid).getByRole('img', { name: 'DSC00001' })).toHaveAttribute('src', '/api/browse/thumb?folder=raw%2F2019-03-02&stem=DSC00001&source=lightroom&px=320')
    expect(screen.getByText('In trip')).toBeInTheDocument()
  })

  it('adds chosen pictures to a new project', async () => {
    open('/browse?folder=raw/2019-03-02')
    const grid = await screen.findByRole('grid', { name: 'Pictures' })
    await userEvent.click(within(grid).getByRole('checkbox', { name: /DSC00002/ }))
    await userEvent.click(screen.getByRole('button', { name: /^Add 1.picture to a project/ }))
    const dialog = await screen.findByRole('dialog', { name: /^Add 1.picture to a project/ })
    expect(within(dialog).getByLabelText('New project name')).toHaveValue('2019-03-02')
    await userEvent.clear(within(dialog).getByLabelText('New project name'))
    await userEvent.click(within(dialog).getByRole('button', { name: 'Create project and add' }))
    expect(await within(dialog).findByText('Give the new project a name.')).toBeInTheDocument()
    await userEvent.type(within(dialog).getByLabelText('New project name'), 'Picks')
    await userEvent.click(within(dialog).getByRole('button', { name: 'Create project and add' }))
    await waitFor(() => expect(browse.added).toEqual([{ name: 'Picks', pictures: { 'raw/2019-03-02': ['DSC00002'] } }]))
  })

  it('adds the folder with its sub-folders to an existing project', async () => {
    open('/browse?folder=raw')
    await userEvent.click(await screen.findByRole('button', { name: 'Add with sub-folders…' }))
    const dialog = await screen.findByRole('dialog')
    await userEvent.selectOptions(within(dialog).getByLabelText('Project'), 'demo')
    await userEvent.click(within(dialog).getByRole('button', { name: 'Add to project' }))
    await waitFor(() => expect(browse.added).toEqual([{ project: 'demo', folders: ['raw'], subfolders: true }]))
  })

  it('says when pictures are still with the nudity filter, and when a folder has none', async () => {
    server.use(http.get('/api/browse/pictures', () => HttpResponse.json({ pictures: [], checking: 3 })))
    open('/browse?folder=raw/2019-03-02')
    expect(await screen.findByText(/3 pictures still being checked/)).toBeInTheDocument()
    server.use(http.get('/api/browse/pictures', () => HttpResponse.json({ pictures: [], checking: 0 })))
    open('/browse?folder=raw/empty')
    expect(await screen.findByText('No pictures here')).toBeInTheDocument()
  })

  it('sorts, groups, sizes and picks the thumbnail source from the view bar, kept in the address', async () => {
    open('/browse?folder=raw/2019-03-02')
    await screen.findByRole('grid', { name: 'Pictures' })
    await userEvent.selectOptions(screen.getByRole('combobox', { name: /Sort/ }), 'name-desc')
    await userEvent.selectOptions(screen.getByRole('combobox', { name: /Group/ }), 'type')
    expect(await screen.findByRole('grid', { name: 'Pictures, Raw + JPEG' })).toBeInTheDocument()
    expect(screen.getAllByRole('heading', { name: /^Raw\b/ })).toHaveLength(2)
    for (const v of ['time', 'time-desc', 'name']) await userEvent.selectOptions(screen.getByRole('combobox', { name: /Sort/ }), v)
    for (const v of ['hour', 'day', 'project']) await userEvent.selectOptions(screen.getByRole('combobox', { name: /Group/ }), v)
    expect(screen.getByRole('grid', { name: 'Pictures, In no project' })).toBeInTheDocument()
    await userEvent.click(screen.getByRole('radio', { name: 'Small' }))
    await userEvent.click(screen.getByRole('radio', { name: 'Camera preview' }))
    expect(screen.getByRole('img', { name: 'DSC00001' })).toHaveAttribute('src', '/api/browse/thumb?folder=raw%2F2019-03-02&stem=DSC00001&source=embedded&px=160')
  })

  it('adds the folder in view, and shows why an add failed', async () => {
    server.use(http.post('/api/browse/add', () => HttpResponse.json({ detail: 'no project called x' }, { status: 400 })))
    open('/browse?folder=raw/2019-03-02')
    await userEvent.click(await screen.findByRole('button', { name: 'Add folder…' }))
    const dialog = await screen.findByRole('dialog', { name: 'Add this folder to a project' })
    expect(within(dialog).queryByRole('button', { name: 'Add with sub-folders…' })).toBeNull()
    expect(within(dialog).getByLabelText('New project name')).toHaveValue('2019-03-02')
    await userEvent.click(within(dialog).getByRole('button', { name: 'Create project and add' }))
    expect((await within(dialog).findAllByText(/no project called x/)).length).toBeGreaterThan(0)
    await userEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }))
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  })

  it('says when nothing was new, offers to open the project, and still makes one when the projects cannot be read', async () => {
    server.use(
      http.post('/api/browse/add', () => HttpResponse.json({ project: 'picks', sources: 0, pictures: 0, unchanged: 2 })),
      http.get('/api/projects', () => HttpResponse.json({ detail: 'down' }, { status: 500 })),
      http.get('/api/browse/pictures', () => HttpResponse.json({ checking: 0, pictures: [{ stem: 'P1', kind: 'pano', files: ['P1-Pano.jpg'], taken: null, bytes: null, projects: ['a', 'b'], sources: [] }] })),
    )
    open('/browse?folder=raw/2019-03-02')
    expect(await screen.findByText('In 2 projects')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('checkbox', { name: 'Select all shown' }))
    await userEvent.click(screen.getByRole('button', { name: /^Add 1.picture to a project/ }))
    const dialog = await screen.findByRole('dialog')
    expect(await within(dialog).findByText(/could not be read/)).toBeInTheDocument()
    await userEvent.clear(within(dialog).getByLabelText('New project name'))
    await userEvent.type(within(dialog).getByLabelText('New project name'), 'Picks')
    await userEvent.click(within(dialog).getByRole('button', { name: 'Create project and add' }))
    expect(await screen.findByText('Nothing new: picks already has them')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Open project' }))
    await waitFor(() => expect(location.hash).toBe('#/p/picks'))
  })
})
