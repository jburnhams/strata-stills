import { describe, expect, it } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import App from '../../src/App'
import { generatedRecord, items, mock, presets, server } from '../utils/server'

const open = (hash: string) => { window.location.hash = hash; return render(<App />) }
const willMake = () => screen.getByTestId('will-make')

describe('generate', () => {
  it('needs nothing chosen: the default preset, its summary, one button; everything else is behind Advanced', async () => {
    open('#/p/demo/i/abcd1234~A_1')
    expect(await screen.findByText(/Contrast is half gone at about 2822 px across/)).toBeInTheDocument()
    await waitFor(() => expect(willMake()).toHaveTextContent('Will make: JPEG q90 · size auto · levelled · sharpened 3'))
    expect(screen.getByRole('combobox', { name: 'Pipeline' })).toHaveValue('best'); expect(screen.getByRole('button', { name: 'Generate' })).toBeEnabled()
    expect(screen.getByText('Advanced').closest('details')).not.toHaveAttribute('open')                                  // collapsed by default
    await userEvent.click(screen.getByRole('button', { name: 'Generate' }))
    expect(mock.generates.at(-1)).toMatchObject({ variant: 'raw', preset: 'best', options: null })
    expect(await screen.findByText(/Generating/)).toBeInTheDocument()
  })

  it('shows the Advanced settings of the pipeline that will run, and sends only what was changed, laid over the preset', async () => {
    open('#/p/demo/i/abcd1234~A_1'); await waitFor(() => expect(willMake()).toHaveTextContent('Will make'))
    await userEvent.click(screen.getByText('Advanced'))
    expect(screen.getByLabelText('Size')).toHaveValue('auto'); expect(screen.getByRole('option', { name: /Recommended: 6016 px \(27\.1 MP\)/ })).toBeInTheDocument()
    expect(screen.getByLabelText('JPEG quality')).toHaveValue('90'); expect(screen.getByLabelText('Sharpness')).toHaveValue('3'); expect(screen.getByLabelText('Highlight contrast')).toHaveValue('0.8')
    await userEvent.selectOptions(screen.getByLabelText('Format'), 'tiff'); await waitFor(() => expect(willMake()).toHaveTextContent('TIFF 16 bit'))
    expect(screen.queryByLabelText('JPEG quality')).toBeNull(); expect(willMake()).toHaveTextContent('changed from “Best quality (automatic)”')
    await userEvent.selectOptions(screen.getByLabelText('Format'), 'jpeg'); await userEvent.selectOptions(screen.getByLabelText('JPEG quality'), '95'); await userEvent.click(screen.getByLabelText('Level'))
    await userEvent.selectOptions(screen.getByLabelText('Sharpness'), '2'); await userEvent.selectOptions(screen.getByLabelText('Highlight contrast'), '1.5'); await userEvent.selectOptions(screen.getByLabelText('Exposure'), 'match')
    await waitFor(() => expect(willMake()).toHaveTextContent('JPEG q95 · size auto · sharpened 2'))
    await userEvent.click(screen.getByRole('button', { name: 'Generate' }))
    expect(mock.generates.at(-1)).toMatchObject({ preset: 'best', options: { format: 'jpeg', quality: 95, level: false, sharpen_amount: 2, highlight_contrast: 1.5, exposure: 'match' } })
    await userEvent.click(screen.getByRole('button', { name: 'Back to the preset' })); await waitFor(() => expect(willMake()).toHaveTextContent('JPEG q90 · size auto · levelled'))
  })

  it('sends a custom size and another preset, and can save the settings as a preset', async () => {
    open('#/p/demo/i/abcd1234~A_1'); await waitFor(() => expect(willMake()).toHaveTextContent('Will make')); await userEvent.click(screen.getByText('Advanced'))
    await userEvent.selectOptions(screen.getByLabelText('Size'), 'custom'); const f = await screen.findByLabelText('Long side in pixels'); await userEvent.clear(f); await userEvent.type(f, '3000')
    await waitFor(() => expect(willMake()).toHaveTextContent('size 3000'))
    await userEvent.type(screen.getByLabelText('Preset name'), 'web3k'); await userEvent.click(screen.getByRole('button', { name: 'Save preset' }))
    await waitFor(() => expect(mock.saved).toHaveLength(1)); expect(mock.saved[0]).toMatchObject(['web3k', { stages: expect.arrayContaining([{ stage: 'resize', size: '3000' }]) }])
    await userEvent.selectOptions(screen.getByLabelText('Pipeline'), 'archive'); await waitFor(() => expect(willMake()).toHaveTextContent('TIFF 16 bit · size native'))
  })

  it('greys Generate out when these settings have been generated, and shows the existing variant', async () => {
    mock.exists = true; open('#/p/demo/i/abcd1234~A_1')
    const done = await screen.findByRole('button', { name: 'Already generated' }); expect(done).toBeDisabled()
    expect(screen.queryByRole('button', { name: 'Generate' })).toBeNull()
    await userEvent.click(screen.getByRole('button', { name: 'Show it' }))                                                  // selects the generated variant (which the mock item does not list: nothing to crash on)
  })

  it('lists a generated variant with its settings, size and a download link, and compares it in the loupe', async () => {
    const gen = { id: 'gen-abc123', label: 'Generated 6016×4512', path: 'C:\\p\\generated\\abc123.jpg', kind: 'generated' as const, ext: '.jpg', key: 'abc123', summary: 'JPEG q90 · size auto · levelled · sharpened 3', quality: 90, bytes: 15728640, output: [6016, 4512] as [number, number] }
    server.use(http.get('/api/project/item', () => HttpResponse.json({ ...items[0], variants: [...items[0].variants, gen] })))
    open('#/p/demo/i/abcd1234~A_1')
    const card = (await screen.findByRole('radio', { name: 'Generated 6016×4512' })).parentElement!
    expect(within(card).getByText(/JPEG q90 · size auto · levelled · sharpened 3 \(q90\) · 15\.0 MB/)).toBeInTheDocument()
    expect(within(card).getByRole('link', { name: 'Download' })).toHaveAttribute('href', '/api/project/generated/file?id=demo&item=abcd1234~A_1&variant=gen-abc123')
    expect(within(card).getByRole('button', { name: 'Make primary' })).toBeInTheDocument()
    await userEvent.click(screen.getByRole('radio', { name: 'Generated 6016×4512' }))                                       // choosing it as the one shown: generating then starts from a source variant, not from it
    await waitFor(() => expect(willMake()).toHaveTextContent('Will make')); expect(mock.checks.at(-1)).toMatchObject({ variant: null })
  })
})

describe('auto-generate a source folder', () => {
  it('chooses presets for the folder, then generates what is missing and reports the progress', async () => {
    open('#/p/demo/s/abcd1234')
    expect(await screen.findByRole('button', { name: 'Nothing chosen' })).toBeDisabled()
    await userEvent.click(await screen.findByLabelText('Best quality (automatic)'))
    expect(mock.rules).toEqual([{ source: 'abcd1234', pipeline: 'best' }])
    const go = await screen.findByRole('button', { name: 'Generate 3 missing' }); await userEvent.click(go)
    expect(mock.applies).toBe(1); expect(await screen.findByText(/done: 3 of 3/, {}, { timeout: 4000 })).toBeInTheDocument()
    await userEvent.click(screen.getByLabelText('Best quality (automatic)')); expect(mock.rules).toEqual([])
  })
  it('reports a failed picture', async () => {
    mock.rules = [{ source: 'abcd1234', pipeline: 'best' }]
    server.use(http.get('/api/project/apply', () => HttpResponse.json({ state: 'done', done: 3, total: 3, failed: [['abcd1234~A_2', 'boom']] })))
    open('#/p/demo/s/abcd1234'); await userEvent.click(await screen.findByRole('button', { name: 'Generate 3 missing' }))
    expect(await screen.findByText('A_2: boom', {}, { timeout: 4000 })).toBeInTheDocument(); expect(screen.getByText(/1 failed/)).toBeInTheDocument()
  })
})

describe('generate: the background job and saved presets', () => {
  it('polls a running job and stops showing it as running once it is done', async () => {
    let polls = 0
    server.use(http.get('/api/project/generate', () => { polls += 1; return HttpResponse.json({ state: 'done', result: generatedRecord }) }))
    open('#/p/demo/i/abcd1234~A_1'); await waitFor(() => expect(willMake()).toHaveTextContent('Will make'))
    await userEvent.click(screen.getByRole('button', { name: 'Generate' }))
    expect(await screen.findByText(/Generating/)).toBeInTheDocument()
    await waitFor(() => expect(screen.queryByText(/Generating/)).toBeNull(), { timeout: 5000 })           // the poll runs every 1.5 s
    expect(polls).toBeGreaterThanOrEqual(1)
  })

  it('keeps polling while the job runs and shows the server error if asking for its state fails', async () => {
    server.use(http.get('/api/project/generate', () => HttpResponse.json({ detail: 'the job vanished' }, { status: 500 })))
    open('#/p/demo/i/abcd1234~A_1'); await waitFor(() => expect(willMake()).toHaveTextContent('Will make'))
    await userEvent.click(screen.getByRole('button', { name: 'Generate' }))
    expect(await screen.findByText('the job vanished', {}, { timeout: 5000 })).toBeInTheDocument()
  })

  it('deletes a preset the user saved, and goes back to the best-quality one', async () => {
    const mine = { name: 'web3k', label: 'Web 3k', builtin: false, stages: presets[0].stages }; const deleted: string[] = []
    server.use(
      http.get('/api/pipelines', () => HttpResponse.json([...presets, mine])),
      http.delete('/api/pipeline', ({ request }) => { deleted.push(new URL(request.url).searchParams.get('name')!); return HttpResponse.json(presets) }),
    )
    open('#/p/demo/i/abcd1234~A_1'); await waitFor(() => expect(willMake()).toHaveTextContent('Will make'))
    await userEvent.click(screen.getByText('Advanced')); expect(screen.queryByRole('button', { name: /^Delete/ })).toBeNull()      // nothing to delete on a built-in preset
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Pipeline' }), 'web3k')
    await userEvent.click(await screen.findByRole('button', { name: 'Delete “Web 3k”' }))
    await waitFor(() => expect(deleted).toEqual(['web3k']))
    await waitFor(() => expect(screen.getByRole('combobox', { name: 'Pipeline' })).toHaveValue('best'))
  })
})
