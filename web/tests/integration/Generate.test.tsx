import { describe, expect, it } from 'vitest'
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { open } from '../utils/render'
import { generatedRecord, items, mock, presets, server } from '../utils/server'

const A1 = '/p/demo/i/abcd1234~A_1'
const willMake = () => screen.getByTestId('will-make')
const advanced = async () => userEvent.click(await screen.findByRole('button', { name: 'Advanced' }))

describe('generate', () => {
  it('needs nothing chosen: the default preset, its summary and one button; the rest is behind Advanced', async () => {
    server.use(http.get('/api/project/generate', () => HttpResponse.json(mock.generates.length ? { state: 'running', text: 'developing' } : { state: 'idle' })))
    open(A1)
    expect(await screen.findByText(/Contrast is half gone at about 2822 px across/)).toBeInTheDocument()
    await waitFor(() => expect(willMake()).toHaveTextContent('Will make: JPEG q90 · size auto · levelled · sharpened 3'))
    expect(screen.getByRole('combobox', { name: 'Pipeline' })).toHaveValue('best')
    expect(screen.getByRole('button', { name: 'Advanced' })).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByRole('combobox', { name: 'Size' })).toBeNull()
    await userEvent.click(screen.getByRole('button', { name: 'Generate' }))
    expect(mock.generates.at(-1)).toMatchObject({ variant: 'raw', preset: 'best', options: null })
    expect(await screen.findByText(/developing… \(a 48 MP raw takes a minute or two\)/)).toBeInTheDocument()
  })

  it('shows the settings of the pipeline that will run, and sends only what was changed, laid over the preset', async () => {
    open(A1); await waitFor(() => expect(willMake()).toHaveTextContent('Will make')); await advanced()
    expect(screen.getByLabelText('Size')).toHaveValue('auto'); expect(screen.getByRole('option', { name: /Recommended: 6016 px \(27\.1 MP\)/ })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: /Native: 8064 px \(48\.8 MP\)/ })).toBeInTheDocument()
    expect(screen.getByLabelText('JPEG quality')).toHaveValue('90'); expect(screen.getByLabelText('Sharpness')).toHaveValue('3'); expect(screen.getByLabelText('Highlight contrast')).toHaveValue('0.8')
    await userEvent.selectOptions(screen.getByLabelText('Format'), 'tiff'); await waitFor(() => expect(willMake()).toHaveTextContent('TIFF 16 bit'))
    expect(screen.queryByLabelText('JPEG quality')).toBeNull(); expect(willMake()).toHaveTextContent('changed from “Best quality (automatic)”')
    await userEvent.selectOptions(screen.getByLabelText('Format'), 'jpeg'); await userEvent.selectOptions(screen.getByLabelText('JPEG quality'), '95')
    await userEvent.selectOptions(screen.getByLabelText('JPEG quality'), 'auto'); await userEvent.selectOptions(screen.getByLabelText('JPEG quality'), '95')
    await userEvent.click(screen.getByRole('checkbox', { name: 'Level' }))
    await userEvent.selectOptions(screen.getByLabelText('Sharpness'), '2'); await userEvent.selectOptions(screen.getByLabelText('Highlight contrast'), '1.5')
    await userEvent.selectOptions(screen.getByLabelText('Exposure'), 'match'); await userEvent.selectOptions(screen.getByLabelText('Exposure'), '-1'); await userEvent.selectOptions(screen.getByLabelText('Exposure'), 'match')
    await waitFor(() => expect(willMake()).toHaveTextContent('JPEG q95 · size auto · sharpened 2'))
    await userEvent.click(screen.getByRole('button', { name: 'Generate' }))
    expect(mock.generates.at(-1)).toMatchObject({ preset: 'best', options: { format: 'jpeg', quality: 95, level: false, sharpen_amount: 2, highlight_contrast: 1.5, exposure: 'match' } })
    await userEvent.click(screen.getByRole('button', { name: 'Back to the preset' })); await waitFor(() => expect(willMake()).toHaveTextContent('JPEG q90 · size auto · levelled'))
  })

  it('takes off sharpening, which hides its strength', async () => {
    open(A1); await waitFor(() => expect(willMake()).toHaveTextContent('Will make')); await advanced()
    await userEvent.click(screen.getByRole('checkbox', { name: 'Sharpen to the measured blur' }))
    await waitFor(() => expect(screen.queryByLabelText('Sharpness')).toBeNull())
  })

  it('sends a custom size and another preset, and saves the settings as a preset', async () => {
    open(A1); await waitFor(() => expect(willMake()).toHaveTextContent('Will make')); await advanced()
    await userEvent.selectOptions(screen.getByLabelText('Size'), 'custom'); const f = await screen.findByLabelText('Long side in pixels'); await userEvent.clear(f); await userEvent.type(f, '3000')
    await waitFor(() => expect(willMake()).toHaveTextContent('size 3000'))
    await userEvent.type(screen.getByLabelText('Save as preset'), 'web3k'); await userEvent.click(screen.getByRole('button', { name: 'Save preset' }))
    await waitFor(() => expect(mock.saved).toHaveLength(1)); expect(mock.saved[0]).toMatchObject(['web3k', { stages: expect.arrayContaining([{ stage: 'resize', size: '3000' }]) }])
    await userEvent.selectOptions(screen.getByLabelText('Pipeline'), 'archive'); await waitFor(() => expect(willMake()).toHaveTextContent('TIFF 16 bit · size native'))
  })

  it('deletes a preset of one\'s own, and goes back to the default', async () => {
    let deleted = ''
    server.use(
      http.get('/api/pipelines', () => HttpResponse.json([...presets, { name: 'mine', label: 'Mine', builtin: false, stages: presets[0].stages }])),
      http.delete('/api/pipeline', ({ request }) => { deleted = new URL(request.url).searchParams.get('name')!; return HttpResponse.json(presets) }),
    )
    open(A1); await waitFor(() => expect(willMake()).toHaveTextContent('Will make'))
    await userEvent.selectOptions(screen.getByLabelText('Pipeline'), 'mine'); await advanced()
    await userEvent.click(screen.getByRole('button', { name: 'Delete “Mine”' }))
    await waitFor(() => expect(deleted).toBe('mine'))
    await waitFor(() => expect(screen.getByLabelText('Pipeline')).toHaveValue('best'))
  })

  it('says these settings were generated already, and shows that variant', async () => {
    mock.exists = true
    server.use(http.get('/api/project/item', () => HttpResponse.json({ ...items[0], variants: [...items[0].variants, { id: 'gen-abc123', label: 'Generated', path: 'x', kind: 'generated', ext: '.jpg' }] })))
    open(A1)
    expect(await screen.findByRole('button', { name: 'Already generated' })).toBeDisabled()
    await userEvent.click(screen.getByRole('button', { name: 'Show it' }))
    expect(screen.getByRole('combobox', { name: 'Variant shown' })).toHaveValue('gen-abc123')
  })

  it('shows an existing variant when the server answers that it exists', async () => {
    server.use(http.post('/api/project/generate', () => HttpResponse.json({ state: 'exists', variant: 'jpeg' })))
    open(A1); await waitFor(() => expect(willMake()).toHaveTextContent('Will make'))
    await userEvent.click(screen.getByRole('button', { name: 'Generate' }))
    await waitFor(() => expect(screen.getByRole('combobox', { name: 'Variant shown' })).toHaveValue('jpeg'))
  })

  it('shows the new variant once the job it watched ends', async () => {
    let polls = 0
    server.use(
      http.get('/api/project/generate', () => { if (!mock.generates.length) return HttpResponse.json({ state: 'idle' }); polls++; return HttpResponse.json(polls < 2 ? { state: 'running', text: 'developing' } : { state: 'done', result: generatedRecord }) }),
      http.get('/api/project/item', () => HttpResponse.json(polls >= 2 ? { ...items[0], variants: [...items[0].variants, { id: 'gen-abc123', label: 'Generated', path: 'x', kind: 'generated', ext: '.jpg' }] } : items[0])),
    )
    open(A1); await waitFor(() => expect(willMake()).toHaveTextContent('Will make'))
    await userEvent.click(screen.getByRole('button', { name: 'Generate' }))
    await waitFor(() => expect(screen.getByRole('combobox', { name: 'Variant shown' })).toHaveValue('gen-abc123'), { timeout: 6000 })
  }, 10000)

  it('does not jump to a variant that was made before the page opened', async () => {
    open(A1); await waitFor(() => expect(willMake()).toHaveTextContent('Will make'))
    expect(screen.getByRole('combobox', { name: 'Variant shown' })).toHaveValue('raw')
  })

  it('says why generating failed', async () => {
    server.use(http.get('/api/project/generate', () => HttpResponse.json({ state: 'failed', error: 'out of memory' })))
    open(A1)
    expect(await screen.findByText('out of memory')).toBeInTheDocument()
    expect(screen.getByText('Failed')).toBeInTheDocument()
  })

  it('says why a failed job gave no reason, and why starting one was refused', async () => {
    server.use(
      http.get('/api/project/generate', () => HttpResponse.json({ state: 'failed' })),
      http.post('/api/project/generate', () => HttpResponse.json({ detail: 'busy with another picture' }, { status: 409 })),
    )
    open(A1)
    expect(await screen.findByText('Generating failed')).toBeInTheDocument()
    await waitFor(() => expect(willMake()).toHaveTextContent('Will make'))
    await userEvent.click(screen.getByRole('button', { name: 'Generate' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('busy with another picture')
  })

  it('says why the check failed, and when the detail cannot be measured', async () => {
    server.use(
      http.post('/api/project/generate/check', () => HttpResponse.json({ detail: 'unknown preset' }, { status: 400 })),
      http.get('/api/project/detail', () => HttpResponse.json({ ok: false, reason: 'too little texture' })),
    )
    open(A1)
    expect(await screen.findByRole('alert')).toHaveTextContent('unknown preset')
    expect(screen.getByText(/Could not measure the detail \(too little texture\)/)).toBeInTheDocument()
    expect(willMake()).toHaveTextContent('Checking…')
    await advanced()
    expect(screen.getByRole('option', { name: 'Recommended' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'Native' })).toBeInTheDocument()
  })

  it('works from the source picture when a generated variant is shown', async () => {
    server.use(http.get('/api/project/item', () => HttpResponse.json({ ...items[0], primary: 'gen-abc123', variants: [...items[0].variants, { id: 'gen-abc123', label: 'Generated', path: 'x', kind: 'generated', ext: '.jpg' }] })))
    open(A1); await waitFor(() => expect(willMake()).toHaveTextContent('Will make'))
    await waitFor(() => expect(mock.checks.at(-1)).toMatchObject({ variant: null }))
    expect(within(screen.getByRole('region', { name: 'Generate the best picture' })).getByText('Measuring the real detail of this picture…')).toBeInTheDocument()
  })
})
