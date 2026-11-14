// Generating a picture is running a pipeline preset on the item's source: one button, nothing to choose (the default preset is the automatic best).
// Every setting is behind "Advanced" (a Disclosure, mounted only while open), laid over the preset. The result is a new variant of the item (listed with the others, comparable in the loupe);
// the same settings twice find the first one, and the button says so. The server's generate/check says what will run, so this panel never repeats the preset logic.
// Generating is a server job: useJob polls its status while it runs and reads the item again once when it ends.
import { useEffect, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { Alert, AsyncButton, Button, Checkbox, Disclosure, Field, Input, JobStatus, NativeSelect, useAsyncAction, useJob, useLoad } from 'strata-react'
import { api, hasStage, stageValue, type Detail, type GenerateBody, type Options } from '../api'

export const mp = (w: number, h: number) => (w * h / 1e6).toFixed(1) + ' MP'
export const mb = (b: number) => (b / 1048576).toFixed(1) + ' MB'

/** What the measurement says, in words: how many pixels of the picture are real detail. */
export function detailText(d: Detail | null): string {
  if (!d) return 'Measuring the real detail of this picture…'
  if (!d.ok || !d.native) return d.reason ? `Could not measure the detail (${d.reason}); native size is offered.` : 'Could not measure the detail.'
  const long = Math.max(...d.native), px = (s?: number) => Math.round((s ?? 0) * long)
  return `Native ${d.native[0]}×${d.native[1]}. Contrast is half gone at about ${px(d.scale50)} px across and 30 % at ${px(d.scale30)} px: beyond that the pixels only enlarge the blur.`
}

const BEST = { name: 'best', label: 'Best quality (automatic)', builtin: true, stages: [] }

export default function GeneratePanel({ project, item, variant, onGenerated, onShow }: { project: string; item: string; variant: string | null; onGenerated: (variant: string) => void; onShow: (variant: string) => void }) {
  const client = useQueryClient()
  const presets = useLoad({ queryKey: ['pipelines'], queryFn: api.pipelines })
  const det = useLoad({ queryKey: ['detail', project, item, variant], queryFn: () => (variant ? api.detail(project, item, variant) : Promise.resolve(null)) })
  const [preset, setPreset] = useState('best'), [opts, setOpts] = useState<Options>({}), [custom, setCustom] = useState('4000'), [customMode, setCustomMode] = useState(false), [saveAs, setSaveAs] = useState('')
  useEffect(() => { setOpts({}); setCustomMode(false) }, [item])
  const body: GenerateBody = { variant, preset, options: Object.keys(opts).length ? opts : null }
  const job = useJob({ queryKey: ['generate', project, item], queryFn: () => api.generateStatus(project, item), isRunning: j => j.state === 'running', every: 1500, invalidates: [['item', project, item], ['generate-check', project, item]] })
  const check = useLoad({ queryKey: ['generate-check', project, item, JSON.stringify(body)], queryFn: () => api.checkGenerate(project, item, body), staleTime: 0 })
  // A job seen running here and then done: show what it made. A job already done when the page opened is not news.
  const sawRunning = useRef(false)
  useEffect(() => {
    if (job.status !== 'ready') return
    if (job.data.state === 'running') sawRunning.current = true
    else if (sawRunning.current) { sawRunning.current = false; if (job.data.state === 'done' && job.data.result) onGenerated('gen-' + job.data.result.key) }
  }, [job, onGenerated])
  const go = useAsyncAction(async () => {
    const j = await api.generate(project, item, body)
    if (j.state === 'exists' && j.variant) { onShow(j.variant); return }
    client.setQueryData(['generate', project, item], j); job.poll()
  })
  const set = (o: Options) => setOpts(p => ({ ...p, ...o }))
  const c = check.status === 'ready' ? check.data : null, st = c?.stages, d = det.status === 'ready' ? det.data : null
  const size = String(stageValue(st, 'resize', 'size') ?? 'auto'), customSize = customMode || (size !== 'auto' && size !== 'native')
  const quality = String(stageValue(st, 'encode', 'quality') ?? 'auto'), fmt = String(stageValue(st, 'encode', 'format') ?? 'jpeg')
  const list = presets.status === 'ready' ? presets.data : [BEST]
  const presetLabel = list.find(p => p.name === preset)?.label, userPreset = list.find(p => p.name === preset && !p.builtin)
  const save = useAsyncAction(async () => { if (st) await api.savePipeline(saveAs.trim(), saveAs.trim(), st); await client.invalidateQueries({ queryKey: ['pipelines'] }); setPreset(saveAs.trim()); setOpts({}); setSaveAs('') })
  const remove = useAsyncAction(async (name: string) => { await api.deletePipeline(name); await client.invalidateQueries({ queryKey: ['pipelines'] }); setPreset('best'); setOpts({}) })
  const j = job.status === 'ready' ? job.data : null
  const failure = go.error ?? save.error ?? remove.error ?? (check.status === 'error' ? check.message : undefined) ?? (det.status === 'error' ? det.message : undefined)

  return (
    <section aria-labelledby="generate" className="grid gap-3 border-t border-border pt-6">
      <h2 id="generate" className="type-heading">Generate the best picture</h2>
      <p className="type-body-sm text-muted-foreground">{detailText(d)}</p>
      <div className="flex flex-wrap items-end gap-3">
        <Field label="Pipeline"><NativeSelect value={preset} onChange={e => { setPreset(e.target.value); setOpts({}); setCustomMode(false) }}>{list.map(p => <option key={p.name} value={p.name}>{p.label}</option>)}</NativeSelect></Field>
        {c?.exists
          ? <><Button variant="default" disabled>Already generated</Button><Button onPress={() => c.variant && onShow(c.variant)}>Show it</Button></>
          : <AsyncButton variant="default" disabled={job.running || !c} phase={job.running ? 'pending' : go.phase} error={go.error} labels={{ idle: 'Generate', pending: 'Generating…', done: 'Generated', error: 'Try again' }} onPress={() => void go.run()} />}
      </div>
      <p className="type-caption text-muted-foreground" data-testid="will-make">{c ? <>Will make: {c.summary}</> : 'Checking…'}{Object.keys(opts).length > 0 && presetLabel ? <> (changed from “{presetLabel}”)</> : null}</p>
      {j && (j.state === 'running' || j.state === 'failed') && (
        <JobStatus label="Generate" running={job.running} words={{ running: 'Generating', finished: 'Generated', failed: 'Failed' }} failed={j.state === 'failed'}
          summary={j.state === 'running' ? `${j.text ?? 'Working'}… (a 48 MP raw takes a minute or two)` : (j.error ?? 'Generating failed')} />
      )}
      {failure && <Alert tone="danger" announce>{failure}</Alert>}

      <Disclosure summary="Advanced">
        <div className="grid gap-4 pt-3">
          <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
            <Field label="Size">
              <NativeSelect value={customSize ? 'custom' : size} onChange={e => { setCustomMode(e.target.value === 'custom'); set({ size: e.target.value === 'custom' ? custom : e.target.value }) }}>
                <option value="auto">{d?.recommended && d.native ? `Recommended: ${d.recommended} px (${mp(d.recommended, Math.round(d.recommended * d.native[1] / d.native[0]))})` : 'Recommended'}</option>
                <option value="native">{d?.native ? `Native: ${Math.max(...d.native)} px (${mp(d.native[0], d.native[1])})` : 'Native'}</option>
                <option value="custom">Custom…</option>
              </NativeSelect>
            </Field>
            {customSize && <Field label="Long side in pixels"><Input inputMode="numeric" className="w-32" value={custom} onChange={e => { setCustom(e.target.value); set({ size: e.target.value }) }} /></Field>}
            <Field label="Format"><NativeSelect value={fmt} onChange={e => set({ format: e.target.value as 'jpeg' | 'tiff' })}><option value="jpeg">JPEG</option><option value="tiff">TIFF, 16 bit</option></NativeSelect></Field>
            {fmt === 'jpeg' && (
              <Field label="JPEG quality">
                <NativeSelect value={quality} onChange={e => set({ quality: e.target.value === 'auto' ? 'auto' : Number(e.target.value) })}>
                  <option value="auto">Automatic (the lowest that is invisible)</option><option value="100">100</option><option value="98">98</option><option value="95">95</option><option value="90">90</option>
                </NativeSelect>
              </Field>
            )}
            <Field label="Look">
              <NativeSelect value={String(stageValue(st, 'look', 'variant') ?? 'none')} onChange={e => set({ look: e.target.value })}>
                <option value="auto7">Your look (auto7: auto5 with the face step)</option><option value="auto5">auto5 (the arena's best so far)</option><option value="auto6">auto6</option><option value="auto4">auto4</option><option value="auto3">auto3</option><option value="auto2">auto2</option><option value="auto1">auto1</option><option value="none">None (our neutral develop)</option>
              </NativeSelect>
            </Field>
            <Field label="Exposure">
              <NativeSelect value={String(stageValue(st, 'develop', 'exposure') ?? 'auto')} onChange={e => { const v = e.target.value; set({ exposure: v === 'auto' || v === 'camera' || v === 'match' ? v : Number(v) }) }}>
                <option value="auto">Auto (an even lightness)</option><option value="match">Match the camera's JPEG (each picture)</option><option value="camera">Camera's own (fixed curve)</option><option value="-1">Camera −1 EV</option><option value="1">Camera +1 EV</option>
              </NativeSelect>
            </Field>
            <Field label="Highlight contrast">
              <NativeSelect value={String(stageValue(st, 'develop', 'highlight_contrast'))} onChange={e => set({ highlight_contrast: Number(e.target.value) })}><option value="0">Off</option><option value="0.8">Standard</option><option value="1.5">Strong</option></NativeSelect>
            </Field>
            {hasStage(st, 'sharpen') && (
              <Field label="Sharpness">
                <NativeSelect value={String(stageValue(st, 'sharpen', 'amount'))} onChange={e => set({ sharpen_amount: Number(e.target.value) })}>
                  <option value="1">Restore (undo the measured blur, no more)</option><option value="2">Crisp</option><option value="3">Best (automatic: as crisp as the camera's JPEG)</option><option value="3.5">Extra</option>
                </NativeSelect>
              </Field>
            )}
          </div>
          <div className="flex flex-wrap gap-x-6 gap-y-2">
            <Checkbox label="Level" checked={hasStage(st, 'level')} onCheckedChange={v => set({ level: v })} />
            <Checkbox label="Sharpen to the measured blur" checked={hasStage(st, 'sharpen')} onCheckedChange={v => set({ sharpen: v })} />
          </div>
          <div className="flex flex-wrap items-end gap-3">
            <Button size="sm" disabled={Object.keys(opts).length === 0} onPress={() => { setOpts({}); setCustomMode(false) }}>Back to the preset</Button>
            <form className="contents" onSubmit={e => { e.preventDefault(); if (saveAs.trim() && st) void save.run() }}>
              <Field label="Save as preset" action={<AsyncButton size="sm" type="submit" disabled={!saveAs.trim() || !st} phase={save.phase} error={save.error} labels={{ idle: 'Save preset', pending: 'Saving…', done: 'Saved', error: 'Try again' }} />}>
                <Input size="sm" value={saveAs} onChange={e => setSaveAs(e.target.value)} />
              </Field>
            </form>
            {userPreset && <AsyncButton size="sm" variant="destructive" phase={remove.phase} error={remove.error} labels={{ idle: `Delete “${userPreset.label}”`, pending: 'Deleting…', done: 'Deleted', error: 'Try again' }} onPress={() => void remove.run(userPreset.name)} />}
          </div>
        </div>
      </Disclosure>
    </section>
  )
}
