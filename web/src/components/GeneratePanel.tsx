import { useEffect, useRef, useState } from 'react'
import { api, hasStage, stageValue, type Detail, type GenerateCheck, type GenerateJob, type Options } from '../api'
import { useLoad } from '../route'
import { Err, btn, btnQuiet } from './ui'

export const mp = (w: number, h: number) => (w * h / 1e6).toFixed(1) + ' MP'
export const mb = (b: number) => (b / 1048576).toFixed(1) + ' MB'

/** What the measurement says, in words: how many pixels of the picture are real detail. */
export function detailText(d: Detail | null): string {
  if (!d) return 'Measuring the real detail of this picture…'
  if (!d.ok || !d.native) return d.reason ? `Could not measure the detail (${d.reason}); native size is offered.` : 'Could not measure the detail.'
  const long = Math.max(...d.native), px = (s?: number) => Math.round((s ?? 0) * long)
  return `Native ${d.native[0]}×${d.native[1]}. Contrast is half gone at about ${px(d.scale50)} px across and 30 % at ${px(d.scale30)} px: beyond that the pixels only enlarge the blur.`
}

const field = 'rounded-md border border-stone-300 dark:border-stone-700 bg-transparent px-2 py-1 text-sm'

/**
 * Generating a picture is running a pipeline preset on the item's source: one button, nothing to choose (the default preset is the automatic best).
 * Every setting is behind "Advanced", laid over the preset. The result is a new variant of the item (listed with the others, comparable in the loupe); the same settings twice find the first one, and the button is greyed out.
 */
export default function GeneratePanel({ project, item, variant, onGenerated, onShow }: { project: string; item: string; variant: string | null; onGenerated: (variant: string) => void; onShow: (variant: string) => void }) {
  const presets = useLoad(() => api.pipelines(), [])
  const det = useLoad(() => (variant ? api.detail(project, item, variant) : Promise.resolve(null)), [project, item, variant])
  const [preset, setPreset] = useState('best'), [opts, setOpts] = useState<Options>({}), [check, setCheck] = useState<GenerateCheck | null>(null)
  const [job, setJob] = useState<GenerateJob | null>(null), [err, setErr] = useState<string | null>(null), [custom, setCustom] = useState('4000'), [customMode, setCustomMode] = useState(false), [saveAs, setSaveAs] = useState('')
  const timer = useRef<number | null>(null)
  const stop = () => { if (timer.current != null) { window.clearInterval(timer.current); timer.current = null } }
  useEffect(() => stop, [])
  useEffect(() => { setJob(null); stop(); setOpts({}); setCustomMode(false) }, [item])

  const body = { variant, preset, options: Object.keys(opts).length ? opts : null }
  const bodyKey = JSON.stringify(body)
  useEffect(() => {
    let live = true; setErr(null)
    api.checkGenerate(project, item, body).then(c => live && setCheck(c), e => { if (live) { setCheck(null); setErr(e.message) } })
    return () => { live = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project, item, bodyKey, job?.state])

  const poll = () => {
    stop()
    timer.current = window.setInterval(async () => {
      try {
        const j = await api.generateStatus(project, item); setJob(j)
        if (j.state !== 'running') { stop(); if (j.state === 'done' && j.result) onGenerated('gen-' + j.result.key) }
      } catch (e) { setErr((e as Error).message); stop() }
    }, 1500)
  }
  const go = async () => {
    setErr(null)
    try { const j = await api.generate(project, item, body); if (j.state === 'exists' && j.variant) { onShow(j.variant); return } setJob(j); poll() } catch (e) { setErr((e as Error).message) }
  }
  const set = (o: Options) => setOpts(p => ({ ...p, ...o }))
  const st = check?.stages, running = job?.state === 'running', d = det.data
  const size = String(stageValue(st, 'resize', 'size') ?? 'auto'), customSize = customMode || (size !== 'auto' && size !== 'native')
  const quality = String(stageValue(st, 'encode', 'quality') ?? 'auto'), format = String(stageValue(st, 'encode', 'format') ?? 'jpeg')
  const presetLabel = presets.data?.find(p => p.name === preset)?.label
  const savePreset = async () => {
    setErr(null)
    try { if (st) await api.savePipeline(saveAs.trim(), saveAs.trim(), st); presets.reload(); setPreset(saveAs.trim()); setOpts({}); setSaveAs('') } catch (e) { setErr((e as Error).message) }
  }
  const userPreset = presets.data?.find(p => p.name === preset && !p.builtin)

  return (
    <section aria-label="Generate" className="mt-8 border-t border-stone-300 dark:border-stone-700 pt-4">
      <h2 className="text-base font-semibold mb-1">Generate the best picture</h2>
      <p className="text-sm text-stone-500 mb-2">{detailText(d ?? null)}</p>
      <Err msg={det.error} />
      <div className="flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-2 text-sm">Pipeline
          <select aria-label="Pipeline" className={field} value={preset} onChange={e => { setPreset(e.target.value); setOpts({}); setCustomMode(false) }}>
            {(presets.data ?? [{ name: 'best', label: 'Best quality (automatic)', builtin: true, stages: [] }]).map(p => <option key={p.name} value={p.name}>{p.label}</option>)}
          </select>
        </label>
        {check?.exists
          ? <><button className={btn} disabled>Already generated</button><button className={btnQuiet} onClick={() => check.variant && onShow(check.variant)}>Show it</button></>
          : <button className={btn} disabled={running || !check} onClick={go}>{running ? 'Generating…' : 'Generate'}</button>}
      </div>
      <p className="text-xs text-stone-500 mt-2" data-testid="will-make">{check ? <>Will make: {check.summary}</> : 'Checking…'}{Object.keys(opts).length > 0 && presetLabel ? <> (changed from “{presetLabel}”)</> : null}</p>
      {running && <p role="status" className="text-sm text-stone-500 mt-2">{job?.text}… (a 48 MP raw takes a minute or two)</p>}
      {job?.state === 'failed' && <Err msg={job.error ?? 'Generating failed'} />}
      <Err msg={err} />

      <details className="mt-3 text-sm">
        <summary className="cursor-pointer text-stone-600 dark:text-stone-300">Advanced</summary>
        <div className="mt-3 flex flex-wrap gap-x-6 gap-y-3 items-center">
          <label className="flex items-center gap-2">Size
            <select aria-label="Size" className={field} value={customSize ? 'custom' : size} onChange={e => { setCustomMode(e.target.value === 'custom'); set({ size: e.target.value === 'custom' ? custom : e.target.value }) }}>
              <option value="auto">{d?.recommended && d.native ? `Recommended: ${d.recommended} px (${mp(d.recommended, Math.round(d.recommended * d.native[1] / d.native[0]))})` : 'Recommended'}</option>
              <option value="native">{d?.native ? `Native: ${Math.max(...d.native)} px (${mp(d.native[0], d.native[1])})` : 'Native'}</option>
              <option value="custom">Custom…</option>
            </select>
            {customSize && <input aria-label="Long side in pixels" className={field + ' w-24'} value={custom} onChange={e => { setCustom(e.target.value); set({ size: e.target.value }) }} />}
          </label>
          <label className="flex items-center gap-2">Format
            <select aria-label="Format" className={field} value={format} onChange={e => set({ format: e.target.value as 'jpeg' | 'tiff' })}><option value="jpeg">JPEG</option><option value="tiff">TIFF, 16 bit</option></select>
          </label>
          {format === 'jpeg' && (
            <label className="flex items-center gap-2">JPEG quality
              <select aria-label="JPEG quality" className={field} value={quality} onChange={e => set({ quality: e.target.value === 'auto' ? 'auto' : Number(e.target.value) })}>
                <option value="auto">Automatic (the lowest that is invisible)</option><option value="100">100</option><option value="98">98</option><option value="95">95</option><option value="90">90</option>
              </select>
            </label>
          )}
          <label className="flex items-center gap-2"><input type="checkbox" checked={hasStage(st, 'level')} onChange={e => set({ level: e.target.checked })} />Level</label>
          <label className="flex items-center gap-2"><input type="checkbox" checked={hasStage(st, 'sharpen')} onChange={e => set({ sharpen: e.target.checked })} />Sharpen to the measured blur</label>
          {hasStage(st, 'sharpen') && (
            <label className="flex items-center gap-2">Sharpness
              <select aria-label="Sharpness" className={field} value={String(stageValue(st, 'sharpen', 'amount'))} onChange={e => set({ sharpen_amount: Number(e.target.value) })}>
                <option value="1">Restore (undo the measured blur, no more)</option><option value="1.5">Crisp (like the camera's JPEG)</option><option value="2">Extra</option>
              </select>
            </label>
          )}
          <label className="flex items-center gap-2">Exposure
            <select aria-label="Exposure" className={field} value={String(stageValue(st, 'develop', 'exposure') ?? 'auto')} onChange={e => { const v = e.target.value; set({ exposure: v === 'auto' || v === 'camera' || v === 'match' ? v : Number(v) }) }}>
              <option value="auto">Auto (an even lightness)</option><option value="match">Match the camera's JPEG (each picture)</option><option value="camera">Camera's own (fixed curve)</option><option value="-1">Camera −1 EV</option><option value="1">Camera +1 EV</option>
            </select>
          </label>
          <label className="flex items-center gap-2">Highlight contrast
            <select aria-label="Highlight contrast" className={field} value={String(stageValue(st, 'develop', 'highlight_contrast'))} onChange={e => set({ highlight_contrast: Number(e.target.value) })}>
              <option value="0">Off</option><option value="0.8">Standard</option><option value="1.5">Strong</option>
            </select>
          </label>
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <button className={btnQuiet} disabled={Object.keys(opts).length === 0} onClick={() => { setOpts({}); setCustomMode(false) }}>Back to the preset</button>
          <input aria-label="Preset name" className={field + ' w-40'} placeholder="Save as preset…" value={saveAs} onChange={e => setSaveAs(e.target.value)} />
          <button className={btnQuiet} disabled={!saveAs.trim() || !st} onClick={savePreset}>Save preset</button>
          {userPreset && <button className={btnQuiet} onClick={async () => { await api.deletePipeline(userPreset.name); presets.reload(); setPreset('best'); setOpts({}) }}>Delete “{userPreset.label}”</button>}
        </div>
      </details>
    </section>
  )
}
