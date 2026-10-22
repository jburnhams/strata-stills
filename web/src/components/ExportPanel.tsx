import { useEffect, useRef, useState } from 'react'
import { api, exportFileUrl, type Detail, type ExportJob, type ExportRecord } from '../api'
import { useLoad } from '../route'
import { Err, btn } from './ui'

export const mp = (w: number, h: number) => (w * h / 1e6).toFixed(1) + ' MP'
export const mb = (b: number) => (b / 1048576).toFixed(1) + ' MB'

/** What the measurement says, in words: how many pixels of the picture are real detail. */
export function detailText(d: Detail | null): string {
  if (!d) return 'Measuring the real detail of this picture…'
  if (!d.ok || !d.native) return d.reason ? `Could not measure the detail (${d.reason}); native size is offered.` : 'Could not measure the detail.'
  const long = Math.max(...d.native), px = (s?: number) => Math.round((s ?? 0) * long)
  return `Native ${d.native[0]}×${d.native[1]}. Contrast is half gone at about ${px(d.scale50)} px across and 30 % at ${px(d.scale30)} px: beyond that the pixels only enlarge the blur.`
}

export default function ExportPanel({ project, item, stem, variant }: { project: string; item: string; stem: string; variant: string }) {
  const det = useLoad(() => api.detail(project, item, variant), [project, item, variant])
  const past = useLoad(() => api.exports(project, stem), [project, stem])
  const [size, setSize] = useState('auto'), [custom, setCustom] = useState('4000'), [format, setFormat] = useState<'jpeg' | 'tiff'>('jpeg'), [level, setLevel] = useState(true), [sharpen, setSharpen] = useState(true), [amount, setAmount] = useState('1.5')
  const [job, setJob] = useState<ExportJob | null>(null), [err, setErr] = useState<string | null>(null), timer = useRef<number | null>(null)
  const stop = () => { if (timer.current != null) { window.clearInterval(timer.current); timer.current = null } }
  useEffect(() => stop, [])
  useEffect(() => { setJob(null); stop() }, [item, variant])

  const poll = () => {
    stop()
    timer.current = window.setInterval(async () => {
      try { const j = await api.exportStatus(project, item); setJob(j); if (j.state !== 'running') { stop(); past.reload() } } catch (e) { setErr((e as Error).message); stop() }
    }, 1500)
  }
  const go = async () => {
    setErr(null)
    try { setJob(await api.startExport(project, item, { variant, size: size === 'custom' ? custom : size, format, level, sharpen, sharpen_amount: Number(amount) })); poll() } catch (e) { setErr((e as Error).message) }
  }
  const d = det.data, rec = d?.recommended, running = job?.state === 'running'
  return (
    <section aria-label="Export" className="mt-8 border-t border-stone-300 dark:border-stone-700 pt-4">
      <h2 className="text-base font-semibold mb-1">Export the final picture</h2>
      <p className="text-sm text-stone-500 mb-3">{detailText(d ?? null)}</p>
      <Err msg={det.error} />
      <div className="flex flex-wrap gap-x-6 gap-y-2 items-center text-sm">
        <label className="flex items-center gap-2">Size
          <select aria-label="Size" className="rounded-md border border-stone-300 dark:border-stone-700 bg-transparent px-2 py-1" value={size} onChange={e => setSize(e.target.value)}>
            <option value="auto">{rec && d?.native ? `Recommended: ${rec} px (${mp(rec, Math.round(rec * d.native[1] / d.native[0]))})` : 'Recommended'}</option>
            <option value="native">{d?.native ? `Native: ${Math.max(...d.native)} px (${mp(d.native[0], d.native[1])})` : 'Native'}</option>
            <option value="custom">Custom…</option>
          </select>
          {size === 'custom' && <input aria-label="Long side in pixels" className="w-24 rounded-md border border-stone-300 dark:border-stone-700 bg-transparent px-2 py-1" value={custom} onChange={e => setCustom(e.target.value)} />}
        </label>
        <label className="flex items-center gap-2">Format
          <select aria-label="Format" className="rounded-md border border-stone-300 dark:border-stone-700 bg-transparent px-2 py-1" value={format} onChange={e => setFormat(e.target.value as 'jpeg' | 'tiff')}>
            <option value="jpeg">JPEG, quality 95</option><option value="tiff">TIFF, 16 bit</option>
          </select>
        </label>
        <label className="flex items-center gap-2"><input type="checkbox" checked={level} onChange={e => setLevel(e.target.checked)} />Level</label>
        <label className="flex items-center gap-2"><input type="checkbox" checked={sharpen} onChange={e => setSharpen(e.target.checked)} />Sharpen to the measured blur</label>
        {sharpen && (
          <label className="flex items-center gap-2">Sharpness
            <select aria-label="Sharpness" className="rounded-md border border-stone-300 dark:border-stone-700 bg-transparent px-2 py-1" value={amount} onChange={e => setAmount(e.target.value)}>
              <option value="1">Restore (undo the measured blur, no more)</option><option value="1.5">Crisp (like the camera's JPEG)</option><option value="2">Extra</option>
            </select>
          </label>
        )}
        <button className={btn} disabled={running} onClick={go}>{running ? 'Exporting…' : 'Export'}</button>
      </div>
      {running && <p role="status" className="text-sm text-stone-500 mt-2">{job?.text}… (a 48 MP raw takes about a minute)</p>}
      {job?.state === 'failed' && <Err msg={job.error ?? 'Export failed'} />}
      <Err msg={err} />
      {job?.state === 'done' && job.result && <p role="status" className="text-sm text-green-700 dark:text-green-400 mt-2">Done: {job.result.name ?? job.result.file.split(/[\\/]/).pop()}</p>}
      <ExportList project={project} items={past.data} />
    </section>
  )
}

function ExportList({ project, items }: { project: string; items: ExportRecord[] | null }) {
  if (!items || items.length === 0) return null
  return (
    <div className="mt-4">
      <h3 className="text-sm font-medium mb-1">Exports of this picture</h3>
      <ul className="text-sm space-y-2">
        {items.map(r => (
          <li key={r.file} className="rounded-lg border border-stone-300 dark:border-stone-700 p-2">
            <a className="font-mono text-blue-600 hover:underline" href={exportFileUrl(project, r.name)}>{r.name}</a>
            <span className="text-stone-500"> — {r.output[0]}×{r.output[1]} ({mp(r.output[0], r.output[1])}), {mb(r.bytes)}, {r.size.why}</span>
            <div className="text-xs text-stone-500">{r.steps.join(' · ')}{r.level?.applied ? '' : r.level ? ` · not levelled (${r.level.reason})` : ''}</div>
          </li>
        ))}
      </ul>
    </div>
  )
}
