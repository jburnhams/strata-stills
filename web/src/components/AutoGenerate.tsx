import { useEffect, useRef, useState } from 'react'
import { api, type ApplyJob, type Rule } from '../api'
import { useLoad } from '../route'
import { Err, btn, btnQuiet } from './ui'

/**
 * Auto-generate for a source folder: choose the pipeline preset(s) that should be applied to everything in it (a rule saved in the project), and generate what is missing in one go.
 * Pictures that already have a variant made with the same settings are skipped, so this is safe to press again after adding photos to the folder.
 */
export default function AutoGenerate({ project, source, onDone }: { project: string; source: string; onDone: () => void }) {
  const presets = useLoad(() => api.pipelines(), [])
  const rules = useLoad(() => api.rules(project), [project])
  const [job, setJob] = useState<ApplyJob | null>(null), [err, setErr] = useState<string | null>(null), timer = useRef<number | null>(null)
  const stop = () => { if (timer.current != null) { window.clearInterval(timer.current); timer.current = null } }
  useEffect(() => stop, [])

  const mine = (rules.data?.rules ?? []).filter(r => r.source === source || r.source === '*'), chosen = new Set(mine.map(r => r.pipeline))
  const save = async (names: string[]) => {
    setErr(null)
    try {
      const others = (rules.data?.rules ?? []).filter(r => r.source !== source && r.source !== '*')
      const next: Rule[] = [...others, ...names.map(n => ({ source, pipeline: n }))]; await api.setRules(project, next); rules.reload()
    } catch (e) { setErr((e as Error).message) }
  }
  const toggle = (name: string) => save([...(chosen.has(name) ? [...chosen].filter(n => n !== name) : [...chosen, name])])
  const poll = () => {
    stop()
    timer.current = window.setInterval(async () => {
      try { const j = await api.applyStatus(project); setJob(j); if (j.state !== 'running') { stop(); rules.reload(); onDone() } } catch (e) { setErr((e as Error).message); stop() }
    }, 1500)
  }
  const go = async () => { setErr(null); try { setJob(await api.applyRules(project)); poll() } catch (e) { setErr((e as Error).message) } }
  const pending = rules.data?.pending ?? 0, running = job?.state === 'running'

  return (
    <section aria-label="Auto-generate" className="rounded-lg border border-stone-300 dark:border-stone-700 p-3 mb-4 text-sm">
      <h2 className="font-semibold mb-1">Auto-generate</h2>
      <p className="text-stone-500 text-xs mb-2">Pick the pipelines whose pictures you want for everything in this folder; Generate makes the ones that are missing, as new variants.</p>
      <div className="flex flex-wrap gap-x-5 gap-y-1">
        {presets.data?.map(p => <label key={p.name} className="flex items-center gap-2"><input type="checkbox" checked={chosen.has(p.name)} onChange={() => toggle(p.name)} />{p.label}</label>)}
      </div>
      <div className="mt-3 flex items-center gap-3 flex-wrap">
        <button className={btn} disabled={running || pending === 0} onClick={go}>{running ? 'Generating…' : pending === 0 ? (chosen.size ? 'Everything is generated' : 'Nothing chosen') : `Generate ${pending} missing`}</button>
        {running && <button className={btnQuiet} onClick={() => api.stopApply(project)}>Stop after this picture</button>}
        {job && job.state !== 'idle' && <span role="status" className="text-xs text-stone-500">{job.state === 'running' ? `${job.done ?? 0} of ${job.total ?? '…'}: ${job.text ?? ''}` : `${job.state}: ${job.done ?? 0} of ${job.total ?? 0}${job.failed?.length ? `, ${job.failed.length} failed` : ''}`}</span>}
      </div>
      {job?.failed?.map(([iid, msg]) => <p key={iid} className="text-xs text-red-600 mt-1">{iid.split('~').pop()}: {msg}</p>)}
      <Err msg={err || rules.error} />
    </section>
  )
}
