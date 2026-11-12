// Auto-generate for a source folder: choose the pipeline preset(s) that should be applied to everything in it (a rule saved in the project), and generate what is missing in one go.
// Pictures that already have a variant made with the same settings are skipped, so this is safe to press again after adding photos to the folder.
// The run is a server job: strata-react's useJob polls its status only while it runs and reads the folder's contents again once when it ends; JobStatus says where it is,
// with the words, the bar and each picture that failed (never a hand-written interval).
import { useQueryClient } from '@tanstack/react-query'
import { Alert, AsyncButton, Checkbox, JobStatus, useAsyncAction, useJob, useLoad } from 'strata-react'
import { api, type ApplyJob, type Rule } from '../api'

/** The job's sentence: where it is while it runs, how it went once it ends. */
export function jobSummary(j: ApplyJob): string {
  if (j.state === 'running') return `${j.done ?? 0} of ${j.total ?? '…'}${j.text ? `: ${j.text}` : ''}`
  if (j.state === 'stopped') return `Stopped after ${j.done ?? 0} of ${j.total ?? 0}.`
  return `Made ${j.done ?? 0} of ${j.total ?? 0}${j.failed?.length ? `; ${j.failed.length} failed` : ''}.`
}

export default function AutoGenerate({ project, source }: { project: string; source: string }) {
  const client = useQueryClient()
  const presets = useLoad({ queryKey: ['pipelines'], queryFn: api.pipelines })
  const rules = useLoad({ queryKey: ['rules', project], queryFn: () => api.rules(project) })
  const job = useJob({ queryKey: ['apply', project], queryFn: () => api.applyStatus(project), isRunning: j => j.state === 'running', every: 1500, invalidates: [['items', project, source], ['rules', project]] })
  const all = rules.status === 'ready' ? rules.data.rules : []
  const chosen = new Set(all.filter(r => r.source === source || r.source === '*').map(r => r.pipeline))
  const toggle = useAsyncAction(async (name: string) => {
    const names = chosen.has(name) ? [...chosen].filter(n => n !== name) : [...chosen, name]
    const next: Rule[] = [...all.filter(r => r.source !== source && r.source !== '*'), ...names.map(n => ({ source, pipeline: n }))]
    client.setQueryData(['rules', project], await api.setRules(project, next))
  })
  const start = useAsyncAction(async () => { client.setQueryData(['apply', project], await api.applyRules(project)); job.poll() })
  const pending = rules.status === 'ready' ? rules.data.pending : 0
  const j = job.status === 'ready' ? job.data : null

  return (
    <section aria-labelledby="auto-generate" className="grid gap-3 rounded-surface border border-border p-4">
      <h2 id="auto-generate" className="type-heading">Auto-generate</h2>
      <p className="type-body-sm text-muted-foreground">Pick the pipelines whose pictures you want for everything in this folder; Generate makes the ones that are missing, as new variants.</p>
      <div role="group" aria-label="Pipelines for this folder" className="flex flex-wrap gap-x-6 gap-y-2">
        {presets.status === 'ready' && presets.data.map(p => <Checkbox key={p.name} label={p.label} checked={chosen.has(p.name)} disabled={toggle.phase === 'pending'} onCheckedChange={() => void toggle.run(p.name)} />)}
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <AsyncButton variant="default" disabled={job.running || pending === 0} phase={start.phase} error={start.error}
          labels={{ idle: pending === 0 ? (chosen.size ? 'Everything is generated' : 'Nothing chosen') : `Generate ${pending} missing`, pending: 'Starting…', done: 'Started', error: 'Try again' }} onPress={() => void start.run()} />
      </div>
      {j && j.state !== 'idle' && (
        <JobStatus label="Auto-generate" running={job.running} words={{ running: 'Generating', finished: 'Generated', failed: 'Generated with failures', cancelled: 'Stopped' }}
          summary={jobSummary(j)} current={j.done} total={j.total} cancelled={j.state === 'stopped'} failed={j.state === 'failed' && !j.failed?.length}
          failures={j.failed?.map(([id, reason]) => ({ key: id, label: id.split('~').pop() ?? id, reason }))}
          onCancel={() => void api.stopApply(project).then(() => job.poll())} cancelLabel="Stop after this picture" />
      )}
      {(toggle.error || (rules.status === 'error' && rules.message)) && <Alert tone="danger" announce>{toggle.error ?? (rules.status === 'error' ? rules.message : '')}</Alert>}
    </section>
  )
}
