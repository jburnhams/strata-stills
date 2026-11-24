// Jobs: "What is the server making, and did it work?" The server's queue (strata-python's /api/jobs: pictures being generated, rules being applied to a project)
// as strata-react's JobsTable, newest first: each job's state as a word with its progress line or its error, and Cancel or Run again where they apply.
// The list is read again every two seconds while a job is queued or running (useJob), and once when the last one ends.
import { DataView, EmptyState, JobsTable, PageHeader, Skeleton, useJob } from 'strata-react'
import { useQueryClient } from '@tanstack/react-query'
import { active, api, type JobsPage as Page, type Task } from '../api'

/** A job in the person's words: what it makes and of what. */
export function describeJob(j: Pick<Task, 'kind' | 'domain'>): string {
  const d = j.domain ?? ''
  if (j.kind === 'generate') {
    const slash = d.lastIndexOf('/'), item = d.slice(slash + 1), project = d.slice(0, slash)
    return `Generate ${item.split('~').pop() || item} in ${project || 'a project'}`
  }
  if (j.kind === 'apply') return `Auto-generate ${d || 'a project'}`
  return j.kind
}

export default function JobsPage() {
  const client = useQueryClient()
  const jobs = useJob<Page>({ queryKey: ['jobs'], queryFn: api.jobs, isRunning: p => p.items.some(active), every: 2000 })
  const changed = () => client.invalidateQueries({ queryKey: ['jobs'] })
  return (
    <section className="grid gap-4">
      <PageHeader title="Jobs" scope="What the server is making: pictures generated one at a time and projects auto-generated. They carry on when you leave the page." />
      <DataView
        state={jobs}
        isEmpty={p => p.items.length === 0}
        skeleton={<div className="grid gap-2">{[0, 1, 2].map(i => <Skeleton key={i} className="h-8" />)}</div>}
        empty={<EmptyState title="No jobs yet">Generate a picture on an item's page, or auto-generate a folder of a project; each run is listed here.</EmptyState>}
      >
        {p => (
          <JobsTable
            caption="Jobs"
            jobs={p.items}
            describe={describeJob}
            onCancel={j => api.cancelJob(j.id).then(changed)}
            onRetry={j => api.retryJob(j.id).then(changed)}
          />
        )}
      </DataView>
    </section>
  )
}
