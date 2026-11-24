"""The server's long work as jobs of strata-python's durable queue (`strata_python.tasks.queue`): generating one picture of an item (`generate`) and applying a
project's rules to its sources (`apply`). They used to be threads the server kept in two dicts, lost on a restart; now each is a row in a SQLite file beside the
settings (`<STRATASTILLS_HOME>/jobs.sqlite`), so the web app reads them all at /api/jobs (strata-python's routes, strata-react's JobsTable), a cancel is a
request the job sees (`job.stopping`), and a job whose server died (the CUDA restart, `gpu.exit_if_cuda_broken`) runs again once the server is back.

A file of this machine's own, not the shared database: the jobs are this server's work on this machine's GPU and folders, and the app runs without a database
(`settings.database_url` unset). The table is created with the file (no migration to run on deploy).

Jobs run on threads of the server process (`isolate=False`): the pixel work keeps its GPU lease, its loaded nets and its caches across jobs, as the threads did;
cancelling is cooperative (an apply stops before its next picture; a picture being made finishes). One job per kind and subject at a time: a second start of the
same generate (same item and settings: `dedupe`) or of an apply of the same project is the first job, and an apply holds its project (`resources`)."""

import contextvars
import os
import threading

from sqlalchemy import MetaData, create_engine
from strata_python.tasks.progress import report_progress
from strata_python.tasks.queue import Job, Queue, Worker, jobs_table

from stratastills import settings

GENERATE, APPLY = "generate", "apply"
RETRIES = 1  # a failure runs once more (after a CUDA restart that is all it needs); a picture that cannot be made fails for good
_queues = {}
_lock = threading.Lock()


def path():
    return os.path.join(settings.home(), "jobs.sqlite")


def queue():
    """The queue of this STRATASTILLS_HOME (one per folder, so each test has its own)."""
    p = path()
    with _lock:
        q = _queues.get(p)
        if q is None:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            engine = create_engine(f"sqlite:///{p}")
            table = jobs_table(MetaData())
            table.metadata.create_all(engine)
            q = _queues[p] = Queue(engine, table, retries=RETRIES)
        return q


def item_domain(pid, iid):
    return f"{pid}/{iid}"


def enqueue_generate(pid, iid, variant=None, preset=None, options=None, stages=None):
    """The id of the job that makes this picture (an existing one when the same is already queued or running)."""
    return queue().enqueue(
        GENERATE,
        dict(project=pid, item=iid, variant=variant, preset=preset, options=options, stages=stages),
        domain=item_domain(pid, iid),
        dedupe=True,
    )


def enqueue_apply(pid, workers=None):
    """The id of the job that applies the project's rules: the queued or running one when there is one (whatever its `workers`), else a new one."""
    active = queue().page(kind=APPLY, domain=pid, state=["queued", "running"], limit=1).items
    if active:
        return int(active[0].id)
    return queue().enqueue(APPLY, dict(project=pid, workers=workers), domain=pid, resources=[f"apply:{pid}"])


def latest(kind, domain):
    """The newest job of `kind` about `domain`, as a Task, or None."""
    page = queue().page(kind=kind, domain=domain, limit=1)
    return page.items[0] if page.items else None


def _gpu_restart(e):
    from stratastills import gpu

    gpu.exit_if_cuda_broken(
        e
    )  # a sticky CUDA error: the process exits and is restarted; the queue runs the job again then


def run_generate(job: Job):
    """Make one picture; its stages report their progress (pipeline.generate runs them as a strata_python.workflow pipeline)."""
    from stratastills import pipeline as PL

    o = job.options
    try:
        r = PL.generate(o["project"], o["item"], o.get("variant"), o.get("preset"), o.get("options"), o.get("stages"))
    except Exception as e:
        _gpu_restart(e)
        raise
    return dict(key=r["key"], existing=r.get("existing", False), summary=r.get("summary"))


def run_apply(job: Job):
    """Make what the project's rules ask for; progress is pictures done of the total (reported from the batch's own threads, hence the copied context)."""
    from stratastills import pipeline as PL

    ctx = contextvars.copy_context()  # the job's progress reporter; ThreadPoolExecutor threads do not inherit it
    try:
        r = PL.apply_rules(
            job.options["project"],
            lambda i, n, t: ctx.run(report_progress, i, n, t),
            stop=lambda: job.stopping,
            workers=job.options.get("workers"),
        )
    except Exception as e:
        _gpu_restart(e)
        raise
    report_progress(r["done"] + len(r["failed"]), r["total"], "stopped" if job.stopping else "finished")
    return dict(r, failed=[list(f) for f in r["failed"]])


HANDLERS = {GENERATE: run_generate, APPLY: run_apply}


def worker(concurrency=2):
    """The server's worker (not started): two jobs at once, so a picture can be made while a project's rules are applied, as before."""
    return Worker(queue(), HANDLERS, concurrency=concurrency, isolate=False, poll_s=0.5)
