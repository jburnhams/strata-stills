"""stratastills.jobs: generate and apply as jobs of strata-python's durable queue, run here one at a time with `Queue.step` (the pipeline itself is stubbed)."""

import threading

import pytest

from stratastills import jobs
from stratastills import pipeline as PL


@pytest.fixture
def q():
    return jobs.queue()


def test_a_generate_job_runs_once_for_the_same_picture_and_keeps_its_result(q, monkeypatch):
    calls = []
    monkeypatch.setattr(PL, "generate", lambda *a: calls.append(a) or dict(key="k1", summary="JPEG q90"))
    a = jobs.enqueue_generate("p", "i~1", options=dict(quality=90))
    assert jobs.enqueue_generate("p", "i~1", options=dict(quality=90)) == a  # queued already: the same job
    assert jobs.enqueue_generate("p", "i~1", options=dict(quality=80)) != a  # other settings: another picture
    q.step(jobs.HANDLERS)
    t = jobs.latest(jobs.GENERATE, jobs.item_domain("p", "i~1"))
    assert calls[0] == ("p", "i~1", None, None, dict(quality=90), None)
    assert q.get(a).state == "done" and q.get(a).result == dict(key="k1", existing=False, summary="JPEG q90")
    assert t is not None and t.state == "queued"  # the newest is the second one, not run yet


def test_an_apply_job_reports_pictures_done_from_its_threads_and_stops_when_cancelled(q, monkeypatch):
    running = []

    def apply_rules(pid, progress, stop, workers):
        t = threading.Thread(target=progress, args=(1, 3, "a done"))  # the batch reports from its own threads
        t.start()
        t.join()
        assert not stop()
        q.cancel(job)  # the person presses Stop while it runs; the worker's heartbeat brings the request to the job
        q.heartbeat(running[0])
        assert stop()
        return dict(done=1, failed=[("i~2", "bad raw")], total=3, workers=2)

    monkeypatch.setattr(PL, "apply_rules", apply_rules)
    job = jobs.enqueue_apply("p", workers=2)
    assert jobs.enqueue_apply("p") == job
    q.step({jobs.APPLY: lambda j: running.append(j) or jobs.run_apply(j)})
    done = q.get(job)
    assert done.state == "cancelled" and done.result["failed"] == [["i~2", "bad raw"]]
    assert done.progress is not None and (done.progress.done, done.progress.total) == (2, 3)


def test_a_failed_job_fails_for_good_after_one_more_try(q, monkeypatch):
    monkeypatch.setattr(PL, "generate", lambda *a: (_ for _ in ()).throw(RuntimeError("no pixels")))
    monkeypatch.setattr(q, "base_s", 0.0)
    jid = jobs.enqueue_generate("p", "i~3")
    q.step(jobs.HANDLERS)
    assert q.get(jid).state == "queued"  # retried once
    q.step(jobs.HANDLERS)
    assert q.get(jid).state == "failed" and "no pixels" in (q.get(jid).error or "")
