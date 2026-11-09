"""gpu.lease: the strata-models GPU lease around batch work, and every way it steps aside (no CUDA, no client, no service, switched off)."""

import sys
import threading
import types

import pytest

from stratastills import gpu


class FakeLease:
    def __init__(self, log):
        self.log = log

    def release(self):
        self.log.append("release")


def fake_client(log, running=True):
    m = types.ModuleType("strata_models_client")

    class Unavailable(RuntimeError):
        pass

    m.Unavailable = Unavailable
    m.available = lambda: running

    def acquire(app, vram_gb=None, note=None):
        log.append(("acquire", app, vram_gb, note))
        return FakeLease(log)

    m.acquire = acquire
    return m


@pytest.fixture
def cuda(monkeypatch):
    monkeypatch.setattr(gpu, "_give_back_memory", gpu._give_back_memory)
    monkeypatch.delenv("STRATASTILLS_LEASE", raising=False)
    monkeypatch.setattr(gpu, "available", lambda: True)
    monkeypatch.setattr(gpu, "dev", lambda: types.SimpleNamespace(type="cuda"))
    gpu._lease_held[:] = [0, None]
    gpu._lease_cap = None
    monkeypatch.delenv("STRATASTILLS_LEASE_GB", raising=False)


def test_a_lease_is_taken_once_for_nested_blocks_and_worker_threads(cuda, monkeypatch):
    log = []
    monkeypatch.setitem(sys.modules, "strata_models_client", fake_client(log))
    with gpu.lease("arena build") as l:
        assert isinstance(l, FakeLease)
        with gpu.lease("full-size view") as inner:
            assert inner is l
        seen = []

        def worker():
            with gpu.lease("worker") as w:
                seen.append(w)

        t = threading.Thread(target=worker)
        t.start()
        t.join()  # a batch worker thread shares the same lease
        assert seen == [l]
    assert (
        log == [("acquire", "strata-stills", 8.0, "arena build"), "release"]
        and gpu._lease_held == [0, None]
        and gpu._lease_cap is None
    )


def test_the_card_memory_is_returned_before_the_lease_goes_back(cuda, monkeypatch):
    from stratastills import denoise_ai

    log = []
    monkeypatch.setitem(sys.modules, "strata_models_client", fake_client(log))
    monkeypatch.setattr(gpu, "free", lambda: log.append("empty_cache"))
    monkeypatch.setitem(denoise_ai._models, "scunet", object())
    with gpu.lease("batch"):
        pass
    assert log[1:] == ["empty_cache", "release"] and "scunet" not in denoise_ai._models


def test_the_lease_is_given_back_when_the_block_fails(cuda, monkeypatch):
    log = []
    monkeypatch.setitem(sys.modules, "strata_models_client", fake_client(log))
    with pytest.raises(ValueError), gpu.lease("batch", vram_gb=4):
        raise ValueError("develop failed")
    assert log == [("acquire", "strata-stills", 4, "batch"), "release"]


def test_without_cuda_client_service_or_with_it_switched_off_the_block_just_runs(cuda, monkeypatch):
    log = []
    monkeypatch.setitem(sys.modules, "strata_models_client", fake_client(log, running=False))
    with gpu.lease("x") as l:
        assert l is None
    monkeypatch.setitem(sys.modules, "strata_models_client", None)  # not installed: import fails
    with gpu.lease("x") as l:
        assert l is None
    monkeypatch.setitem(sys.modules, "strata_models_client", fake_client(log))
    monkeypatch.setenv("STRATASTILLS_LEASE", "0")
    with gpu.lease("x") as l:
        assert l is None
    monkeypatch.delenv("STRATASTILLS_LEASE")
    monkeypatch.setattr(gpu, "dev", lambda: types.SimpleNamespace(type="mps"))
    with gpu.lease("x") as l:
        assert l is None
    monkeypatch.setattr(gpu, "available", lambda: False)
    with gpu.lease("x") as l:
        assert l is None
    assert log == []


def test_a_sized_lease_caps_what_this_process_plans_with(cuda, monkeypatch):
    log = []
    monkeypatch.setitem(sys.modules, "strata_models_client", fake_client(log))
    with gpu.lease("batch"):
        assert gpu._lease_cap == 8e9
    monkeypatch.setenv("STRATASTILLS_LEASE_GB", "5.5")
    with gpu.lease("batch"):
        assert gpu._lease_cap == 5.5e9
    monkeypatch.setenv("STRATASTILLS_LEASE_GB", "whole")
    with gpu.lease("batch"):
        assert gpu._lease_cap is None
    monkeypatch.setenv("STRATASTILLS_LEASE_GB", "lots")
    assert gpu.lease_gb() == 8.0
    assert [e[2] for e in log if isinstance(e, tuple)] == [8.0, 5.5, None]
    t = types.SimpleNamespace(
        cuda=types.SimpleNamespace(mem_get_info=lambda: (10e9, 12e9), memory_reserved=lambda: 1e9)
    )
    monkeypatch.setattr(gpu, "torch_", lambda: t)
    assert gpu._cuda_total() == 11e9
    gpu._lease_cap = 4e9
    assert gpu._cuda_total() == 4e9
    gpu._lease_cap = None


def test_slots_made_before_a_lease_are_not_shared_by_work_under_it(cuda, monkeypatch):
    monkeypatch.setattr(gpu, "_slots", None)
    monkeypatch.delenv("STRATASTILLS_GPU_SLOTS", raising=False)
    t = types.SimpleNamespace(cuda=types.SimpleNamespace(mem_get_info=lambda: (24e9, 24e9), memory_reserved=lambda: 0))
    monkeypatch.setattr(gpu, "torch_", lambda: t)
    monkeypatch.setattr(gpu, "_tiled_batch", False)
    assert gpu.slots() == 3  # 24 GB: three whole pictures
    log = []
    monkeypatch.setitem(sys.modules, "strata_models_client", fake_client(log))
    with gpu.lease("full-size view"):
        assert gpu.slots() == 1  # under the 8 GB lease: one
    assert gpu.slots() == 3
