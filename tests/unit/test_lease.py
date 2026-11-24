"""gpu.lease: the strata-models GPU lease around batch work (against strata-python's `fake_models` stand-in service), and every way it steps aside (no CUDA, no service, switched off)."""

import threading
import types

import pytest

from stratastills import gpu


def asked(fake):
    """(app, vram_gb, note) of each lease the stand-in service was asked for."""
    return [(x["app"], x["vram_gb"], x["note"]) for x in fake.leases]


@pytest.fixture
def cuda(monkeypatch):
    monkeypatch.setattr(gpu, "_give_back_memory", gpu._give_back_memory)
    monkeypatch.delenv("STRATASTILLS_LEASE", raising=False)
    monkeypatch.setattr(gpu, "available", lambda: True)
    monkeypatch.setattr(gpu, "dev", lambda: types.SimpleNamespace(type="cuda"))
    gpu._lease_held[:] = [0, None]
    gpu._lease_cap = None
    monkeypatch.delenv("STRATASTILLS_LEASE_GB", raising=False)


def test_a_lease_is_taken_once_for_nested_blocks_and_worker_threads(cuda, fake_models):
    with gpu.lease("arena build") as l:
        assert l is not None and l.id == 1
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
        asked(fake_models) == [("strata-stills", 8.0, "arena build")]
        and fake_models.released == [1]
        and gpu._lease_held == [0, None]
        and gpu._lease_cap is None
    )


def test_the_card_memory_is_returned_before_the_lease_goes_back(cuda, fake_models, monkeypatch):
    from stratastills import denoise_ai

    log = []
    monkeypatch.setattr(gpu, "free", lambda: log.append(("empty_cache", list(fake_models.released))))
    monkeypatch.setitem(denoise_ai._models, "scunet", object())
    with gpu.lease("batch"):
        pass
    assert log == [("empty_cache", [])] and fake_models.released == [1] and "scunet" not in denoise_ai._models


def test_the_lease_is_given_back_when_the_block_fails(cuda, fake_models):
    with pytest.raises(ValueError), gpu.lease("batch", vram_gb=4):
        raise ValueError("develop failed")
    assert asked(fake_models) == [("strata-stills", 4, "batch")] and fake_models.released == [1]


def test_without_cuda_or_service_or_with_it_switched_off_the_block_just_runs(cuda, fake_models, monkeypatch):
    fake_models.down = True  # the service answers 503 (starting, or stopped behind its proxy)
    with gpu.lease("x") as l:
        assert l is None
    fake_models.down = False
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
    assert fake_models.leases == []


def test_a_sized_lease_caps_what_this_process_plans_with(cuda, fake_models, monkeypatch):
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
    assert [x[1] for x in asked(fake_models)] == [8.0, 5.5, None]
    t = types.SimpleNamespace(
        cuda=types.SimpleNamespace(mem_get_info=lambda: (10e9, 12e9), memory_reserved=lambda: 1e9)
    )
    monkeypatch.setattr(gpu, "torch_", lambda: t)
    assert gpu._cuda_total() == 11e9
    gpu._lease_cap = 4e9
    assert gpu._cuda_total() == 4e9
    gpu._lease_cap = None


def test_slots_made_before_a_lease_are_not_shared_by_work_under_it(cuda, fake_models, monkeypatch):
    monkeypatch.setattr(gpu, "_slots", None)
    monkeypatch.delenv("STRATASTILLS_GPU_SLOTS", raising=False)
    t = types.SimpleNamespace(cuda=types.SimpleNamespace(mem_get_info=lambda: (24e9, 24e9), memory_reserved=lambda: 0))
    monkeypatch.setattr(gpu, "torch_", lambda: t)
    monkeypatch.setattr(gpu, "_tiled_batch", False)
    assert gpu.slots() == 3  # 24 GB: three whole pictures
    with gpu.lease("full-size view"):
        assert gpu.slots() == 1  # under the 8 GB lease: one
    assert gpu.slots() == 3
