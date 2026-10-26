"""resources.py: how much memory there is, and how many pictures and how big a tile that allows."""
import ctypes, os, sys, types

from stratastills import resources as R


def test_the_memory_of_this_machine_is_found_and_zero_when_it_cannot_be(monkeypatch):
    assert R.total_ram() > 0
    def boom(name): raise OSError('no sysconf')
    monkeypatch.setattr(os, 'sysconf', boom, raising=False); assert R.total_ram() == 0


def test_windows_asks_the_kernel_for_its_physical_memory(monkeypatch):
    def fake(m): m.ullTotalPhys = 12345
    monkeypatch.setattr(sys, 'platform', 'win32'); monkeypatch.setattr(ctypes, 'windll', types.SimpleNamespace(kernel32=types.SimpleNamespace(GlobalMemoryStatusEx=fake)), raising=False)
    monkeypatch.setattr(ctypes, 'byref', lambda m: m)
    assert R.total_ram() == 12345


def test_without_a_gpu_the_workers_follow_the_cores_and_the_memory_up_to_six(monkeypatch):
    monkeypatch.setattr(os, 'cpu_count', lambda: 64)
    monkeypatch.setattr(R, 'total_ram', lambda: 0); assert R.auto_workers(False) == 1                    # memory unknown: one at a time
    monkeypatch.setattr(R, 'total_ram', lambda: 10 ** 13); assert R.auto_workers(False) == 6             # plenty of both: the cap
    monkeypatch.setattr(R, 'total_ram', lambda: int(R.PER_PICTURE_BYTES * 2.5 / 0.8)); assert R.auto_workers(False) == 2     # memory for two
    monkeypatch.setattr(os, 'cpu_count', lambda: 2); monkeypatch.setattr(R, 'total_ram', lambda: 10 ** 13); assert R.auto_workers(False) == 1   # fewer than 4 cores
    monkeypatch.setattr(os, 'cpu_count', lambda: None); assert R.auto_workers(False) == 1


def test_with_a_gpu_the_workers_follow_the_card_slots_and_the_memory(monkeypatch):
    from stratastills import gpu
    monkeypatch.setattr(gpu, 'plan_workers', lambda: 3)
    monkeypatch.setattr(R, 'total_ram', lambda: 10 ** 13); assert R.auto_workers(True) == 3
    monkeypatch.setattr(R, 'total_ram', lambda: 0); assert R.auto_workers(True) == 1
    monkeypatch.setattr(R, 'total_ram', lambda: int(R.GPU_PICTURE_BYTES * 1.5 / 0.7)); assert R.auto_workers(True) == 1


def test_a_batch_shares_the_memory_between_its_pictures_and_restores_it_afterwards(monkeypatch):
    monkeypatch.setattr(R, 'total_ram', lambda: 1000)
    assert R.budget() == 700.0
    with R.batch(4): assert R.budget() == 175.0
    assert R.budget() == 700.0
    with R.batch(0): assert R.budget() == 700.0                                                          # never fewer than one
    monkeypatch.setattr(R, 'total_ram', lambda: 0); assert R.budget() is None


def test_the_tile_is_a_multiple_of_64_within_the_limits_and_the_largest_when_memory_is_unknown(monkeypatch):
    monkeypatch.setattr(R, 'budget', lambda: None); assert R.tile_for(10 ** 6, 200) == R.MAX_TILE
    assert R.tile_for(10 ** 6, 200, budget_bytes=10 ** 15) == R.MAX_TILE and R.tile_for(10 ** 6, 200, budget_bytes=1) == R.MIN_TILE
    t = R.tile_for(10 ** 6, 200, budget_bytes=200 * 1024 ** 2); assert t % 64 == 0 and R.MIN_TILE <= t <= R.MAX_TILE


def test_a_picture_that_fits_is_developed_whole_else_in_tiles_and_the_environment_can_force_a_tile(monkeypatch):
    monkeypatch.delenv('STRATASTILLS_TILE', raising=False)
    monkeypatch.setattr(R, 'budget', lambda: None); assert R.cpu_tile(10 ** 8) is None
    monkeypatch.setattr(R, 'budget', lambda: 10 ** 12); assert R.cpu_tile(10 ** 6) is None
    monkeypatch.setattr(R, 'budget', lambda: 2 * 10 ** 9); t = R.cpu_tile(48 * 10 ** 6); assert t is not None and R.MIN_TILE <= t <= R.MAX_TILE
    monkeypatch.setenv('STRATASTILLS_TILE', '10'); assert R.cpu_tile(10 ** 6) == 64                      # forced, but not below 64
