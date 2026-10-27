import importlib.util, os, pytest

spec = importlib.util.spec_from_file_location('setup_torch', os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'setup_torch.py')); ST = importlib.util.module_from_spec(spec); spec.loader.exec_module(ST)


@pytest.mark.parametrize('cards,system,plan', [
    ([('NVIDIA GeForce GTX 1050', (6, 1))], 'Linux', 'cu126'), ([('Tesla V100', (7, 0))], 'Linux', 'cu126'),
    ([('NVIDIA GeForce RTX 3060', (8, 6))], 'Windows', 'cu128'), ([('NVIDIA GeForce RTX 5090', (12, 0))], 'Windows', 'cu128'),
    ([('RTX 3060', (8, 6)), ('GTX 1050', (6, 1))], 'Linux', 'cu126'),                                  # the oldest card decides
    ([], 'Linux', 'cpu'), ([], 'Darwin', 'mac'), ([('x', (8, 0))], 'Darwin', 'mac'),
])
def test_the_plan_follows_the_oldest_card_and_the_os(cards, system, plan):
    assert ST.choose(cards, system)[0] == plan


def test_a_cudnn_of_the_wrong_cuda_is_removed_and_the_right_one_put_back(monkeypatch):
    calls = []
    monkeypatch.setattr(ST, 'installed', lambda py, prefix: {'nvidia-cudnn-cu12': '9.10.2.21', 'nvidia-cudnn-cu13': '9.24.0.43'})
    monkeypatch.setattr(ST, 'pip', lambda py, *a, dry=False: calls.append(a) or 0)
    assert ST.repair_cudnn('py', 'cu126') is True
    assert calls == [('uninstall', '-y', 'nvidia-cudnn-cu13'), ('install', '--no-deps', '--force-reinstall', 'nvidia-cudnn-cu12==9.10.2.21')]
    calls.clear(); monkeypatch.setattr(ST, 'installed', lambda py, prefix: {'nvidia-cudnn-cu12': '9.10.2.21'}); assert ST.repair_cudnn('py', 'cu126') is False and calls == []
    assert ST.repair_cudnn('py', 'cpu') is False


def test_a_plan_that_does_not_probe_ok_goes_down_the_ladder(monkeypatch):
    done = []
    monkeypatch.setattr(ST, 'gpus', lambda: [('RTX 3060', (8, 6))]); monkeypatch.setattr(ST, 'install', lambda py, plan, dry=False: done.append(plan) or 0); monkeypatch.setattr(ST, 'repair_cudnn', lambda *a, **k: False)
    monkeypatch.setattr(ST, 'probe', lambda py: dict(ok=len(done) > 1)); monkeypatch.setattr(ST.platform, 'system', lambda: 'Linux')
    assert ST.main(['--python', 'py']) == 0 and done == ['cu128', 'cu126']


@pytest.mark.parametrize('cards,system,amd,plan', [
    ([], 'Linux', [110003], 'rocm64'), ([], 'Linux', [110000], 'rocm64'), ([], 'Linux', [90006], 'cpu'),              # gfx1103, gfx1100, a Vega (gfx906): too old for ROCm 6.4 wheels
    ([], 'Windows', [110003], 'cpu'), ([], 'Darwin', [110003], 'mac'), ([('RTX 3060', (8, 6))], 'Linux', [110003], 'cu128'),   # an NVIDIA card wins over an AMD one
])
def test_an_amd_gpu_without_an_nvidia_card_gets_the_rocm_build_on_linux(cards, system, amd, plan):
    assert ST.choose(cards, system, amd)[0] == plan


def test_the_amd_chips_come_from_the_kernel_topology(tmp_path):
    for i, (vendor, ver) in enumerate([(0, 0), (4098, 110003), (4318, 80000)]):             # CPU node, a Radeon 780M, an NVIDIA node (not listed by KFD in practice, but not ours)
        d = tmp_path / str(i); d.mkdir(); (d / 'properties').write_text(f'cpu_cores_count 16\nvendor_id {vendor}\ngfx_target_version {ver}\n')
    assert ST.amd_chips(str(tmp_path)) == [110003] and ST.amd_chips(str(tmp_path / 'nothing')) == []


def test_only_the_rdna3_chips_below_gfx1100_get_the_override_and_a_set_value_wins(monkeypatch):
    monkeypatch.delenv('HSA_OVERRIDE_GFX_VERSION', raising=False)
    assert ST.amd_env([110003]) == {'HSA_OVERRIDE_GFX_VERSION': '11.0.0'} and ST.amd_env([110000]) == {} and ST.amd_env([]) == {}
    monkeypatch.setenv('HSA_OVERRIDE_GFX_VERSION', '10.3.0'); assert ST.amd_env([110003]) == {}
