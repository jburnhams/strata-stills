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
