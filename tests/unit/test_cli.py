import pytest, sys, json
from stratastills import cli

def test_doctor_passes_when_everything_is_present(monkeypatch, capsys):
    monkeypatch.setattr(cli.importlib, 'import_module', lambda m: type('M', (), {'__version__': '1.0'})())
    monkeypatch.setattr(cli.shutil, 'which', lambda n: '/bin/ffmpeg')
    from stratastills import gpu, denoise_ai, externaltools
    monkeypatch.setattr(gpu, 'available', lambda: True)
    class FakeDev:
        type = 'cuda'
    monkeypatch.setattr(gpu, 'dev', lambda: FakeDev())
    class FakeTorch:
        __version__ = '2.0'
        class cuda:
            @staticmethod
            def get_device_name(n): return 'GPU'
            @staticmethod
            def is_available(): return True
        class backends:
            class cudnn:
                enabled = True
        @staticmethod
        def ones(*args, **kwargs):
            class T:
                def sum(self): return 1.0
                def __matmul__(self, other): return self
            return T()
    monkeypatch.setitem(sys.modules, 'torch', FakeTorch)
    monkeypatch.setattr(denoise_ai, 'available', lambda: True)
    monkeypatch.setitem(sys.modules, 'spandrel', type('M', (), {})())
    monkeypatch.setattr(externaltools, 'missing', lambda: [])
    assert cli.doctor(None) == 0
    out = capsys.readouterr().out
    assert 'ok' in out and 'MISSING' not in out

def test_doctor_fails_when_missing_deps(monkeypatch, capsys):
    monkeypatch.setattr(cli.importlib, 'import_module', lambda m: (_ for _ in ()).throw(ImportError('missing')))
    monkeypatch.setattr(cli.shutil, 'which', lambda n: None)
    from stratastills import gpu, denoise_ai, externaltools
    monkeypatch.setattr(gpu, 'available', lambda: False)
    monkeypatch.setitem(sys.modules, 'torch', None)
    monkeypatch.setitem(sys.modules, 'spandrel', None)
    monkeypatch.setattr(denoise_ai, 'available', lambda: False)
    monkeypatch.setattr(externaltools, 'missing', lambda: ['darktable'])
    assert cli.doctor(None) == 1
    out = capsys.readouterr().out
    assert 'MISSING' in out and 'missing' in out

def test_scan(monkeypatch, capsys):
    from stratastills import ingest, meta
    class Shot:
        def __init__(self):
            self.stem = 'shot'
            self.kind = 'still'
            self.raw = True
            self.jpeg = True
            self.video = False
            self.derived = []
        def to_json(self):
            return dict(stem=self.stem, kind=self.kind, raw=self.raw, jpeg=self.jpeg, video=self.video, derived=self.derived)

    monkeypatch.setattr(ingest, 'scan', lambda f, recursive: [Shot()])
    monkeypatch.setattr(meta, 'photo_info', lambda j: dict(taken=None, dji={'GimbalPitchDegree': 0}, width=10, height=10))
    monkeypatch.setattr(meta, 'is_nadir', lambda d: True)

    class Args:
        folder = 'f'
        recursive = False
        info = True
        json = False

    assert cli.scan(Args()) == 0
    out = capsys.readouterr().out
    assert 'shot' in out

def test_scan_json(monkeypatch, capsys):
    from stratastills import ingest
    monkeypatch.setattr(ingest, 'scan', lambda f, recursive: [])
    class Args:
        folder = 'f'
        recursive = False
        info = False
        json = True
    assert cli.scan(Args()) == 0
    out = capsys.readouterr().out
    assert json.loads(out) == []

def test_cmd_projects(monkeypatch, capsys):
    from stratastills import projects, settings
    monkeypatch.setattr(projects, 'list_projects', lambda: [{'id': '1', 'name': 'n', 'n_sources': 1}])
    assert cli.cmd_projects(None) == 0
    out = capsys.readouterr().out
    assert 'n' in out

def test_cmd_new(monkeypatch, capsys):
    from stratastills import projects
    monkeypatch.setattr(projects, 'create', lambda n, s, group: {'id': '1'})
    monkeypatch.setattr(projects, 'folder', lambda id: 'f')
    class Args:
        name = 'n'
        source = []
        group = None
    assert cli.cmd_new(Args()) == 0

def test_cmd_source(monkeypatch, capsys):
    from stratastills import projects
    monkeypatch.setattr(projects, 'add_source', lambda p, f: {'id': '1', 'sources': ['f']})
    class Args:
        project = '1'
        folder = 'f'
        remove = False
    assert cli.cmd_source(Args()) == 0

def test_cmd_set(monkeypatch, capsys):
    from stratastills import settings
    monkeypatch.setattr(settings, 'all_settings', lambda: {'k': 'v'})
    monkeypatch.setattr(settings, 'save', lambda d: None)
    class Args:
        key = 'k'
        value = 'v'
    assert cli.cmd_set(Args()) == 0
    class ArgsNone:
        key = 'k'
        value = None
    assert cli.cmd_set(ArgsNone()) == 0

def test_cmd_serve(monkeypatch, capsys):
    from stratastills.server import app
    monkeypatch.setattr(app, 'main', lambda a: None)
    class Args:
        host = 'h'
        port = 80
    assert cli.cmd_serve(Args()) == 0

def test_main_parsing(monkeypatch):
    monkeypatch.setattr(cli, 'doctor', lambda args: 0)
    assert cli.main(['doctor']) == 0

def test_cmd_new_error(monkeypatch, capsys):
    from stratastills import projects
    monkeypatch.setattr(projects, 'create', lambda *a, **k: (_ for _ in ()).throw(projects.ProjectError('err')))
    class Args:
        name = 'n'
        source = []
        group = None
    with pytest.raises(SystemExit):
        cli.cmd_new(Args())
    out = capsys.readouterr().err

def test_cmd_source_error(monkeypatch, capsys):
    from stratastills import projects
    monkeypatch.setattr(projects, 'add_source', lambda *a, **k: (_ for _ in ()).throw(projects.ProjectError('err')))
    monkeypatch.setattr(projects, 'remove_source', lambda *a, **k: (_ for _ in ()).throw(projects.ProjectError('err')))
    class Args:
        project = '1'
        folder = 'f'
        remove = False
    with pytest.raises(SystemExit):
        cli.cmd_source(Args())
    class ArgsRem:
        project = '1'
        folder = 'f'
        remove = True
    with pytest.raises(SystemExit):
        cli.cmd_source(ArgsRem())

def test_cmd_set_error(monkeypatch, capsys):
    from stratastills import settings
    monkeypatch.setattr(settings, 'all_settings', lambda: {'k': 'v'})
    monkeypatch.setattr(settings, 'save', lambda d: (_ for _ in ()).throw(KeyError('err')))
    class Args:
        key = 'k'
        value = 'v'
    with pytest.raises(SystemExit):
        cli.cmd_set(Args())

def test_scan_no_results(monkeypatch, capsys):
    from stratastills import ingest
    monkeypatch.setattr(ingest, 'scan', lambda f, recursive: [])
    class Args:
        folder = 'f'
        recursive = False
        info = False
        json = False
    assert cli.scan(Args()) == 0
    out = capsys.readouterr().out
    assert '0 shots' in out

def test_scan_info_no_taken_no_gimbal(monkeypatch, capsys):
    from stratastills import ingest, meta
    class Shot:
        def __init__(self):
            self.stem = 'shot'
            self.kind = 'still'
            self.raw = True
            self.jpeg = True
            self.video = False
            self.derived = []
        def to_json(self):
            return dict(stem=self.stem, kind=self.kind, raw=self.raw, jpeg=self.jpeg, video=self.video, derived=self.derived)

    monkeypatch.setattr(ingest, 'scan', lambda f, recursive: [Shot()])
    monkeypatch.setattr(meta, 'photo_info', lambda j: dict(taken=None, dji={}, width=10, height=10))
    monkeypatch.setattr(meta, 'is_nadir', lambda d: False)
    class Args:
        folder = 'f'
        recursive = False
        info = True
        json = False
    assert cli.scan(Args()) == 0
    out = capsys.readouterr().out
    assert 'shot' in out

def test_cmd_projects_empty(monkeypatch, capsys):
    from stratastills import projects, settings
    monkeypatch.setattr(projects, 'list_projects', lambda: [])
    assert cli.cmd_projects(None) == 0
    out = capsys.readouterr().out
    assert 'none yet' in out

def test_doctor_gpu_cudnn_disabled(monkeypatch, capsys):
    import sys
    monkeypatch.setattr(cli.importlib, 'import_module', lambda m: type('M', (), {'__version__': '1.0'})())
    monkeypatch.setattr(cli.shutil, 'which', lambda n: '/bin/ffmpeg')
    from stratastills import gpu, denoise_ai, externaltools
    monkeypatch.setattr(gpu, 'available', lambda: True)
    class FakeDev:
        type = 'cuda'
    monkeypatch.setattr(gpu, 'dev', lambda: FakeDev())
    class FakeTorch:
        __version__ = '2.0'
        class cuda:
            @staticmethod
            def get_device_name(n): return 'GPU'
            @staticmethod
            def is_available(): return True
        class backends:
            class cudnn:
                enabled = False
        @staticmethod
        def ones(*args, **kwargs):
            class T:
                def sum(self): return 1.0
                def __matmul__(self, other): return self
            return T()
    monkeypatch.setitem(sys.modules, 'torch', FakeTorch)
    monkeypatch.setattr(denoise_ai, 'available', lambda: True)
    monkeypatch.setitem(sys.modules, 'spandrel', type('M', (), {})())
    monkeypatch.setattr(externaltools, 'missing', lambda: [])
    assert cli.doctor(None) == 0
    out = capsys.readouterr().out
    assert 'cuDNN does not work on this card' in out

def test_doctor_gpu_no_spandrel(monkeypatch, capsys):
    import sys
    monkeypatch.setattr(cli.importlib, 'import_module', lambda m: type('M', (), {'__version__': '1.0'})())
    monkeypatch.setattr(cli.shutil, 'which', lambda n: '/bin/ffmpeg')
    from stratastills import gpu, denoise_ai, externaltools
    monkeypatch.setattr(gpu, 'available', lambda: True)
    class FakeDev:
        type = 'cuda'
    monkeypatch.setattr(gpu, 'dev', lambda: FakeDev())
    class FakeTorch:
        __version__ = '2.0'
        class cuda:
            @staticmethod
            def get_device_name(n): return 'GPU'
            @staticmethod
            def is_available(): return True
        class backends:
            class cudnn:
                enabled = True
        @staticmethod
        def ones(*args, **kwargs):
            class T:
                def sum(self): return 1.0
                def __matmul__(self, other): return self
            return T()
    monkeypatch.setitem(sys.modules, 'torch', FakeTorch)
    monkeypatch.setattr(denoise_ai, 'available', lambda: True)

    # Simulate missing spandrel
    monkeypatch.setitem(sys.modules, 'spandrel', None)
    monkeypatch.setattr(externaltools, 'missing', lambda: [])
    assert cli.doctor(None) == 0
    out = capsys.readouterr().out
    assert 'missing  learned denoise' in out

def test_doctor_gpu_broken(monkeypatch, capsys):
    import sys
    monkeypatch.setattr(cli.importlib, 'import_module', lambda m: type('M', (), {'__version__': '1.0'})())
    monkeypatch.setattr(cli.shutil, 'which', lambda n: '/bin/ffmpeg')
    from stratastills import gpu, denoise_ai, externaltools
    monkeypatch.setattr(gpu, 'available', lambda: True)
    class FakeDev:
        type = 'cuda'
    monkeypatch.setattr(gpu, 'dev', lambda: FakeDev())
    class FakeTorch:
        __version__ = '2.0'
        class cuda:
            @staticmethod
            def get_device_name(n): return 'GPU'
            @staticmethod
            def is_available(): return True
        class backends:
            class cudnn:
                enabled = True
        @staticmethod
        def ones(*args, **kwargs):
            raise Exception('Cannot run kernel')
    monkeypatch.setitem(sys.modules, 'torch', FakeTorch)
    monkeypatch.setattr(denoise_ai, 'available', lambda: True)
    monkeypatch.setitem(sys.modules, 'spandrel', type('M', (), {})())
    monkeypatch.setattr(externaltools, 'missing', lambda: [])
    assert cli.doctor(None) == 1
    out = capsys.readouterr().out
    assert 'BROKEN' in out and 'Cannot run kernel' in out

def test_doctor_gpu_not_available(monkeypatch, capsys):
    import sys
    monkeypatch.setattr(cli.importlib, 'import_module', lambda m: type('M', (), {'__version__': '1.0'})())
    monkeypatch.setattr(cli.shutil, 'which', lambda n: '/bin/ffmpeg')
    from stratastills import gpu, denoise_ai, externaltools
    monkeypatch.setattr(gpu, 'available', lambda: False)
    class FakeTorch:
        __version__ = '2.0'
    monkeypatch.setitem(sys.modules, 'torch', FakeTorch)
    monkeypatch.setattr(denoise_ai, 'available', lambda: True)
    monkeypatch.setitem(sys.modules, 'spandrel', type('M', (), {})())
    monkeypatch.setattr(externaltools, 'missing', lambda: [])
    assert cli.doctor(None) == 0
    out = capsys.readouterr().out
    assert 'no CUDA or MPS device' in out

def test_scan_info_gimbal_nadir_taken(monkeypatch, capsys):
    from stratastills import ingest, meta
    import datetime
    class Shot:
        def __init__(self):
            self.stem = 'shot'
            self.kind = 'still'
            self.raw = True
            self.jpeg = True
            self.video = False
            self.derived = []
        def to_json(self):
            return dict(stem=self.stem, kind=self.kind, raw=self.raw, jpeg=self.jpeg, video=self.video, derived=self.derived)

    monkeypatch.setattr(ingest, 'scan', lambda f, recursive: [Shot()])
    monkeypatch.setattr(meta, 'photo_info', lambda j: dict(taken=datetime.datetime(2020, 1, 1), dji={'GimbalPitchDegree': 90}, width=10, height=10))
    monkeypatch.setattr(meta, 'is_nadir', lambda d: True)
    class Args:
        folder = 'f'
        recursive = False
        info = True
        json = False
    assert cli.scan(Args()) == 0
    out = capsys.readouterr().out
    assert 'pitch 90' in out and 'nadir' in out and '2020-01-01' in out
