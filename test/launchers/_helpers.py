"""Shared test helpers for per-launcher test files.

`LaunchFixture` mocks the side effects that every launcher's `launch()` method
triggers (OTEL port discovery, CUPTI env setup, watcher subprocess spawn,
target resolution, and the final `os.execv`) so a test can assert on the
launcher's argv mutations and ordering without touching the OS.
"""

from unittest.mock import MagicMock, patch


class LaunchFixture:
    """Shared mocks for launch() smoke tests across launchers.

    Pass the launcher module (e.g. ``graphsignal.launchers.vllm_launcher``)
    so the patches land on the symbols actually referenced by that module
    (each launcher imports `start_watcher` / `_resolve` by name).
    """

    def __init__(self, module):
        self.module = module
        self.find_port = patch.object(module.OTELCollector, 'find_port', return_value=4242)
        self.setup_env = patch.object(module.CuptiProfiler, 'setup_env_vars', return_value=True)
        self.start_watcher = patch.object(module, 'start_watcher', return_value=MagicMock())
        self.execv = patch('os.execv')
        self.resolve = patch.object(module, '_resolve', return_value='/abs/exec')

    def __enter__(self):
        self.find_port_m = self.find_port.start()
        self.setup_env_m = self.setup_env.start()
        self.start_watcher_m = self.start_watcher.start()
        self.execv_m = self.execv.start()
        self.resolve_m = self.resolve.start()
        return self

    def __exit__(self, *exc):
        self.resolve.stop()
        self.execv.stop()
        self.start_watcher.stop()
        self.setup_env.stop()
        self.find_port.stop()
