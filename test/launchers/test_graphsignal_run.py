import sys
import unittest
from unittest.mock import patch

from graphsignal.commands import graphsignal_run
from graphsignal.launchers.vllm_launcher import VllmLauncher
from graphsignal.launchers.sglang_launcher import SglangLauncher
from graphsignal.launchers.trtllm_launcher import TrtllmLauncher
from graphsignal.launchers.fallback_launcher import FallbackLauncher


class GraphsignalRunDispatchTest(unittest.TestCase):
    """`graphsignal-run` walks the launcher list and dispatches to the first
    one whose match() returns True."""

    def _run_with_argv(self, argv):
        with patch.object(VllmLauncher, 'launch') as v, \
             patch.object(SglangLauncher, 'launch') as s, \
             patch.object(TrtllmLauncher, 'launch') as t, \
             patch.object(FallbackLauncher, 'launch') as f, \
             patch.object(sys, 'argv', argv):
            try:
                graphsignal_run.main()
            except SystemExit:
                pass
        return v, s, t, f

    def test_no_args_exits(self):
        with patch.object(sys, 'argv', ['graphsignal-run']):
            with self.assertRaises(SystemExit) as cm:
                graphsignal_run.main()
        self.assertEqual(cm.exception.code, 1)

    def test_vllm_wins_over_fallback(self):
        v, s, t, f = self._run_with_argv(['graphsignal-run', 'vllm', 'serve', 'm'])
        v.assert_called_once()
        s.assert_not_called()
        t.assert_not_called()
        f.assert_not_called()

    def test_sglang_executable_wins(self):
        v, s, t, f = self._run_with_argv(
            ['graphsignal-run', 'sglang', 'serve', '--model', 'm'])
        s.assert_called_once()
        v.assert_not_called()
        t.assert_not_called()
        f.assert_not_called()

    def test_sglang_python_module_form_wins(self):
        v, s, t, f = self._run_with_argv(
            ['graphsignal-run', 'python', '-m', 'sglang.launch_server'])
        s.assert_called_once()
        f.assert_not_called()

    def test_trtllm_wins(self):
        v, s, t, f = self._run_with_argv(['graphsignal-run', 'trtllm-serve', '--model', 'm'])
        t.assert_called_once()
        v.assert_not_called()
        s.assert_not_called()
        f.assert_not_called()

    def test_unrecognised_falls_back(self):
        v, s, t, f = self._run_with_argv(['graphsignal-run', 'python', 'my_app.py'])
        f.assert_called_once()
        v.assert_not_called()
        s.assert_not_called()
        t.assert_not_called()

    def test_enable_otel_flag_stripped_and_command_still_matches(self):
        # `--enable-otel` is graphsignal-run's own flag: consumed before the
        # command, so the workload command still matches its launcher.
        v, s, t, f = self._run_with_argv(
            ['graphsignal-run', '--enable-otel', 'vllm', 'serve', 'm'])
        v.assert_called_once()
        s.assert_not_called()
        t.assert_not_called()
        f.assert_not_called()


class ExtractFlagsTest(unittest.TestCase):
    def test_extracts_leading_enable_otel(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(['--enable-otel', 'vllm', 'serve']),
            (True, ['vllm', 'serve']))

    def test_absent_flag_defaults_off(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(['vllm', 'serve']),
            (False, ['vllm', 'serve']))

    def test_flag_after_command_left_for_workload(self):
        # Only leading flags are parsed; an identically named workload flag
        # later in argv is forwarded untouched.
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(['vllm', '--enable-otel']),
            (False, ['vllm', '--enable-otel']))


if __name__ == '__main__':
    unittest.main()
