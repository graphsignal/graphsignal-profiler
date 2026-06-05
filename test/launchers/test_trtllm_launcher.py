import os
import unittest

from graphsignal.launchers import trtllm_launcher as trtllm_mod
from graphsignal.launchers.trtllm_launcher import TrtllmLauncher, _inject_trtllm_args

from test.launchers._helpers import LaunchFixture


class TrtllmMatchTest(unittest.TestCase):
    def test_matches(self):
        self.assertTrue(TrtllmLauncher(['trtllm', 'serve']).match())
        self.assertTrue(TrtllmLauncher(['trtllm-serve', '--model', 'm']).match())
        self.assertTrue(TrtllmLauncher(['trtllm-llmapi-launch']).match())
        self.assertTrue(TrtllmLauncher(['/usr/bin/trtllm-serve']).match())

    def test_does_not_match(self):
        self.assertFalse(TrtllmLauncher(['python', 'app.py']).match())
        self.assertFalse(TrtllmLauncher(['trtllm-other']).match())


class TrtllmArgInjectionTest(unittest.TestCase):
    def test_appends_otlp_endpoint_when_missing(self):
        out = _inject_trtllm_args(['trtllm-serve', 'model'], otel_port=4317)
        self.assertEqual(
            out, ['trtllm-serve', 'model', '--otlp_traces_endpoint', '127.0.0.1:4317'])

    def test_preserves_user_otlp_endpoint(self):
        out = _inject_trtllm_args(
            ['trtllm-serve', '--otlp_traces_endpoint', 'them:4317'],
            otel_port=4317)
        self.assertEqual(out.count('--otlp_traces_endpoint'), 1)
        idx = out.index('--otlp_traces_endpoint')
        self.assertEqual(out[idx + 1], 'them:4317')

    def test_no_otel_port_means_no_endpoint_injection(self):
        out = _inject_trtllm_args(
            ['trtllm-serve', '--otlp_traces_endpoint', 'them:4317'],
            otel_port=None)
        self.assertEqual(out.count('--otlp_traces_endpoint'), 1)


class TrtllmLaunchTest(unittest.TestCase):
    def test_serve_command_injects_otel_endpoint_when_otel_enabled(self):
        launcher = TrtllmLauncher(['trtllm-serve', '--model', 'm'], enable_otel=True)
        with LaunchFixture(trtllm_mod) as fx:
            launcher.launch()

        fx.find_port_m.assert_called_once()
        fx.start_watcher_m.assert_called_once_with(os.getpid(), otel_collector_port=4242)
        called_argv = fx.execv_m.call_args[0][1]
        self.assertIn('--otlp_traces_endpoint', called_argv)
        idx = called_argv.index('--otlp_traces_endpoint')
        self.assertEqual(called_argv[idx + 1], '127.0.0.1:4242')

    def test_serve_command_no_otel_by_default(self):
        # Default (no --enable-otel): no collector port, no endpoint injection.
        launcher = TrtllmLauncher(['trtllm-serve', '--model', 'm'])
        with LaunchFixture(trtllm_mod) as fx:
            launcher.launch()

        fx.find_port_m.assert_not_called()
        fx.start_watcher_m.assert_called_once_with(os.getpid(), otel_collector_port=None)
        called_argv = fx.execv_m.call_args[0][1]
        self.assertNotIn('--otlp_traces_endpoint', called_argv)

    def test_serve_command_skips_collector_when_user_endpoint_present(self):
        launcher = TrtllmLauncher(
            ['trtllm-serve', '--model', 'm', '--otlp_traces_endpoint', 'them:4317'],
            enable_otel=True)
        with LaunchFixture(trtllm_mod) as fx:
            launcher.launch()

        fx.find_port_m.assert_not_called()
        fx.start_watcher_m.assert_called_once_with(os.getpid(), otel_collector_port=None)
        called_argv = fx.execv_m.call_args[0][1]
        idx = called_argv.index('--otlp_traces_endpoint')
        self.assertEqual(called_argv[idx + 1], 'them:4317')

    def test_non_serve_command_skips_all_injection(self):
        # `trtllm-llmapi-launch` runs a user script — argv is forwarded
        # unchanged and no OTEL collector is spawned.
        launcher = TrtllmLauncher(['trtllm-llmapi-launch', 'user_script.py'])
        with LaunchFixture(trtllm_mod) as fx:
            launcher.launch()

        fx.find_port_m.assert_not_called()
        fx.start_watcher_m.assert_called_once_with(os.getpid(), otel_collector_port=None)
        called_argv = fx.execv_m.call_args[0][1]
        self.assertEqual(called_argv, ['trtllm-llmapi-launch', 'user_script.py'])
        self.assertNotIn('--otlp_traces_endpoint', called_argv)


if __name__ == '__main__':
    unittest.main()
