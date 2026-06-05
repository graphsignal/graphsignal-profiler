import subprocess
import sys
import unittest
from unittest.mock import patch, MagicMock

from graphsignal.launchers.command_utils import start_watcher


class StartWatcherTest(unittest.TestCase):
    def test_spawn_args_default(self):
        fake_popen = MagicMock(name='Popen')
        with patch.object(subprocess, 'Popen', return_value=fake_popen) as popen_m:
            result = start_watcher(12345)

        self.assertIs(result, fake_popen)
        popen_m.assert_called_once()
        cmd = popen_m.call_args[0][0]
        kwargs = popen_m.call_args[1]

        self.assertEqual(cmd[0], sys.executable)
        self.assertEqual(cmd[1:5], ['-m', 'graphsignal.commands.graphsignal_watch',
                                    '--pid', '12345'])
        # `--otel-collector-port` is omitted when not provided.
        self.assertNotIn('--otel-collector-port', cmd)
        # Watcher must be its own session so it survives `os.execv` in the parent.
        self.assertTrue(kwargs.get('start_new_session'))

    def test_spawn_args_with_otel_port(self):
        with patch.object(subprocess, 'Popen', return_value=MagicMock()) as popen_m:
            start_watcher(54321, otel_collector_port=4317)
        cmd = popen_m.call_args[0][0]
        self.assertEqual(cmd[5:], ['--otel-collector-port', '4317'])

    def test_returns_none_on_failure(self):
        with patch.object(subprocess, 'Popen', side_effect=OSError('boom')):
            result = start_watcher(1)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
