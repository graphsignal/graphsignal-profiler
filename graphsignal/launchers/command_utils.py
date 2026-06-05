import logging
import subprocess
import sys
from typing import Optional

logger = logging.getLogger('graphsignal')


def start_watcher(pid: int, otel_collector_port: Optional[int] = None) -> Optional[subprocess.Popen]:
    """Spawn the graphsignal-watch subprocess to observe the given pid.

    Used by `graphsignal.watch()` (pid = self) and by the per-engine launchers,
    which call this just before `execv`-ing the workload (pid = self, which
    becomes the workload's pid after exec).
    """
    cmd = [
        sys.executable,
        '-m', 'graphsignal.commands.graphsignal_watch',
        '--pid', str(int(pid)),
    ]
    if otel_collector_port is not None:
        cmd.extend(['--otel-collector-port', str(int(otel_collector_port))])

    logger.debug('Starting graphsignal-watch: %s', ' '.join(cmd))
    try:
        return subprocess.Popen(
            cmd,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            close_fds=True,
        )
    except Exception as exc:
        logger.error('Failed to start graphsignal-watch: %s', exc, exc_info=True)
        return None
