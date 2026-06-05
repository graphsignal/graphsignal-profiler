import logging
import os
import sys

from graphsignal.launchers.vllm_launcher import VllmLauncher
from graphsignal.launchers.sglang_launcher import SglangLauncher
from graphsignal.launchers.trtllm_launcher import TrtllmLauncher
from graphsignal.launchers.fallback_launcher import FallbackLauncher

log = logging.getLogger(__name__)


def _setup_logging():
    """Emit the `graphsignal` logger to stderr so launcher diagnostics (which
    launcher matched, the final injected args, the watcher command + OTEL port)
    are visible in the workload's log instead of being silently dropped. The
    launcher runs in the workload process before `execv`, so without this its
    debug output goes nowhere. Verbosity follows GRAPHSIGNAL_DEBUG."""
    debug = os.getenv('GRAPHSIGNAL_DEBUG', '').strip().lower() in ('1', 'true', 'yes')
    gs_logger = logging.getLogger('graphsignal')
    gs_logger.setLevel(logging.DEBUG if debug else logging.WARNING)
    if not any(isinstance(h, logging.StreamHandler) for h in gs_logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s graphsignal-run: %(message)s'))
        gs_logger.addHandler(handler)

USAGE = """
Run a target application with the Graphsignal profiler.

Options (must precede the command):
  --enable-otel   Enable OpenTelemetry trace capture for supported engines
                  (injects the engine's trace flags and starts a local OTLP
                  collector). Requires OpenTelemetry installed in the engine's
                  environment. Off by default.

Example:
  graphsignal-run vllm serve facebook/opt-125m --port 8001
  graphsignal-run --enable-otel sglang serve --model-path <model>
  graphsignal-run python myapp.py
  graphsignal-run app.py
"""


def _extract_graphsignal_flags(argv):
    """Pull graphsignal-run's own flags out of argv before the workload command.

    `--enable-otel` opts into OTEL trace injection (engine trace flags + local
    collector). It is consumed here and never forwarded to the workload. Only
    leading flags (before the workload command) are parsed, so an identically
    named workload flag later in argv is left untouched.
    """
    enable_otel = False
    i = 0
    while i < len(argv):
        if argv[i] == '--enable-otel':
            enable_otel = True
            i += 1
            continue
        break
    return enable_otel, argv[i:]


def main():
    if len(sys.argv) < 2:
        print("graphsignal-run: no command specified\n")
        print(USAGE)
        sys.exit(1)

    _setup_logging()

    enable_otel, target_args = _extract_graphsignal_flags(sys.argv[1:])
    log.debug('graphsignal-run target args: %s (enable_otel=%s)', target_args, enable_otel)

    launchers = [
        VllmLauncher(target_args, enable_otel=enable_otel),
        SglangLauncher(target_args, enable_otel=enable_otel),
        TrtllmLauncher(target_args, enable_otel=enable_otel),
        FallbackLauncher(target_args, enable_otel=enable_otel),
    ]

    for launcher in launchers:
        try:
            matched = launcher.match()
        except Exception as exc:
            log.error('Error matching launcher %s: %s', type(launcher).__name__, exc, exc_info=True)
            continue
        if matched:
            log.debug('Selected launcher: %s', type(launcher).__name__)
            launcher.launch()
            return

    print("graphsignal-run: no launcher matched")
    sys.exit(1)


if __name__ == '__main__':
    main()
