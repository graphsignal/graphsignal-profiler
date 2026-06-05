import logging
import os
import shutil

from graphsignal.launchers.base_launcher import BaseLauncher
from graphsignal.launchers.command_utils import start_watcher
from graphsignal.otel.otel_collector import OTELCollector
from graphsignal.profilers.cupti_profiler import CuptiProfiler

logger = logging.getLogger('graphsignal')

_TRTLLM_NAMES = {'trtllm', 'trtllm-serve', 'trtllm-llmapi-launch'}


class TrtllmLauncher(BaseLauncher):
    def match(self) -> bool:
        return self.executable_name() in _TRTLLM_NAMES

    def launch(self) -> None:
        # Only `trtllm-serve` (the OpenAI-compatible HTTP server) supports
        # `--otlp_traces_endpoint`. For other entry points (e.g.
        # `trtllm-llmapi-launch <user_script>`) the workload is arbitrary
        # Python and we don't touch argv.
        is_serve = self._is_serve_command()

        # OTEL trace injection is opt-in via `graphsignal-run --enable-otel`
        # (requires OpenTelemetry installed in the TRT-LLM environment).
        if not self.enable_otel:
            otel_port = None
            logger.debug('TRT-LLM: OTEL tracing not enabled '
                         '(pass --enable-otel to graphsignal-run to enable)')
        elif is_serve and not _has_flag(self.args, '--otlp_traces_endpoint'):
            otel_port = OTELCollector.find_port()
        else:
            otel_port = None

        CuptiProfiler.setup_env_vars()

        new_args = _inject_trtllm_args(self.args, otel_port) if is_serve \
            else list(self.args)

        start_watcher(os.getpid(), otel_collector_port=otel_port)

        executable = _resolve(new_args[0])
        if not executable:
            raise FileNotFoundError(f'executable not found: {new_args[0]}')

        logger.debug('TrtllmLauncher exec: %s %s', executable, new_args)
        os.execv(executable, new_args)

    def _is_serve_command(self) -> bool:
        # Either `trtllm-serve …` directly, or `trtllm serve …` (subcommand
        # form, kept for forward compat with the unified-CLI proposals).
        name = self.executable_name()
        if name == 'trtllm-serve':
            return True
        if name == 'trtllm' and len(self.args) >= 2 and self.args[1] == 'serve':
            return True
        return False


def _inject_trtllm_args(args, otel_port):
    args = list(args)

    # Only inject our OTEL endpoint when the caller allocated a port.
    # `trtllm-serve` uses the underscore form `--otlp_traces_endpoint`.
    # The TensorRT backend's `/metrics` is on by default and gets picked
    # up automatically by `PrometheusRecorder` via port scanning, so no
    # metrics-side argv mutation is needed.
    if otel_port is not None and not _has_flag(args, '--otlp_traces_endpoint'):
        # Explicit IPv4 loopback (not "localhost") so the exporter and the
        # collector can't disagree on IPv4 vs IPv6.
        args.extend(['--otlp_traces_endpoint', f'127.0.0.1:{otel_port}'])

    return args


def _has_flag(args, flag) -> bool:
    for a in args:
        if a == flag or a.startswith(flag + '='):
            return True
    return False


def _resolve(name):
    if os.path.isabs(name) and os.path.isfile(name):
        return name
    return shutil.which(name)
