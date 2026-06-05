import logging
import time
from typing import Optional, Tuple

import psutil
try:
    from prometheus_client.parser import text_string_to_metric_families
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import urllib.request
    import urllib.error
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False

import graphsignal
import graphsignal.sdk
from graphsignal.recorders.base_recorder import BaseRecorder

logger = logging.getLogger('graphsignal')

INITIAL_DETECT_DELAY_SEC = 2.0
MAX_DETECT_DELAY_SEC = 60.0


class PrometheusRecorder(BaseRecorder):
    def __init__(self, pid=None, args=None):
        super().__init__(pid=pid, args=args)
        self._endpoint: Optional[str] = None
        self._last_values: dict = {}
        self._next_detect_ts: float = 0.0
        self._detect_delay_sec: float = INITIAL_DETECT_DELAY_SEC

    def setup(self):
        # Detection is lazy; first on_tick tries after the initial delay.
        self._next_detect_ts = time.time() + INITIAL_DETECT_DELAY_SEC

    def on_tick(self):
        if not PROMETHEUS_AVAILABLE or not HTTP_AVAILABLE:
            return

        if self._endpoint is None:
            if time.time() < self._next_detect_ts:
                return
            self._endpoint = self._detect_endpoint()
            if self._endpoint is None:
                # back off with cap
                self._detect_delay_sec = min(self._detect_delay_sec * 2, MAX_DETECT_DELAY_SEC)
                self._next_detect_ts = time.time() + self._detect_delay_sec
                return
            logger.debug('Prometheus /metrics endpoint discovered: %s', self._endpoint)

        try:
            body = self._fetch_metrics(self._endpoint)
        except Exception as exc:
            logger.debug('Failed to fetch %s: %s', self._endpoint, exc)
            self._endpoint = None
            self._detect_delay_sec = INITIAL_DETECT_DELAY_SEC
            self._next_detect_ts = time.time() + self._detect_delay_sec
            return

        try:
            self._parse_and_emit(body)
        except Exception as exc:
            logger.error('Failed to parse Prometheus metrics: %s', exc, exc_info=True)

    def _candidate_pids(self):
        if self.pid is None:
            return []
        pids = [self.pid]
        try:
            for child in psutil.Process(self.pid).children(recursive=True):
                pids.append(child.pid)
        except psutil.Error:
            pass
        return pids

    def _candidate_ports(self):
        seen = set()
        for pid in self._candidate_pids():
            try:
                proc = psutil.Process(pid)
                conns = proc.net_connections(kind='inet')
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error):
                continue
            for c in conns:
                if c.status != psutil.CONN_LISTEN:
                    continue
                port = getattr(c.laddr, 'port', None)
                if port and port not in seen:
                    seen.add(port)
                    yield port

    def _detect_endpoint(self) -> Optional[str]:
        for port in self._candidate_ports():
            url = f'http://127.0.0.1:{port}/metrics'
            try:
                body = self._fetch_metrics(url, timeout=1.0)
            except Exception:
                continue
            if _looks_like_prometheus(body):
                return url
        return None

    @staticmethod
    def _fetch_metrics(url: str, timeout: float = 2.0) -> str:
        req = urllib.request.Request(url, headers={'Accept': 'text/plain'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            charset = resp.headers.get_content_charset() or 'utf-8'
            return data.decode(charset, errors='replace')

    def _parse_and_emit(self, body: str) -> None:
        sdk = graphsignal.sdk.sdk()
        now_ns = time.time_ns()

        for family in text_string_to_metric_families(body):
            name = family.name
            mtype = family.type

            sample_groups = {}
            for sample in family.samples:
                labels = {k: v for k, v in sample.labels.items() if k not in ('le', 'quantile')}
                group_key = frozenset(labels.items())
                sample_groups.setdefault(group_key, {})[sample.name] = sample

            for group_key, sample_map in sample_groups.items():
                tags = dict(group_key)

                if mtype == 'gauge':
                    s = sample_map.get(name)
                    if s is not None:
                        sdk.set_gauge(name=name, tags=tags, value=s.value, measurement_ts=now_ns)

                elif mtype == 'counter':
                    s = sample_map.get(f'{name}_total') or sample_map.get(name)
                    if s is not None:
                        last_key = (name, group_key)
                        current = s.value
                        prev = self._last_values.get(last_key)
                        if prev is not None:
                            delta = current - prev
                            if delta >= 0:
                                sdk.inc_counter(name=name, tags=tags, value=delta, measurement_ts=now_ns)
                        self._last_values[last_key] = current

                elif mtype in ('histogram', 'summary'):
                    c = sample_map.get(f'{name}_count')
                    su = sample_map.get(f'{name}_sum')
                    if c is not None and su is not None:
                        last_key = (name, group_key)
                        cur_c, cur_s = c.value, su.value
                        prev = self._last_values.get(last_key)
                        if prev is not None:
                            dc, ds = cur_c - prev[0], cur_s - prev[1]
                            if dc > 0:
                                sdk.update_summary(name=name, tags=tags,
                                                   count=int(dc), sum_val=ds, measurement_ts=now_ns)
                        self._last_values[last_key] = (cur_c, cur_s)


def _looks_like_prometheus(body: str) -> bool:
    if not body:
        return False
    # OpenMetrics/Prometheus payloads begin with HELP/TYPE comments or metric samples.
    for line in body.splitlines():
        if not line:
            continue
        if line.startswith('# HELP') or line.startswith('# TYPE'):
            return True
        if line.startswith('#'):
            continue
        if ' ' in line and not line.startswith('<'):
            return True
        return False
    return False
