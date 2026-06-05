import logging
import time

import graphsignal
import graphsignal.sdk
from graphsignal.proto import signals_pb2

logger = logging.getLogger('graphsignal')


class ResourceStore:
    def __init__(self):
        self._resources = {}

    def update_resource(
            self,
            kind,
            tags=None,
            attributes=None,
            first_seen_ts=None,
            last_seen_ts=None):
        if kind is None:
            return

        now_ns = time.time_ns()
        if first_seen_ts is None:
            first_seen_ts = now_ns
        if last_seen_ts is None:
            last_seen_ts = now_ns

        all_tags = graphsignal.sdk.sdk().tags()
        if tags:
            all_tags.update(tags)

        resource_key = (kind, frozenset(all_tags.items()))

        resource = signals_pb2.Resource()
        resource.kind = kind
        for k, v in all_tags.items():
            tag = resource.tags.add()
            tag.key = str(k)[:50]
            tag.value = str(v)[:250]
        if attributes:
            for k, v in attributes.items():
                if v is None:
                    continue
                attr = resource.attributes.add()
                attr.name = str(k)[:50]
                attr.value = str(v)[:2500]
        resource.first_seen_ts = first_seen_ts
        resource.last_seen_ts = last_seen_ts

        self._resources[resource_key] = resource

    def has_unexported(self):
        return len(self._resources) > 0

    def export(self):
        resources = list(self._resources.values())
        self._resources.clear()
        return resources

    def clear(self):
        self._resources.clear()
