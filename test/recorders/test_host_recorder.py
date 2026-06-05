import unittest
import logging
import os
import sys

import graphsignal
import graphsignal.sdk
from graphsignal.recorders.host_recorder import HostRecorder
from test.test_utils import find_last_datapoint

logger = logging.getLogger('graphsignal')


class HostRecorderTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.sdk.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal.sdk.sdk()._auto_tick = False

    def tearDown(self):
        graphsignal.sdk.shutdown()

    def test_setup_sets_sdk_tags(self):
        recorder = HostRecorder()
        recorder.setup()

        self.assertIsNotNone(graphsignal.sdk.sdk().get_tag('host.name'))

    def test_on_tick_emits_host_memory_usage(self):
        recorder = HostRecorder()
        recorder.setup()
        recorder.on_tick()

        store = graphsignal.sdk.sdk().metric_store()
        key = store.metric_key('host.memory.usage', {})
        self.assertTrue(find_last_datapoint(store, key).gauge > 0)

    def test_on_tick_emits_host_resource(self):
        recorder = HostRecorder()
        recorder.setup()
        recorder.on_tick()

        resource_store = graphsignal.sdk.sdk().resource_store()
        resources = resource_store.export()

        host_resources = [r for r in resources if r.kind == 'host']
        self.assertEqual(len(host_resources), 1)
        host_resource = host_resources[0]

        attr_names = [a.name for a in host_resource.attributes]
        self.assertIn('platform.name', attr_names)
        self.assertIn('platform.version', attr_names)

        tag_keys = {t.key for t in host_resource.tags}
        self.assertIn('host.name', tag_keys)
