import unittest
import logging
from unittest.mock import patch

import graphsignal
import graphsignal.sdk
from graphsignal.sdk.signal_uploader import SignalUploader
from graphsignal.sdk.config_loader import ConfigLoader

logger = logging.getLogger('graphsignal')


class SdkTest(unittest.TestCase):
    def setUp(self):
        graphsignal.sdk.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal.sdk.sdk()._auto_tick = False

    def tearDown(self):
        graphsignal.sdk.shutdown()

    @patch.object(SignalUploader, 'upload_metric')
    @patch.object(SignalUploader, 'flush')
    @patch.object(ConfigLoader, 'update_config')
    def test_shutdown_upload(self, mocked_update_config, mocked_flush, mocked_upload_metric):
        graphsignal.sdk.shutdown()
        graphsignal.sdk.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal.sdk.sdk().set_gauge(name='n1', tags={}, value=1, measurement_ts=1)
        graphsignal.sdk.shutdown()

        self.assertTrue(mocked_upload_metric.call_count > 0)

