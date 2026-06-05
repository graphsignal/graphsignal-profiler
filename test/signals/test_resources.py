import logging
import unittest
from unittest.mock import patch

import graphsignal
import graphsignal.sdk
from test.test_utils import find_attribute, find_tag

logger = logging.getLogger('graphsignal')


class ResourceStoreTest(unittest.TestCase):
    def setUp(self):
        graphsignal.sdk.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal.sdk.sdk()._auto_tick = False

    def tearDown(self):
        graphsignal.sdk.shutdown()

    def test_update_and_export(self):
        graphsignal.sdk.sdk().set_tag('host.name', 'h1')
        store = graphsignal.sdk.sdk().resource_store()
        store.clear()

        store.update_resource(
            kind='process',
            tags={'process.pid': '1234'},
            attributes={'process.command_line': 'python app.py',
                        'platform.name': 'Linux'},
            first_seen_ts=100,
            last_seen_ts=200)
        protos = store.export()

        self.assertEqual(len(protos), 1)
        proto = protos[0]
        self.assertEqual(proto.kind, 'process')
        self.assertEqual(proto.first_seen_ts, 100)
        self.assertEqual(proto.last_seen_ts, 200)
        # SDK tags merge in alongside the per-resource tag.
        self.assertEqual(find_tag(proto, 'host.name'), 'h1')
        self.assertEqual(find_tag(proto, 'process.pid'), '1234')
        # Attributes come through.
        self.assertEqual(find_attribute(proto, 'process.command_line'), 'python app.py')
        self.assertEqual(find_attribute(proto, 'platform.name'), 'Linux')

    def test_export_clears_store(self):
        store = graphsignal.sdk.sdk().resource_store()
        store.clear()
        store.update_resource(kind='device', tags={'device.id': '0'})
        self.assertTrue(store.has_unexported())

        first = store.export()
        self.assertEqual(len(first), 1)
        self.assertFalse(store.has_unexported())
        # Second export is empty.
        self.assertEqual(store.export(), [])

    def test_dedup_by_kind_and_tags(self):
        """The same (kind, tag-set) is upserted in place — only the latest
        update is kept."""
        store = graphsignal.sdk.sdk().resource_store()
        store.clear()

        store.update_resource(
            kind='device', tags={'device.id': '0'},
            attributes={'name': 'first'},
            first_seen_ts=10, last_seen_ts=20)
        store.update_resource(
            kind='device', tags={'device.id': '0'},
            attributes={'name': 'second'},
            first_seen_ts=30, last_seen_ts=40)

        protos = store.export()
        self.assertEqual(len(protos), 1)
        # Latest update wins.
        self.assertEqual(find_attribute(protos[0], 'name'), 'second')
        self.assertEqual(protos[0].first_seen_ts, 30)
        self.assertEqual(protos[0].last_seen_ts, 40)

    def test_distinct_kinds_kept_separately(self):
        store = graphsignal.sdk.sdk().resource_store()
        store.clear()
        store.update_resource(kind='process', tags={'process.pid': '1'})
        store.update_resource(kind='device', tags={'process.pid': '1'})

        protos = store.export()
        kinds = sorted(p.kind for p in protos)
        self.assertEqual(kinds, ['device', 'process'])

    def test_distinct_tag_sets_kept_separately(self):
        store = graphsignal.sdk.sdk().resource_store()
        store.clear()
        store.update_resource(kind='device', tags={'device.id': '0'})
        store.update_resource(kind='device', tags={'device.id': '1'})

        protos = store.export()
        self.assertEqual(len(protos), 2)
        device_ids = sorted(find_tag(p, 'device.id') for p in protos)
        self.assertEqual(device_ids, ['0', '1'])

    @patch('time.time_ns', return_value=99)
    def test_default_timestamps_use_time_ns(self, mocked_time_ns):
        store = graphsignal.sdk.sdk().resource_store()
        store.clear()
        store.update_resource(kind='process', tags={'process.pid': '7'})
        proto = store.export()[0]
        self.assertEqual(proto.first_seen_ts, 99)
        self.assertEqual(proto.last_seen_ts, 99)

    def test_none_kind_is_ignored(self):
        store = graphsignal.sdk.sdk().resource_store()
        store.clear()
        store.update_resource(kind=None, tags={'k': 'v'})
        self.assertFalse(store.has_unexported())

    def test_none_attribute_values_are_dropped(self):
        store = graphsignal.sdk.sdk().resource_store()
        store.clear()
        store.update_resource(
            kind='process', tags={'process.pid': '1'},
            attributes={'present': 'yes', 'absent': None})
        proto = store.export()[0]
        self.assertEqual(find_attribute(proto, 'present'), 'yes')
        self.assertIsNone(find_attribute(proto, 'absent'))

    def test_long_strings_are_truncated(self):
        store = graphsignal.sdk.sdk().resource_store()
        store.clear()
        long_key = 'x' * 100
        long_tag_value = 'y' * 500
        long_attr_value = 'z' * 5000
        store.update_resource(
            kind='process',
            tags={long_key: long_tag_value},
            attributes={long_key: long_attr_value})

        proto = store.export()[0]
        tag = next(t for t in proto.tags if t.key.startswith('x'))
        self.assertEqual(len(tag.key), 50)        # tag key truncated to 50
        self.assertEqual(len(tag.value), 250)     # tag value truncated to 250

        attr = next(a for a in proto.attributes if a.name.startswith('x'))
        self.assertEqual(len(attr.name), 50)      # attribute name truncated to 50
        self.assertEqual(len(attr.value), 2500)   # attribute value truncated to 2500

    def test_has_unexported(self):
        store = graphsignal.sdk.sdk().resource_store()
        store.clear()
        self.assertFalse(store.has_unexported())
        store.update_resource(kind='process', tags={'process.pid': '1'})
        self.assertTrue(store.has_unexported())


if __name__ == '__main__':
    unittest.main()
