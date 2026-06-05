import logging
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

import graphsignal
import graphsignal.sdk
from graphsignal.signals.spans import SpanStore, TokenBucketSampler
from test.test_utils import find_tag, find_attribute, find_counter


logger = logging.getLogger('graphsignal')


def _config_loader(traces_per_sec=1000.0):
    """A stand-in config loader that returns fixed option values."""
    loader = MagicMock()
    loader.get_float_option.side_effect = (
        lambda name: traces_per_sec if name == 'traces_per_sec' else None)
    return loader


class TokenBucketSamplerTest(unittest.TestCase):
    def test_first_call_samples(self):
        sampler = TokenBucketSampler(rate_per_sec=1000.0)
        self.assertTrue(sampler.should_sample())

    def test_second_call_skips_when_bucket_empty(self):
        sampler = TokenBucketSampler(rate_per_sec=0.001, capacity=1.0)
        self.assertTrue(sampler.should_sample())
        self.assertFalse(sampler.should_sample())

    def test_refill_allows_after_elapsed_time(self):
        sampler = TokenBucketSampler(rate_per_sec=1.0, capacity=1.0)
        with patch.object(TokenBucketSampler, '_now_ms', side_effect=[0, 0, 1000]):
            self.assertTrue(sampler.should_sample())
            self.assertFalse(sampler.should_sample())
            self.assertTrue(sampler.should_sample())

    def test_rejects_out_of_range_rate(self):
        with self.assertRaises(ValueError):
            TokenBucketSampler(rate_per_sec=0)
        with self.assertRaises(ValueError):
            TokenBucketSampler(rate_per_sec=10000)


class SpanStoreTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.sdk.configure(api_key='k1', tags={'deployment': 'd1'}, debug_mode=True)
        graphsignal.sdk.sdk()._auto_tick = False

    def tearDown(self):
        graphsignal.sdk.shutdown()

    def _make_store(self, traces_per_sec=1000.0):
        return SpanStore(config_loader=_config_loader(traces_per_sec=traces_per_sec))

    def test_record_span_proto_fields(self):
        store = self._make_store()
        start_ts = time.time_ns()
        end_ts = start_ts + 1_000_000  # 1 ms

        proto = store.record_span(
            trace_id='abc123',
            span_id='span1',
            name='op1',
            start_ts=start_ts,
            end_ts=end_ts,
            attributes={'k': 'v'},
            tags={'extra': 'e'},
        )

        self.assertIsNotNone(proto)
        self.assertEqual(proto.name, 'op1')
        self.assertEqual(proto.trace_id, 'abc123')
        self.assertEqual(proto.span_id, 'span1')
        self.assertEqual(proto.start_ts, start_ts)
        self.assertEqual(proto.end_ts, end_ts)
        # SDK tags merged in (deployment) plus per-span tags.
        self.assertEqual(find_tag(proto, 'deployment'), 'd1')
        self.assertEqual(find_tag(proto, 'extra'), 'e')
        # Attributes copied through.
        self.assertEqual(find_attribute(proto, 'k'), 'v')
        # span.duration counter added automatically.
        self.assertEqual(find_counter(proto, 'span.duration'), float(end_ts - start_ts))

    def test_record_span_enqueues_for_export(self):
        store = self._make_store()
        proto = store.record_span(
            trace_id='t', span_id='s', name='op', start_ts=1, end_ts=2)
        self.assertIsNotNone(proto)
        self.assertTrue(store.has_unexported())

        exported = store.export()
        self.assertEqual(len(exported), 1)
        self.assertIs(exported[0], proto)
        self.assertFalse(store.has_unexported())

    def test_invalid_inputs_drop_span(self):
        store = self._make_store()
        self.assertIsNone(store.record_span(
            trace_id='', span_id='s', name='op', start_ts=1, end_ts=2))
        self.assertIsNone(store.record_span(
            trace_id='t', span_id='', name='op', start_ts=1, end_ts=2))
        self.assertIsNone(store.record_span(
            trace_id='t', span_id='s', name='', start_ts=1, end_ts=2))
        self.assertIsNone(store.record_span(
            trace_id='t', span_id='s', name='op', start_ts=0, end_ts=2))
        self.assertFalse(store.has_unexported())

    def test_children_inherit_trace_sampling_decision(self):
        store = self._make_store(traces_per_sec=0.001)
        for i in range(10):
            store.record_span(
                trace_id=f'filler-{i}', span_id='s', name='x', start_ts=1, end_ts=2)

        root = store.record_span(
            trace_id='trace-A', span_id='root', name='op1', start_ts=1, end_ts=2)
        self.assertIsNotNone(root)

        # Same trace_id → must inherit "sampled" (children with any name).
        child = store.record_span(
            trace_id='trace-A', span_id='child', parent_span_id='root',
            name='op1', start_ts=3, end_ts=4)
        self.assertIsNotNone(child)

        # Different trace, *same name*, same window → must be skipped because
        # the (name='op1', 'random') sampler's budget for the window is used.
        other = store.record_span(
            trace_id='trace-B', span_id='root2', name='op1', start_ts=5, end_ts=6)
        self.assertIsNone(other)

    def test_initial_trace_exports_without_config(self):
        store = SpanStore(config_loader=None)
        for i in range(10):
            self.assertIsNotNone(store.record_span(
                trace_id=f'trace-{i}', span_id='s', name='op',
                start_ts=1, end_ts=2))
        self.assertIsNone(store.record_span(
            trace_id='trace-10', span_id='s', name='op', start_ts=1, end_ts=2))

    def test_initial_trace_exports_before_zero_rate(self):
        store = self._make_store(traces_per_sec=0)
        for i in range(10):
            self.assertIsNotNone(store.record_span(
                trace_id=f'trace-{i}', span_id='s', name='op',
                start_ts=1, end_ts=2))
        self.assertIsNone(store.record_span(
            trace_id='trace-10', span_id='s', name='op', start_ts=1, end_ts=2))

    def test_events_are_copied(self):
        store = self._make_store()
        proto = store.record_span(
            trace_id='t', span_id='s', name='op', start_ts=1, end_ts=2,
            events=[{
                'name': 'exception',
                'event_ts': 99,
                'attributes': {'exception.type': 'ValueError'},
            }])

        self.assertEqual(len(proto.events), 1)
        evt = proto.events[0]
        self.assertEqual(evt.name, 'exception')
        self.assertEqual(evt.event_ts, 99)
        evt_attrs = {a.name: a.value for a in evt.attributes}
        self.assertEqual(evt_attrs.get('exception.type'), 'ValueError')

    def test_reset_samplers_allows_new_trace(self):
        store = self._make_store(traces_per_sec=0.001)
        for i in range(10):
            store.record_span(
                trace_id=f'filler-{i}', span_id='s', name='op', start_ts=1, end_ts=2)

        self.assertIsNotNone(store.record_span(
            trace_id='trace1', span_id='s1', name='op', start_ts=1, end_ts=2))
        self.assertIsNone(store.record_span(
            trace_id='trace2', span_id='s2', name='op', start_ts=3, end_ts=4))

        store.reset_samplers()
        self.assertIsNotNone(store.record_span(
            trace_id='trace3', span_id='s3', name='op', start_ts=5, end_ts=6))

    def test_per_op_cap_rejects_extra_spans(self):
        store = self._make_store()
        store.MAX_SPANS_PER_OP_PER_TRACE = 2
        trace_id = 'trace-repeat'
        attrs = {'service.name': 'sglang'}

        for i in range(2):
            self.assertIsNotNone(store.record_span(
                trace_id=trace_id,
                span_id=f'df{i}',
                name='sglang.decode_forward',
                start_ts=1 + i,
                end_ts=2 + i,
                attributes=attrs,
            ))
        self.assertIsNone(store.record_span(
            trace_id=trace_id,
            span_id='df2',
            name='sglang.decode_forward',
            start_ts=3,
            end_ts=4,
            attributes=attrs,
        ))

        exported = store.export()
        self.assertEqual(len(exported), 2)

    def test_total_cap_rejects_extra_spans(self):
        store = self._make_store()
        store.MAX_SPANS_PER_TRACE = 2
        trace_id = 'trace-total'
        attrs = {'service.name': 'sglang'}

        self.assertIsNotNone(store.record_span(
            trace_id=trace_id, span_id='s1', name='sglang.tokenize',
            start_ts=1, end_ts=2, attributes=attrs))
        self.assertIsNotNone(store.record_span(
            trace_id=trace_id, span_id='s2', name='sglang.prefill_forward',
            start_ts=3, end_ts=4, attributes=attrs))
        self.assertIsNone(store.record_span(
            trace_id=trace_id, span_id='s3', name='sglang.Req',
            start_ts=5, end_ts=6, attributes=attrs))

        self.assertEqual(len(store.export()), 2)

    def test_evicted_trace_id_purges_queue(self):
        store = SpanStore(config_loader=_config_loader(traces_per_sec=1.0))
        store.MAX_TRACE_STATES = 2
        store._initial_trace_exports_remaining = 0

        with patch.object(TokenBucketSampler, '_now_ms', side_effect=[0, 0, 1000]):
            store.record_span(
                trace_id='trace-a', span_id='s1', name='op', start_ts=1, end_ts=2)
            store.record_span(
                trace_id='trace-b', span_id='s1', name='op', start_ts=1, end_ts=2)
            store.record_span(
                trace_id='trace-c', span_id='s1', name='op', start_ts=1, end_ts=2)

        exported = store.export()
        self.assertEqual(len(exported), 1)
        self.assertEqual(exported[0].trace_id, 'trace-c')


if __name__ == '__main__':
    unittest.main()
