import unittest

from graphsignal.otel.span_op_name import stable_otel_op_name


class StableOtelOpNameTest(unittest.TestCase):
    def test_sglang_stage_unchanged(self):
        self.assertEqual(
            stable_otel_op_name('sglang.decode_forward', {'service.name': 'sglang'}),
            'sglang.decode_forward')

    def test_sglang_unified_request_by_pattern(self):
        self.assertEqual(
            stable_otel_op_name(
                'sglang. Req 571f196b',
                {'service.name': 'sglang'}),
            'sglang.Req')

    def test_sglang_pd_request_by_pattern(self):
        self.assertEqual(
            stable_otel_op_name(
                'sglang.prefill Req abc12345',
                {'service.name': 'sglang'}),
            'sglang.prefill.Req')

    def test_sglang_thread_by_pattern(self):
        self.assertEqual(
            stable_otel_op_name(
                'sglang.Scheduler [TP 0] (host:09c2b7e1 | pid:14363)',
                {'service.name': 'sglang'}),
            'sglang.Scheduler [TP 0]')

    def test_sglang_thread_by_thread_label(self):
        self.assertEqual(
            stable_otel_op_name(
                'sglang.Scheduler [TP 0] (host:09c2b7e1 | pid:14363)',
                {
                    'service.name': 'sglang',
                    'thread_label': 'Scheduler',
                    'tp_rank': 0,
                }),
            'sglang.Scheduler [TP 0]')

    def test_sglang_thread_label_with_pp_dp_ranks(self):
        self.assertEqual(
            stable_otel_op_name(
                'sglang.Scheduler (host:abc | pid:1)',
                {
                    'service.name': 'sglang',
                    'thread_label': 'Scheduler',
                    'tp_rank': 0,
                    'pp_rank': 1,
                    'dp_rank': 2,
                }),
            'sglang.Scheduler [TP 0] [PP 1] [DP 2]')

    def test_sglang_diffusion_prefix(self):
        self.assertEqual(
            stable_otel_op_name(
                'sglang-diffusion.gpu_forward',
                {'service.name': 'sglang-diffusion'}),
            'sglang-diffusion.gpu_forward')

    def test_sglang_module_attr_without_service_name(self):
        self.assertEqual(
            stable_otel_op_name(
                ' Req deadbeef',
                {'module': 'sglang::request'}),
            'Req')

    def test_vllm_llm_request_unchanged(self):
        self.assertEqual(
            stable_otel_op_name(
                'vllm-server.llm_request',
                {
                    'service.name': 'vllm-server',
                    'gen_ai.request.id': 'req-uuid',
                }),
            'vllm-server.llm_request')

    def test_vllm_loading_span_unchanged(self):
        self.assertEqual(
            stable_otel_op_name(
                'Loading (GPU)',
                {'code.function': 'GPUModelRunner.load_model'}),
            'Loading (GPU)')

    def test_trtllm_llm_request_unchanged(self):
        self.assertEqual(
            stable_otel_op_name(
                'trtllm-server.llm_request',
                {
                    'service.name': 'trtllm-server',
                    'gen_ai.request.id': 'abc',
                }),
            'trtllm-server.llm_request')

    def test_non_sglang_request_suffix_unchanged(self):
        self.assertEqual(
            stable_otel_op_name('custom. Req abc12345', {}),
            'custom. Req abc12345')

    def test_empty_name(self):
        self.assertEqual(stable_otel_op_name('', {}), '')


if __name__ == '__main__':
    unittest.main()
