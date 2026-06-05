import re
import unittest

from graphsignal.recorders.cupti_recorder import extract_op_name, make_op_name


class ExtractOpNameTest(unittest.TestCase):
    """Verify the name-transformation pipeline: Itanium unwrap, arch-prefix
    strip, mid-`_sm<digits>_` strip, trailing geometry/dtype/transpose/seq
    strip, and the truncated-raw-name fallback."""

    def test_itanium_mangled_last_identifier(self):
        # Last identifier is the function name; boilerplate `kernel` /
        # `Kernel<N>` gets prefixed with the previous identifier.
        self.assertEqual(
            extract_op_name(
                '_ZN8internal5gemvx6kernelIii13__nv_bfloat16S2_S2_fLb0ELb1ELb1E'
                'Lb0ELi6ELb0E18cublasGemvParamsExIi30cublasGemvTensorStridedBatched'
                'IKS2_ES6_S4_IS2_EfEEENSt9enable_ifIXntT5_EvE4typeET'),
            'gemvx_kernel')
        self.assertEqual(
            extract_op_name(
                '_ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_'
                '15PosEncodingModeE0ELj2ELj4ELj8ELj8ELj1ELj16ENS_16Default'
                'AttentionILb0ELb0ELb0ELb0EEE6ParamsEEvT7_'),
            'BatchDecodeWithPagedKVCacheKernel')
        self.assertEqual(
            extract_op_name(
                '_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_'
                '11FillFunctorIN3c108BFloat16EEESt5arrayIPcLm1EEEEviT0_T1_'),
            'vectorized_elementwise_kernel')
        self.assertEqual(
            extract_op_name(
                '_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm'
                '_bf16_16x16_128x1_tn_align8EEvNT_6ParamsE'),
            'cutlass_Kernel2')
        self.assertEqual(
            extract_op_name(
                '_ZN10flashinfer8sampling30TopKTopPSamplingFromProbKernel'
                'ILj1024ELN3cub39_V_300200_SM_800_860_900_1000_1100_1200'
                '18BlockScanAlgorithmE2ELNS3_20BlockReduceAlgorithmE2ELj4'
                'ELb1EfiEEvPT4_PT5_PfS9_S9_S8_fjmm'),
            'TopKTopPSamplingFromProbKernel')

    def test_arch_prefix_strip(self):
        self.assertEqual(extract_op_name('sm80_xmma_gemm_f16f16_f32_tn'), 'xmma_gemm')
        self.assertEqual(extract_op_name('volta_sgemm_128x64_tn'), 'sgemm')
        self.assertEqual(extract_op_name('ampere_hgemm_64x64_nt'), 'hgemm')

    def test_mid_sm_strip_and_nvjet(self):
        # The full nvjet name walks through every suffix-strip pattern:
        # mid-`_sm121_` collapse, then trailing `_TNNN`, `_bz`, `_tmaAB`,
        # `_32x104x64`, `_2`, `_128x208x64` all peel off.
        self.assertEqual(
            extract_op_name('nvjet_sm121_tst_mma_128x208x64_2_32x104x64_tmaAB_bz_TNNN'),
            'nvjet_tst_mma')

    def test_triton_seq_strip(self):
        self.assertEqual(
            extract_op_name('triton_poi_fused_silu_mul_0'),
            'triton_poi_fused_silu_mul')
        self.assertEqual(
            extract_op_name('triton_per_fused_attn_fwd_0d1d2d'),
            'triton_per_fused_attn_fwd')

    def test_already_short_names_untouched(self):
        # No transform applies — name is already a sensible canonical.
        for name in (
            'flash_attn_fwd_kernel',
            'rms_norm_fwd',
            'layer_norm_grad',
            'silu_and_mul_kernel',
            'awq_gemm_kernel',
        ):
            self.assertEqual(extract_op_name(name), name)

    def test_fallback_to_raw(self):
        # No transform fires — raw name preserved.
        self.assertEqual(extract_op_name('mystery_kernel_xyz'), 'mystery_kernel_xyz')
        # Empty input → empty output (the recorder skips events with no event_name).
        self.assertEqual(extract_op_name(''), '')

    def test_length_cap(self):
        long = 'a' * 100
        out = extract_op_name(long)
        self.assertEqual(len(out), 60)
        self.assertTrue(long.startswith(out))


class MakeOpNameTest(unittest.TestCase):
    """make_op_name appends a stable 4-hex-char fingerprint after `@` so raw
    kernels that collapse to the same family stay distinguishable."""

    _FORMAT_RE = re.compile(r'^(.+)@([0-9a-f]{4})$')

    def test_format_is_family_at_4hex(self):
        out = make_op_name('volta_sgemm_128x64_tn')
        m = self._FORMAT_RE.match(out)
        self.assertIsNotNone(m, f'op_name {out!r} should match <family>@<4 hex>')
        self.assertEqual(m.group(1), 'sgemm')

    def test_family_matches_extract_op_name(self):
        for raw in (
            'volta_sgemm_128x64_tn',
            'sm80_xmma_gemm_f16f16_f32_tn',
            'flash_attn_fwd_kernel',
            '_ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILb1ELi128EEvP',
        ):
            self.assertTrue(make_op_name(raw).startswith(extract_op_name(raw) + '@'),
                            f'make_op_name({raw!r}) should start with extract_op_name+@')

    def test_deterministic(self):
        # Same raw name → same op_name across calls (no global state).
        a = make_op_name('sm80_xmma_gemm_f16f16_f32_tn')
        b = make_op_name('sm80_xmma_gemm_f16f16_f32_tn')
        self.assertEqual(a, b)

    def test_disambiguates_same_family(self):
        # Distinct raw kernels that collapse to the same family must produce
        # distinct fingerprints. Variants chosen because they all reduce to
        # family "xmma_gemm" after the arch prefix / dtype tokens / transpose
        # flags are stripped.
        variants = (
            'sm80_xmma_gemm_f16f16_f32_tn',
            'sm80_xmma_gemm_f16f16_f32_nt',
            'sm80_xmma_gemm_f16f16_f32_nn',
            'sm90_xmma_gemm_f16f16_f32_tn',
        )
        families = {extract_op_name(v) for v in variants}
        self.assertEqual(families, {'xmma_gemm'},
                         f'precondition: all variants should share family "xmma_gemm" — got {families!r}')
        # make_op_name keeps them distinct via the fingerprint.
        op_names = {make_op_name(v) for v in variants}
        self.assertEqual(len(op_names), len(variants),
                         f'expected {len(variants)} distinct op_names, got {op_names!r}')

    def test_empty_input(self):
        self.assertEqual(make_op_name(''), '')


if __name__ == '__main__':
    unittest.main()
