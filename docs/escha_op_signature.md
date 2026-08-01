# Escha `escham_reconstruct` — signature + linearity audit

## 1. Introspection

escha module: `/usr/local/lib/python3.12/site-packages/escha/__init__.py`
escha._C: `<module 'escha._C' from '/usr/local/lib/python3.12/site-packages/escha/_C.cpython-312-x86_64-linux-gnu.so'>`

### `dir(escha._C)`
  - `escha_aqlm_auto` ? — escha_aqlm_auto(ta: torch.Tensor, codes: torch.Tensor, codebooks: torch.Tensor, scales: torch.Tensor, codes_T: torch.Tensor | None = None, k_tile: typing.SupportsInt = 64) -> torch.Tensor
  - `escha_aqlm_fused_hmma` ? — escha_aqlm_fused_hmma(ta: torch.Tensor, codes_T: torch.Tensor, codebooks: torch.Tensor, scales: torch.Tensor, k_tile: typing.SupportsInt = 64) -> torch.Tensor
  - `escha_aqlm_gemv` ? — escha_aqlm_gemv(ta: torch.Tensor, codes: torch.Tensor, codebooks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor
  - `escha_aqlm_prepare_codes_transposed` ? — escha_aqlm_prepare_codes_transposed(codes: torch.Tensor) -> torch.Tensor
  - `escha_binary_gemv` ? — escha_binary_gemv(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: torch.Tensor) -> torch.Tensor
  - `escha_binary_gemv_reg` ? — escha_binary_gemv_reg(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: torch.Tensor) -> torch.Tensor
  - `escha_decgemv` ? — escha_decgemv(x: torch.Tensor, packed_codes: torch.Tensor, scale: torch.Tensor, transform_left: torch.Tensor, transform_right: torch.Tensor, a1: torch.Tensor, a2: torch.Tensor, td1: typing.SupportsInt, td2: typing.SupportsInt, ic: typing.SupportsInt, expic: typing.SupportsInt, gain: torch.Tensor | None = None, block_size: typing.SupportsInt = 0) -> torch.Tensor
  - `escha_decgemv_inplace` ? — escha_decgemv_inplace(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: torch.Tensor, arg4: torch.Tensor, arg5: torch.Tensor, arg6: torch.Tensor, arg7: typing.SupportsInt, arg8: typing.SupportsInt, arg9: typing.SupportsInt, arg10: typing.SupportsInt, arg11: torch.Tensor, arg12: torch.Tensor, arg13: torch.Tensor, arg14: torch.Tensor) -> None
  - `escha_decgemv_reg` ? — escha_decgemv_reg(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: torch.Tensor, arg4: torch.Tensor, arg5: torch.Tensor, arg6: torch.Tensor, arg7: typing.SupportsInt, arg8: typing.SupportsInt, arg9: typing.SupportsInt, arg10: typing.SupportsInt) -> torch.Tensor
  - `escha_dequant` ? — escha_dequant(packed_codes: torch.Tensor, scale: torch.Tensor, transform_left: torch.Tensor, transform_right: torch.Tensor, a1: torch.Tensor, a2: torch.Tensor, td1: typing.SupportsInt, td2: typing.SupportsInt, ic: typing.SupportsInt, expic: typing.SupportsInt, gain: torch.Tensor | None = None, block_size: typing.SupportsInt = 0) -> torch.Tensor
  - `escha_fused_dequant_gemm` ? — escha_fused_dequant_gemm(ta_fp16: torch.Tensor, packed_codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor
  - `escha_fused_dequant_gemm_auto` ? — escha_fused_dequant_gemm_auto(ta_fp16: torch.Tensor, packed_codes: torch.Tensor, scale: torch.Tensor, packed_T: torch.Tensor | None = None) -> torch.Tensor
  - `escha_fused_dequant_gemm_v2` ? — escha_fused_dequant_gemm_v2(ta_fp16: torch.Tensor, packed_codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor
  - `escha_fused_dequant_gemm_v3` ? — escha_fused_dequant_gemm_v3(ta_fp16: torch.Tensor, packed_T: torch.Tensor, scale: torch.Tensor) -> torch.Tensor
  - `escha_fused_dequant_gemm_v4` ? — escha_fused_dequant_gemm_v4(ta_fp16: torch.Tensor, packed_T: torch.Tensor, scale: torch.Tensor) -> torch.Tensor
  - `escha_fused_dequant_gemm_v5` ? — escha_fused_dequant_gemm_v5(ta_fp16: torch.Tensor, packed_codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor
  - `escha_init` ? — escha_init() -> None
  - `escha_lut_binary_gemv` ? — escha_lut_binary_gemv(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: torch.Tensor) -> torch.Tensor
  - `escha_prepare_packed_transposed` ? — escha_prepare_packed_transposed(packed_codes: torch.Tensor) -> torch.Tensor
  - `escha_transform` ? — escha_transform(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: torch.Tensor, arg4: torch.Tensor, arg5: typing.SupportsInt, arg6: typing.SupportsInt, arg7: typing.SupportsInt, arg8: typing.SupportsInt) -> list[torch.Tensor]
  - `escha_transform_fast` ? — escha_transform_fast(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: torch.Tensor, arg4: torch.Tensor, arg5: typing.SupportsInt, arg6: typing.SupportsInt, arg7: typing.SupportsInt, arg8: typing.SupportsInt) -> list[torch.Tensor]
  - `escha_transform_fp16` ? — escha_transform_fp16(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: torch.Tensor, arg4: torch.Tensor, arg5: typing.SupportsInt, arg6: typing.SupportsInt, arg7: typing.SupportsInt, arg8: typing.SupportsInt) -> list[torch.Tensor]
  - `escha_transform_fp16_no_bias` ? — escha_transform_fp16_no_bias(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: torch.Tensor, arg4: torch.Tensor, arg5: typing.SupportsInt, arg6: typing.SupportsInt, arg7: typing.SupportsInt, arg8: typing.SupportsInt) -> torch.Tensor
  - `eschax_binary_search` ? — eschax_binary_search(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: typing.SupportsInt, arg4: typing.SupportsInt, arg5: typing.SupportsInt) -> torch.Tensor
  - `eschax_dequant` ? — eschax_dequant(arg0: torch.Tensor, arg1: typing.SupportsFloat) -> torch.Tensor
  - `eschax_eschax_decode` ? — eschax_eschax_decode(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: torch.Tensor, arg4: typing.SupportsInt, arg5: typing.SupportsInt) -> torch.Tensor
  - `eschax_eschax_gemv` ? — eschax_eschax_gemv(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: torch.Tensor, arg4: torch.Tensor, arg5: typing.SupportsFloat, arg6: torch.Tensor, arg7: typing.SupportsInt, arg8: typing.SupportsInt) -> torch.Tensor
  - `eschax_gemv` ? — eschax_gemv(arg0: torch.Tensor, arg1: torch.Tensor, arg2: typing.SupportsFloat, arg3: torch.Tensor) -> torch.Tensor
  - `eschax_huffman_decode` ? — eschax_huffman_decode(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: typing.SupportsInt, arg4: typing.SupportsInt) -> torch.Tensor
  - `eschax_huffman_gemv` ? — eschax_huffman_gemv(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: torch.Tensor, arg4: typing.SupportsFloat, arg5: torch.Tensor, arg6: typing.SupportsInt, arg7: typing.SupportsInt) -> torch.Tensor

### torch.ops.escha.escham_reconstruct
  escha.escham_reconstruct
  overloads: ['default']
  default: escha::escham_reconstruct(Tensor packed, int in_features, int out_features, int K, bool cbA, bool mul1) -> Tensor

## 2. Shape acceptance test

  min-K2 128x128: cshape=(8, 8, 32) in_f=128 out_f=128 K=2 -> OK, w.shape=(128, 128) dtype=torch.float16
  min-K3 128x128: cshape=(8, 8, 48) in_f=128 out_f=128 K=3 -> OK, w.shape=(128, 128) dtype=torch.float16
  escha gate_up (K=2, in=2048/out=1024): cshape=(128, 64, 32) in_f=2048 out_f=1024 K=2 -> OK, w.shape=(2048, 1024) dtype=torch.float16
  escha down (K=3, in=512/out=2048): cshape=(32, 128, 48) in_f=512 out_f=2048 K=3 -> OK, w.shape=(512, 2048) dtype=torch.float16
  leading batch (2, 8, 8, 32) K=2: cshape=(2, 8, 8, 32) in_f=128 out_f=128 K=2 -> OK, w.shape=(128, 128) dtype=torch.float16
  leading batch (16, 8, 8, 32) K=2: cshape=(16, 8, 8, 32) in_f=128 out_f=128 K=2 -> OK, w.shape=(128, 128) dtype=torch.float16
  leading batch (4, 128, 64, 32) K=2: cshape=(4, 128, 64, 32) in_f=2048 out_f=1024 K=2 -> OK, w.shape=(2048, 1024) dtype=torch.float16

## 3. Full delta pattern at slot (0,0,0)

For (in=2048, out=1024, K=2) tile: what is the full (row, col) support
of the delta when we set exactly code[0,0,0] = v, for various v?

  v=     1:   8 nonzero positions, rows=[4, 5, 11, 12, 13], cols=[0, 8]
  v=     2:   8 nonzero positions, rows=[4, 5, 11, 12, 13], cols=[0, 8]
  v=     3:   8 nonzero positions, rows=[4, 5, 11, 12, 13], cols=[0, 8]
  v=     4:   8 nonzero positions, rows=[4, 5, 10, 11, 12, 13], cols=[0, 8]
  v=     5:   9 nonzero positions, rows=[4, 5, 10, 11, 12, 13], cols=[0, 8]
  v=     7:   9 nonzero positions, rows=[4, 5, 10, 11, 12, 13], cols=[0, 8]
  v=    10:   9 nonzero positions, rows=[4, 5, 10, 11, 12, 13], cols=[0, 8]
  v=    16:   8 nonzero positions, rows=[3, 4, 5, 10, 11, 12, 13], cols=[0, 8]
  v=    64:   8 nonzero positions, rows=[2, 3, 4, 5, 10, 11, 12, 13], cols=[0, 8]
  v=   256:   8 nonzero positions, rows=[2, 3, 4, 5, 10, 11, 12], cols=[0, 8]
  v=  1024:   8 nonzero positions, rows=[2, 3, 4, 5, 10, 11], cols=[0, 8]
  v=  4096:   8 nonzero positions, rows=[2, 3, 4, 10, 11], cols=[0, 8]
  v= 16384:   8 nonzero positions, rows=[2, 3, 10, 11], cols=[0, 8]
  v= 32767:  15 nonzero positions, rows=[2, 3, 4, 5, 10, 11, 12, 13], cols=[0, 8]
  v=    -1:  15 nonzero positions, rows=[2, 3, 4, 5, 10, 11, 12, 13], cols=[0, 8]
  v=  -100:  14 nonzero positions, rows=[2, 3, 4, 5, 10, 11, 12, 13], cols=[0, 8]
  v=-32768:   8 nonzero positions, rows=[2, 3, 10, 11], cols=[0, 8]

## 4. Superposition test (LINEARITY in codes)

Test: op(all-zeros with code[bi_a, bj_a, k_a]=v_a AND code[bi_b, bj_b, k_b]=v_b)
      == op(only code[bi_a, bj_a, k_a]=v_a) + op(only code[bi_b, bj_b, k_b]=v_b) - op(zeros)
If yes, we can probe many (bi, bj) slots simultaneously in ONE op call.

  2-pos, distinct (bi,bj), same k: |combined|=1.034e+01 |diff|=0.000e+00 rel=0.000e+00
  2-pos, same (bi,bj), diff k: |combined|=1.034e+01 |diff|=0.000e+00 rel=0.000e+00
  2-pos, same (bi,bj), diff K-slice: |combined|=1.034e+01 |diff|=0.000e+00 rel=0.000e+00
  8-pos random: |combined|=2.231e+01 |diff|=0.000e+00 rel=0.000e+00
  100-pos random: |combined|=8.165e+01 |diff|=0.000e+00 rel=0.000e+00

## 5. Slot invariance test — is the codebook shared across (bi, bj)?

Compare delta patterns for the SAME value v at DIFFERENT (bi, bj) with the same k_slot.
If they are identical up to a (bi*16, bj*16) offset, the codebook is (bi, bj)-invariant.

  v=     1 (bi=0,bj=0) vs (bi=1,bj=0): |diff|=0.000e+00 rel=0.000e+00
  v=     1 (bi=0,bj=0) vs (bi=0,bj=1): |diff|=0.000e+00 rel=0.000e+00
  v=     1 (bi=0,bj=0) vs (bi=1,bj=1): |diff|=0.000e+00 rel=0.000e+00
  v=     1 (bi=0,bj=0) vs (bi=5,bj=3): |diff|=0.000e+00 rel=0.000e+00
  v=     1 (bi=0,bj=0) vs (bi=127,bj=63): |diff|=0.000e+00 rel=0.000e+00
  v=   100 (bi=0,bj=0) vs (bi=1,bj=0): |diff|=0.000e+00 rel=0.000e+00
  v=   100 (bi=0,bj=0) vs (bi=0,bj=1): |diff|=0.000e+00 rel=0.000e+00
  v=   100 (bi=0,bj=0) vs (bi=1,bj=1): |diff|=0.000e+00 rel=0.000e+00
  v=   100 (bi=0,bj=0) vs (bi=5,bj=3): |diff|=0.000e+00 rel=0.000e+00
  v=   100 (bi=0,bj=0) vs (bi=127,bj=63): |diff|=0.000e+00 rel=0.000e+00
  v= 32767 (bi=0,bj=0) vs (bi=1,bj=0): |diff|=0.000e+00 rel=0.000e+00
  v= 32767 (bi=0,bj=0) vs (bi=0,bj=1): |diff|=0.000e+00 rel=0.000e+00
  v= 32767 (bi=0,bj=0) vs (bi=1,bj=1): |diff|=0.000e+00 rel=0.000e+00
  v= 32767 (bi=0,bj=0) vs (bi=5,bj=3): |diff|=0.000e+00 rel=0.000e+00
  v= 32767 (bi=0,bj=0) vs (bi=127,bj=63): |diff|=0.000e+00 rel=0.000e+00
  v=-32768 (bi=0,bj=0) vs (bi=1,bj=0): |diff|=0.000e+00 rel=0.000e+00
  v=-32768 (bi=0,bj=0) vs (bi=0,bj=1): |diff|=0.000e+00 rel=0.000e+00
  v=-32768 (bi=0,bj=0) vs (bi=1,bj=1): |diff|=0.000e+00 rel=0.000e+00
  v=-32768 (bi=0,bj=0) vs (bi=5,bj=3): |diff|=0.000e+00 rel=0.000e+00
  v=-32768 (bi=0,bj=0) vs (bi=127,bj=63): |diff|=0.000e+00 rel=0.000e+00

## 6. k_slot pattern

For each k_slot, what is the row/col support of the (0, 0, k)+v=1 delta?

### K=2, cshape=(128, 64, 32)
  k= 0: rows=[4, 5, 11, 12, 13] cols=[0, 8] n_pos=8
  k= 1: rows=[2, 3, 9, 10, 11] cols=[0, 8] n_pos=8
  k= 2: rows=[0, 1, 8, 9, 15] cols=[1, 8, 9] n_pos=8
  k= 3: rows=[6, 7, 13, 14, 15] cols=[0, 8] n_pos=8
  k= 4: rows=[4, 5, 11, 12, 13] cols=[1, 9] n_pos=8
  k= 5: rows=[2, 3, 9, 10, 11] cols=[1, 9] n_pos=8
  k= 6: rows=[0, 1, 8, 9, 15] cols=[2, 9, 10] n_pos=8
  k= 7: rows=[6, 7, 13, 14, 15] cols=[1, 9] n_pos=8
  k= 8: rows=[4, 5, 11, 12, 13] cols=[2, 10] n_pos=8
  k= 9: rows=[2, 3, 9, 10, 11] cols=[2, 10] n_pos=8
  k=10: rows=[0, 1, 8, 9, 15] cols=[3, 10, 11] n_pos=8
  k=11: rows=[6, 7, 13, 14, 15] cols=[2, 10] n_pos=8
  k=12: rows=[4, 5, 11, 12, 13] cols=[3, 11] n_pos=8
  k=13: rows=[2, 3, 9, 10, 11] cols=[3, 11] n_pos=8
  k=14: rows=[0, 1, 8, 9, 15] cols=[4, 11, 12] n_pos=8
  k=15: rows=[6, 7, 13, 14, 15] cols=[3, 11] n_pos=8
  k=16: rows=[4, 5, 11, 12, 13] cols=[4, 12] n_pos=8
  k=17: rows=[2, 3, 9, 10, 11] cols=[4, 12] n_pos=8
  k=18: rows=[0, 1, 8, 9, 15] cols=[5, 12, 13] n_pos=8
  k=19: rows=[6, 7, 13, 14, 15] cols=[4, 12] n_pos=8
  k=20: rows=[4, 5, 11, 12, 13] cols=[5, 13] n_pos=8
  k=21: rows=[2, 3, 9, 10, 11] cols=[5, 13] n_pos=8
  k=22: rows=[0, 1, 8, 9, 15] cols=[6, 13, 14] n_pos=8
  k=23: rows=[6, 7, 13, 14, 15] cols=[5, 13] n_pos=8
  k=24: rows=[4, 5, 11, 12, 13] cols=[6, 14] n_pos=8
  k=25: rows=[2, 3, 9, 10, 11] cols=[6, 14] n_pos=8
  k=26: rows=[0, 1, 8, 9, 15] cols=[7, 14, 15] n_pos=8
  k=27: rows=[6, 7, 13, 14, 15] cols=[6, 14] n_pos=8
  k=28: rows=[4, 5, 11, 12, 13] cols=[7, 15] n_pos=8
  k=29: rows=[2, 3, 9, 10, 11] cols=[7, 15] n_pos=8
  k=30: rows=[0, 1, 8, 9, 15] cols=[0, 8, 15] n_pos=8
  k=31: rows=[6, 7, 13, 14, 15] cols=[7, 15] n_pos=8
### K=3, cshape=(32, 128, 48)
  k= 0: rows=[2, 3, 10, 11] cols=[0, 8] n_pos=5
  k= 1: rows=[1, 2, 3, 8, 9] cols=[0, 8] n_pos=5
  k= 2: rows=[5, 6, 7, 12, 13] cols=[0, 8] n_pos=5
  k= 3: rows=[4, 5, 11, 12, 13] cols=[0, 8] n_pos=6
  k= 4: rows=[0, 1, 8, 9, 15] cols=[1, 8, 9] n_pos=6
  k= 5: rows=[6, 7, 14, 15] cols=[0, 8] n_pos=5
  k= 6: rows=[2, 3, 10, 11] cols=[1, 9] n_pos=5
  k= 7: rows=[1, 2, 3, 8, 9] cols=[1, 9] n_pos=5
  k= 8: rows=[5, 6, 7, 12, 13] cols=[1, 9] n_pos=5
  k= 9: rows=[4, 5, 11, 12, 13] cols=[1, 9] n_pos=6
  k=10: rows=[0, 1, 8, 9, 15] cols=[2, 9, 10] n_pos=6
  k=11: rows=[6, 7, 14, 15] cols=[1, 9] n_pos=5
  k=12: rows=[2, 3, 10, 11] cols=[2, 10] n_pos=5
  k=13: rows=[1, 2, 3, 8, 9] cols=[2, 10] n_pos=5
  k=14: rows=[5, 6, 7, 12, 13] cols=[2, 10] n_pos=5
  k=15: rows=[4, 5, 11, 12, 13] cols=[2, 10] n_pos=6
  k=16: rows=[0, 1, 8, 9, 15] cols=[3, 10, 11] n_pos=6
  k=17: rows=[6, 7, 14, 15] cols=[2, 10] n_pos=5
  k=18: rows=[2, 3, 10, 11] cols=[3, 11] n_pos=5
  k=19: rows=[1, 2, 3, 8, 9] cols=[3, 11] n_pos=5
  k=20: rows=[5, 6, 7, 12, 13] cols=[3, 11] n_pos=5
  k=21: rows=[4, 5, 11, 12, 13] cols=[3, 11] n_pos=6
  k=22: rows=[0, 1, 8, 9, 15] cols=[4, 11, 12] n_pos=6
  k=23: rows=[6, 7, 14, 15] cols=[3, 11] n_pos=5
  k=24: rows=[2, 3, 10, 11] cols=[4, 12] n_pos=5
  k=25: rows=[1, 2, 3, 8, 9] cols=[4, 12] n_pos=5
  k=26: rows=[5, 6, 7, 12, 13] cols=[4, 12] n_pos=5
  k=27: rows=[4, 5, 11, 12, 13] cols=[4, 12] n_pos=6
  k=28: rows=[0, 1, 8, 9, 15] cols=[5, 12, 13] n_pos=6
  k=29: rows=[6, 7, 14, 15] cols=[4, 12] n_pos=5
  k=30: rows=[2, 3, 10, 11] cols=[5, 13] n_pos=5
  k=31: rows=[1, 2, 3, 8, 9] cols=[5, 13] n_pos=5
  k=32: rows=[5, 6, 7, 12, 13] cols=[5, 13] n_pos=5
  k=33: rows=[4, 5, 11, 12, 13] cols=[5, 13] n_pos=6
  k=34: rows=[0, 1, 8, 9, 15] cols=[6, 13, 14] n_pos=6
  k=35: rows=[6, 7, 14, 15] cols=[5, 13] n_pos=5
  k=36: rows=[2, 3, 10, 11] cols=[6, 14] n_pos=5
  k=37: rows=[1, 2, 3, 8, 9] cols=[6, 14] n_pos=5
  k=38: rows=[5, 6, 7, 12, 13] cols=[6, 14] n_pos=5
  k=39: rows=[4, 5, 11, 12, 13] cols=[6, 14] n_pos=6
  k=40: rows=[0, 1, 8, 9, 15] cols=[7, 14, 15] n_pos=6
  k=41: rows=[6, 7, 14, 15] cols=[6, 14] n_pos=5
  k=42: rows=[2, 3, 10, 11] cols=[7, 15] n_pos=5
  k=43: rows=[1, 2, 3, 8, 9] cols=[7, 15] n_pos=5
  k=44: rows=[5, 6, 7, 12, 13] cols=[7, 15] n_pos=5
  k=45: rows=[4, 5, 11, 12, 13] cols=[7, 15] n_pos=6
  k=46: rows=[0, 1, 8, 9, 15] cols=[0, 8, 15] n_pos=6
  k=47: rows=[6, 7, 14, 15] cols=[7, 15] n_pos=5
