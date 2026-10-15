// Internal: body of LaunchPrefill<USE_SLIDING_WINDOW>, included by fi_prefill_{no,}window.cu.
#pragma once

#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/variants.cuh>

#include "fi_kernels.cuh"

namespace attn::detail {

template <bool USE_SLIDING_WINDOW>
cudaError_t LaunchPrefill(PrefillParams params, bf16* tmp_v, float* tmp_s, uint32_t cta_tile_q,
                          cudaStream_t stream) {
  using Variant = flashinfer::DefaultAttention</*use_custom_mask=*/false, USE_SLIDING_WINDOW,
                                               /*use_logits_soft_cap=*/false, /*use_alibi=*/false>;
  // For head_dim 64 the scheduler (FA2DetermineCtaTileQ) only picks 16, 64 or 128.
  switch (cta_tile_q) {
#define ATTN_PREFILL_CASE(TILE)                                                              \
  case TILE:                                                                                 \
    return flashinfer::BatchPrefillWithPagedKVCacheDispatched<                               \
        /*SAME_KV_STRIDES=*/true, TILE, kHeadDim, kHeadDim, flashinfer::PosEncodingMode::kNone, \
        /*USE_FP16_QK_REDUCTION=*/false, flashinfer::MaskMode::kCausal, Variant, PrefillParams>( \
        params, tmp_v, tmp_s, /*enable_pdl=*/false, stream);
    ATTN_PREFILL_CASE(16)
    ATTN_PREFILL_CASE(64)
    ATTN_PREFILL_CASE(128)
#undef ATTN_PREFILL_CASE
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace attn::detail
