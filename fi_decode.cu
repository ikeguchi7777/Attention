// Decode plan + launch (see fi_kernels.cuh).
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/variants.cuh>

#include "fi_kernels.cuh"

namespace attn::detail {
namespace {

template <bool USE_SLIDING_WINDOW>
using DecodeVariant = flashinfer::DefaultAttention</*use_custom_mask=*/false, USE_SLIDING_WINDOW,
                                                   /*use_logits_soft_cap=*/false,
                                                   /*use_alibi=*/false>;

template <bool USE_SLIDING_WINDOW>
cudaError_t PlanDecodeImpl(void* float_buffer, size_t float_bytes, void* int_buffer,
                           void* pinned_int_buffer, size_t int_bytes,
                           flashinfer::DecodePlanInfo& plan_info, int32_t* kv_indptr_host,
                           uint32_t batch_size, uint32_t page_size, cudaStream_t stream) {
  using Variant = DecodeVariant<USE_SLIDING_WINDOW>;
  auto work_estimation =
      flashinfer::BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
          kGroupSize, kHeadDim, flashinfer::PosEncodingMode::kNone, Variant, DecodeParams>;
  return flashinfer::DecodePlan<kHeadDim, flashinfer::PosEncodingMode::kNone, Variant,
                                DecodeParams>(
      float_buffer, float_bytes, int_buffer, pinned_int_buffer, int_bytes, plan_info,
      kv_indptr_host, batch_size, kNumQoHeads, page_size, /*enable_cuda_graph=*/false, stream,
      work_estimation);
}

template <bool USE_SLIDING_WINDOW>
cudaError_t LaunchDecodeImpl(DecodeParams params, bf16* tmp_v, float* tmp_s, cudaStream_t stream) {
  return flashinfer::BatchDecodeWithPagedKVCacheDispatched<
      kHeadDim, flashinfer::PosEncodingMode::kNone, DecodeVariant<USE_SLIDING_WINDOW>,
      DecodeParams>(params, tmp_v, tmp_s, /*enable_pdl=*/false, stream);
}

}  // namespace

cudaError_t PlanDecode(void* float_buffer, size_t float_bytes, void* int_buffer,
                       void* pinned_int_buffer, size_t int_bytes,
                       flashinfer::DecodePlanInfo& plan_info, int32_t* kv_indptr_host,
                       uint32_t batch_size, uint32_t page_size, bool use_sliding_window,
                       cudaStream_t stream) {
  if (use_sliding_window) {
    return PlanDecodeImpl<true>(float_buffer, float_bytes, int_buffer, pinned_int_buffer,
                                int_bytes, plan_info, kv_indptr_host, batch_size, page_size,
                                stream);
  }
  return PlanDecodeImpl<false>(float_buffer, float_bytes, int_buffer, pinned_int_buffer, int_bytes,
                               plan_info, kv_indptr_host, batch_size, page_size, stream);
}

cudaError_t LaunchDecode(DecodeParams params, bf16* tmp_v, float* tmp_s, bool use_sliding_window,
                         cudaStream_t stream) {
  return use_sliding_window ? LaunchDecodeImpl<true>(params, tmp_v, tmp_s, stream)
                            : LaunchDecodeImpl<false>(params, tmp_v, tmp_s, stream);
}

}  // namespace attn::detail
