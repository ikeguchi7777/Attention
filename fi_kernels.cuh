// Internal: thin wrappers around FlashInfer's kernel templates for the fixed configuration
// (bf16, head_dim 64, MHA, no RoPE). Each wrapper lives in its own translation unit so the
// expensive template instantiations compile in parallel and only once.
#pragma once

#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/scheduler.cuh>

#include "attention.h"

namespace attn::detail {

using PrefillParams = flashinfer::BatchPrefillPagedParams<bf16, bf16, bf16, int32_t>;
using DecodeParams = flashinfer::BatchDecodeParams<bf16, bf16, bf16, int32_t>;

static_assert(kNumQoHeads % kNumKvHeads == 0, "query heads must be a multiple of KV heads");
constexpr uint32_t kGroupSize = kNumQoHeads / kNumKvHeads;

// Causal prefill over the paged cache. cta_tile_q comes from PrefillPlanInfo.
// Defined in fi_prefill_nowindow.cu / fi_prefill_window.cu.
template <bool USE_SLIDING_WINDOW>
cudaError_t LaunchPrefill(PrefillParams params, bf16* tmp_v, float* tmp_s, uint32_t cta_tile_q,
                          cudaStream_t stream);

// FlashInfer's DecodePlan needs the kernel itself (it sizes the grid from occupancy), so it lives
// next to the decode instantiation in fi_decode.cu.
cudaError_t PlanDecode(void* float_buffer, size_t float_bytes, void* int_buffer,
                       void* pinned_int_buffer, size_t int_bytes,
                       flashinfer::DecodePlanInfo& plan_info, int32_t* kv_indptr_host,
                       uint32_t batch_size, uint32_t page_size, bool use_sliding_window,
                       cudaStream_t stream);

cudaError_t LaunchDecode(DecodeParams params, bf16* tmp_v, float* tmp_s, bool use_sliding_window,
                         cudaStream_t stream);

}  // namespace attn::detail
