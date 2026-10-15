// Explicit instantiation of LaunchPrefill<false> (see fi_prefill.cuh).
#include "fi_prefill.cuh"

namespace attn::detail {
template cudaError_t LaunchPrefill<false>(PrefillParams, bf16*, float*, uint32_t, cudaStream_t);
}  // namespace attn::detail
