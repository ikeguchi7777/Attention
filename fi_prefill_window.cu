// Explicit instantiation of LaunchPrefill<true> (see fi_prefill.cuh).
#include "fi_prefill.cuh"

namespace attn::detail {
template cudaError_t LaunchPrefill<true>(PrefillParams, bf16*, float*, uint32_t, cudaStream_t);
}  // namespace attn::detail
