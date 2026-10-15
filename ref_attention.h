// fp32 CPU reference attention and tolerance, shared by test_attention and beam_search_sample.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "attention.h"

namespace attn_ref {

constexpr float kAtol = 1e-2f;
constexpr float kRtol = 1e-2f;

// |got - ref| <= atol + rtol * |ref|; NaN never passes.
inline bool within_tolerance(float got, float ref) {
  return std::fabs(got - ref) <= kAtol + kRtol * std::fabs(ref);
}

// One query token at absolute position p of a sequence whose K/V history (length tokens,
// [length, kNumKvHeads, kHeadDim]) is given. It attends to positions
// [p - window_left, p + window_right] (window_left < 0: from 0), clipped to the history.
// out: [kNumQoHeads, kHeadDim]
inline void reference_attention(const float* q, const float* k_hist, const float* v_hist,
                                int length, int p, int window_left, int window_right,
                                float* out) {
  using attn::kHeadDim;
  using attn::kNumKvHeads;
  using attn::kNumQoHeads;
  const int lo = window_left < 0 ? 0 : std::max(0, p - window_left);
  const int hi = std::min(p + window_right, length - 1);
  const float scale = 1.0f / std::sqrt(float(kHeadDim));
  std::vector<float> logits(hi - lo + 1);
  for (int h = 0; h < kNumQoHeads; ++h) {
    const int kv_h = h / (kNumQoHeads / kNumKvHeads);
    const float* qh = q + h * kHeadDim;
    float m = -INFINITY;
    for (int j = lo; j <= hi; ++j) {
      const float* kj = k_hist + (size_t(j) * kNumKvHeads + kv_h) * kHeadDim;
      float dot = 0.f;
      for (int d = 0; d < kHeadDim; ++d) dot += qh[d] * kj[d];
      logits[j - lo] = dot * scale;
      m = std::max(m, logits[j - lo]);
    }
    float denom = 0.f;
    for (float& l : logits) {
      l = std::exp(l - m);
      denom += l;
    }
    float* oh = out + h * kHeadDim;
    for (int d = 0; d < kHeadDim; ++d) oh[d] = 0.f;
    for (int j = lo; j <= hi; ++j) {
      const float* vj = v_hist + (size_t(j) * kNumKvHeads + kv_h) * kHeadDim;
      const float w = logits[j - lo] / denom;
      for (int d = 0; d < kHeadDim; ++d) oh[d] += w * vj[d];
    }
  }
}

}  // namespace attn_ref
