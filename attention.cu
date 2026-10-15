#include "attention.h"

#include <flashinfer/page.cuh>

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_set>

#include "fi_kernels.cuh"

namespace attn {
namespace {

void require(bool cond, const std::string& msg) {
  if (!cond) throw std::invalid_argument(msg);
}

constexpr size_t kTokenElems = size_t(kNumKvHeads) * kHeadDim;  // elements per token per pool

}  // namespace

// ------------------------------------------------------------------------------------------------
// AttentionWorkspace

struct AttentionWorkspace::Impl {
  // FlashInfer plan workspaces.
  void* float_workspace = nullptr;
  void* int_workspace = nullptr;
  void* pinned_int_workspace = nullptr;  // host mirror the plan fills, then copies to int_workspace
  size_t float_workspace_bytes = 0;
  size_t int_workspace_bytes = 0;

  // Per-step int32 index arrays, staged in pinned memory and copied to the device in one go.
  int32_t* index_device = nullptr;
  int32_t* index_pinned = nullptr;
  size_t index_capacity = 0;

  // Recorded after a step's host-to-device copies; the next plan waits on it before it rewrites
  // the pinned buffers those copies read from.
  cudaEvent_t copies_done = nullptr;
  bool copies_pending = false;

  // The most recent plan.
  enum class Kind { kNone, kPrefill, kDecode } kind = Kind::kNone;
  PagedKVCache* cache = nullptr;
  uint64_t cache_version = 0;
  cudaStream_t stream = nullptr;
  int window_left = kNoWindow;  // as passed to FlashInfer (see flashinfer_window)
  uint32_t batch_size = 0;
  uint32_t num_q = 0;   // rows of q/out
  uint32_t num_kv = 0;  // rows of k/v
  flashinfer::PrefillPlanInfo prefill_info;
  flashinfer::DecodePlanInfo decode_info;
  // Device views into index_device.
  int32_t* kv_indptr = nullptr;
  int32_t* kv_indices = nullptr;
  int32_t* kv_last_page_len = nullptr;
  int32_t* qo_indptr = nullptr;
  int32_t* batch_indices = nullptr;
  int32_t* positions = nullptr;

  ~Impl() {
    if (copies_pending) cudaEventSynchronize(copies_done);
    if (copies_done) cudaEventDestroy(copies_done);
    cudaFree(float_workspace);
    cudaFree(int_workspace);
    cudaFreeHost(pinned_int_workspace);
    cudaFree(index_device);
    cudaFreeHost(index_pinned);
  }

  void wait_previous_copies() {
    if (copies_pending) {
      CUDA_CHECK(cudaEventSynchronize(copies_done));
      copies_pending = false;
    }
  }

  void mark_copies_issued(cudaStream_t s) {
    CUDA_CHECK(cudaEventRecord(copies_done, s));
    copies_pending = true;
  }

  // Must be called after wait_previous_copies(). Growing frees buffers an in-flight kernel may
  // still read, so it drains the device first; it only happens when a step is the largest so far.
  void reserve_index_buffers(size_t count) {
    if (count <= index_capacity) return;
    const size_t new_capacity = std::max(count, index_capacity * 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(index_device));
    CUDA_CHECK(cudaFreeHost(index_pinned));
    index_device = nullptr;
    index_pinned = nullptr;
    index_capacity = 0;
    CUDA_CHECK(cudaMalloc(&index_device, new_capacity * sizeof(int32_t)));
    CUDA_CHECK(cudaMallocHost(&index_pinned, new_capacity * sizeof(int32_t)));
    index_capacity = new_capacity;
  }
};

AttentionWorkspace::AttentionWorkspace(size_t float_workspace_bytes, size_t int_workspace_bytes)
    : impl_(std::make_unique<Impl>()) {
  Impl& w = *impl_;
  w.float_workspace_bytes = float_workspace_bytes;
  w.int_workspace_bytes = int_workspace_bytes;
  CUDA_CHECK(cudaMalloc(&w.float_workspace, float_workspace_bytes));
  CUDA_CHECK(cudaMalloc(&w.int_workspace, int_workspace_bytes));
  CUDA_CHECK(cudaMallocHost(&w.pinned_int_workspace, int_workspace_bytes));
  CUDA_CHECK(cudaEventCreateWithFlags(&w.copies_done, cudaEventDisableTiming));
}

AttentionWorkspace::~AttentionWorkspace() = default;

// ------------------------------------------------------------------------------------------------
// Planning

namespace {

void validate_batch(const PagedKVCache& cache, const std::vector<int>& seq_ids) {
  require(!seq_ids.empty(), "empty batch");
  std::unordered_set<int> seen;
  for (int id : seq_ids) {
    (void)cache.length(id);  // throws on unknown ids
    require(seen.insert(id).second, "sequence " + std::to_string(id) + " appears twice in batch");
  }
}

// Fills kv_indptr / kv_last_page_len / kv_indices for the batch at the given offsets of `h`
// (host) and points the plan's device views at the matching offsets of `d`.
void stage_page_table(const PagedKVCache& cache, const std::vector<int>& seq_ids, int32_t* h,
                      int32_t* d, size_t indptr_off, size_t last_len_off, size_t indices_off,
                      AttentionWorkspace::Impl& w) {
  const int B = int(seq_ids.size());
  int32_t* indptr = h + indptr_off;
  int32_t* last_len = h + last_len_off;
  int32_t* indices = h + indices_off;
  indptr[0] = 0;
  for (int b = 0; b < B; ++b) {
    const auto& pages = cache.pages(seq_ids[b]);
    std::copy(pages.begin(), pages.end(), indices + indptr[b]);
    indptr[b + 1] = indptr[b] + int32_t(pages.size());
    const int len = cache.length(seq_ids[b]);
    last_len[b] = pages.empty() ? 0 : len - (int(pages.size()) - 1) * cache.page_size();
  }
  w.kv_indptr = d + indptr_off;
  w.kv_last_page_len = d + last_len_off;
  w.kv_indices = d + indices_off;
}

size_t total_pages(const PagedKVCache& cache, const std::vector<int>& seq_ids) {
  size_t n = 0;
  for (int id : seq_ids) n += cache.pages(id).size();
  return n;
}

// Windows at least this wide are treated as "no window": they cover any sequence this code can
// hold, and keep FlashInfer's uint32 window arithmetic (window + tile sizes) far from overflow.
constexpr int64_t kUnboundedWindow = int64_t(1) << 30;

// The window_left to hand to FlashInfer for a (window_left, window_right) streaming window.
// Queries trail the end of the cache by window_right, so the query at position p is
// end-aligned like a causal query at p + window_right:
//   * FlashInfer's causal bound  kv <= p + window_right  is exactly the right edge, and
//   * FlashInfer's window  kv >= (p + window_right) - W  is the left edge p - window_left
//     for W = window_left + window_right.
// So the stock causal + sliding-window kernels compute the streaming window exactly.
int flashinfer_window(int window_left, int window_right) {
  if (window_left < 0) return -1;
  const int64_t w = int64_t(window_left) + window_right;
  return w >= kUnboundedWindow ? -1 : int(w);
}

void validate_window_right(int window_right) {
  require(window_right >= 0, "window_right must be >= 0");
  require(window_right < kUnboundedWindow, "window_right too large");
}

// Checks an indptr of batch_size + 1 non-decreasing offsets starting at 0; returns the lengths.
std::vector<int> segment_lengths(const std::vector<int32_t>& indptr, int batch_size,
                                 const char* name) {
  require(int(indptr.size()) == batch_size + 1,
          std::string(name) + " must have batch_size + 1 entries");
  require(indptr[0] == 0, std::string(name) + "[0] must be 0");
  std::vector<int> lens(batch_size);
  for (int b = 0; b < batch_size; ++b) {
    lens[b] = indptr[b + 1] - indptr[b];
    require(lens[b] >= 0, std::string(name) + " must be non-decreasing");
  }
  return lens;
}

void begin_plan(AttentionWorkspace::Impl& w, PagedKVCache& cache, cudaStream_t stream,
                int fi_window) {
  w.kind = AttentionWorkspace::Impl::Kind::kNone;  // invalid until this plan succeeds
  w.cache = &cache;
  w.stream = stream;
  w.window_left = fi_window;
  w.wait_previous_copies();
}

}  // namespace

void plan_prefill(AttentionWorkspace& ws, PagedKVCache& cache, const std::vector<int>& seq_ids,
                  const std::vector<int32_t>& qo_indptr, const std::vector<int32_t>& kv_indptr,
                  int window_left, int window_right, cudaStream_t stream) {
  AttentionWorkspace::Impl& w = *ws.impl_;
  validate_batch(cache, seq_ids);
  validate_window_right(window_right);
  const int B = int(seq_ids.size());
  const std::vector<int> num_q = segment_lengths(qo_indptr, B, "qo_indptr");
  const std::vector<int> num_new = segment_lengths(kv_indptr, B, "kv_indptr");
  std::vector<int> old_len(B);
  for (int b = 0; b < B; ++b) {
    old_len[b] = cache.length(seq_ids[b]);
    // Queries are the last num_q[b] positions whose right context is complete.
    const int64_t computable =
        std::max<int64_t>(0, int64_t(old_len[b]) + num_new[b] - window_right);
    require(num_q[b] <= computable,
            "sequence " + std::to_string(seq_ids[b]) + ": " + std::to_string(num_q[b]) +
                " queries but only " + std::to_string(computable) +
                " positions have their window_right context in the cache");
  }
  const uint32_t nnz_q = uint32_t(qo_indptr[B]);
  const uint32_t nnz_kv = uint32_t(kv_indptr[B]);

  begin_plan(w, cache, stream, flashinfer_window(window_left, window_right));
  cache.extend(seq_ids, num_new);

  // Index buffer layout: kv_indptr | kv_last_page_len | qo_indptr | batch_indices | positions |
  // kv_indices   (batch_indices / positions address the appended K/V rows)
  const size_t off_indptr = 0, off_last = off_indptr + B + 1, off_qo = off_last + B,
               off_bidx = off_qo + B + 1, off_pos = off_bidx + nnz_kv, off_idx = off_pos + nnz_kv;
  const size_t count = off_idx + total_pages(cache, seq_ids);
  w.reserve_index_buffers(count);
  int32_t* h = w.index_pinned;
  int32_t* d = w.index_device;
  stage_page_table(cache, seq_ids, h, d, off_indptr, off_last, off_idx, w);
  std::copy(qo_indptr.begin(), qo_indptr.end(), h + off_qo);
  for (int b = 0; b < B; ++b) {
    for (int i = 0; i < num_new[b]; ++i) {
      h[off_bidx + kv_indptr[b] + i] = b;
      h[off_pos + kv_indptr[b] + i] = old_len[b] + i;
    }
  }
  w.qo_indptr = d + off_qo;
  w.batch_indices = d + off_bidx;
  w.positions = d + off_pos;
  CUDA_CHECK(cudaMemcpyAsync(d, h, count * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

  if (nnz_q > 0) {
    // The scheduler reads the host copies of qo_indptr and the page-granular kv_indptr.
    CUDA_CHECK(flashinfer::PrefillPlan<int32_t>(
        w.float_workspace, w.float_workspace_bytes, w.int_workspace, w.pinned_int_workspace,
        w.int_workspace_bytes, w.prefill_info, h + off_qo, h + off_indptr, nnz_q, B, kNumQoHeads,
        kNumKvHeads, kHeadDim, kHeadDim, cache.page_size(), /*enable_cuda_graph=*/false,
        /*sizeof_dtype_o=*/sizeof(bf16), w.window_left, /*fixed_split_size=*/-1,
        /*disable_split_kv=*/false, /*num_colocated_ctas=*/0, /*uniform_q_len=*/0, stream,
        /*kv_dtype_bytes=*/sizeof(bf16)));
  }
  w.mark_copies_issued(stream);

  w.batch_size = B;
  w.num_q = nnz_q;
  w.num_kv = nnz_kv;
  w.cache_version = cache.version();
  w.kind = AttentionWorkspace::Impl::Kind::kPrefill;
}

void plan_decode(AttentionWorkspace& ws, PagedKVCache& cache, const std::vector<int>& seq_ids,
                 int window_left, int window_right, cudaStream_t stream) {
  AttentionWorkspace::Impl& w = *ws.impl_;
  validate_batch(cache, seq_ids);
  validate_window_right(window_right);
  const int B = int(seq_ids.size());
  for (int id : seq_ids) {
    // The query sits at (length after the append) - 1 - window_right.
    require(cache.length(id) >= window_right,
            "sequence " + std::to_string(id) + " has " + std::to_string(cache.length(id)) +
                " tokens; a decode step needs at least window_right = " +
                std::to_string(window_right) + " before it (use prefill with no queries)");
  }

  begin_plan(w, cache, stream, flashinfer_window(window_left, window_right));
  cache.extend(seq_ids, std::vector<int>(B, 1));

  // Index buffer layout: kv_indptr | kv_last_page_len | kv_indices
  const size_t off_indptr = 0, off_last = off_indptr + B + 1, off_idx = off_last + B;
  const size_t count = off_idx + total_pages(cache, seq_ids);
  w.reserve_index_buffers(count);
  int32_t* h = w.index_pinned;
  int32_t* d = w.index_device;
  stage_page_table(cache, seq_ids, h, d, off_indptr, off_last, off_idx, w);
  w.qo_indptr = w.batch_indices = w.positions = nullptr;
  CUDA_CHECK(cudaMemcpyAsync(d, h, count * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

  CUDA_CHECK(detail::PlanDecode(w.float_workspace, w.float_workspace_bytes, w.int_workspace,
                                w.pinned_int_workspace, w.int_workspace_bytes, w.decode_info,
                                h + off_indptr, B, cache.page_size(), w.window_left >= 0, stream));
  w.mark_copies_issued(stream);

  w.batch_size = B;
  w.num_q = B;
  w.num_kv = B;
  w.cache_version = cache.version();
  w.kind = AttentionWorkspace::Impl::Kind::kDecode;
}

// ------------------------------------------------------------------------------------------------
// Execution

void run_planned(AttentionWorkspace& ws, const bf16* q, const bf16* k, const bf16* v, bf16* out) {
  using flashinfer::GetPtrFromBaseOffset;
  using Kind = AttentionWorkspace::Impl::Kind;
  AttentionWorkspace::Impl& w = *ws.impl_;
  require(w.kind != Kind::kNone, "run_planned: no successful plan");
  require(w.cache->version() == w.cache_version,
          "run_planned: the KV cache changed since the plan; plan again");
  require(w.num_q == 0 || (q && out), "run_planned: null q/out");
  require(w.num_kv == 0 || (k && v), "run_planned: null k/v");

  PagedKVCache& cache = *w.cache;
  flashinfer::paged_kv_t<bf16, int32_t> paged_kv(
      kNumKvHeads, cache.page_size(), kHeadDim, w.batch_size, flashinfer::QKVLayout::kNHD,
      cache.k_pool(), cache.v_pool(), w.kv_indices, w.kv_indptr, w.kv_last_page_len);
  // FlashInfer takes non-const pointers but only reads the inputs.
  bf16* k_in = const_cast<bf16*>(k);
  bf16* v_in = const_cast<bf16*>(v);
  bf16* q_in = const_cast<bf16*>(q);
  const float sm_scale = 1.0f / std::sqrt(float(kHeadDim));
  const int32_t q_stride_n = kNumQoHeads * kHeadDim, q_stride_h = kHeadDim;
  void* fbuf = w.float_workspace;
  void* ibuf = w.int_workspace;

  if (w.kind == Kind::kPrefill) {
    if (w.num_kv > 0) {  // FlashInfer would launch an empty grid for nnz == 0
      CUDA_CHECK(flashinfer::AppendPagedKVCache(paged_kv, k_in, v_in, w.batch_indices,
                                                w.positions, w.num_kv, kTokenElems, kHeadDim,
                                                kTokenElems, kHeadDim, w.stream));
    }
    if (w.num_q == 0) return;  // only K/V this step (e.g. streams still filling their look-ahead)
    const flashinfer::PrefillPlanInfo& info = w.prefill_info;
    detail::PrefillParams p(q_in, paged_kv, /*maybe_custom_mask=*/nullptr, w.qo_indptr,
                            /*maybe_mask_indptr=*/nullptr, /*maybe_q_rope_offset=*/nullptr, out,
                            /*lse=*/nullptr, /*maybe_alibi_slopes=*/nullptr, kNumQoHeads,
                            q_stride_n, q_stride_h, w.window_left, /*logits_soft_cap=*/0.f,
                            sm_scale, /*rope_scale=*/1.f, /*rope_theta=*/1e4f);
    p.request_indices = GetPtrFromBaseOffset<int32_t>(ibuf, info.request_indices_offset);
    p.qo_tile_indices = GetPtrFromBaseOffset<int32_t>(ibuf, info.qo_tile_indices_offset);
    p.kv_tile_indices = GetPtrFromBaseOffset<int32_t>(ibuf, info.kv_tile_indices_offset);
    p.o_indptr = GetPtrFromBaseOffset<int32_t>(ibuf, info.o_indptr_offset);
    p.kv_chunk_size_ptr = GetPtrFromBaseOffset<int32_t>(ibuf, info.kv_chunk_size_ptr_offset);
    p.padded_batch_size = uint32_t(info.padded_batch_size);
    p.max_total_num_rows = uint32_t(info.total_num_rows);
    bf16* tmp_v = nullptr;
    float* tmp_s = nullptr;
    if (info.split_kv) {
      p.merge_indptr = GetPtrFromBaseOffset<int32_t>(ibuf, info.merge_indptr_offset);
      tmp_v = GetPtrFromBaseOffset<bf16>(fbuf, info.v_offset);
      tmp_s = GetPtrFromBaseOffset<float>(fbuf, info.s_offset);
    }
    const uint32_t tile = uint32_t(info.cta_tile_q);
    CUDA_CHECK(w.window_left >= 0 ? detail::LaunchPrefill<true>(p, tmp_v, tmp_s, tile, w.stream)
                                  : detail::LaunchPrefill<false>(p, tmp_v, tmp_s, tile, w.stream));
  } else {
    // Writes each token at position length - 1 of its sequence (the plan already counted it).
    CUDA_CHECK(flashinfer::AppendPagedKVCacheDecode(paged_kv, k_in, v_in, w.stream));
    const flashinfer::DecodePlanInfo& info = w.decode_info;
    detail::DecodeParams p(q_in, /*q_rope_offset=*/nullptr, paged_kv, out, /*lse=*/nullptr,
                           /*maybe_alibi_slopes=*/nullptr, kNumQoHeads, q_stride_n, q_stride_h,
                           w.window_left, /*logits_soft_cap=*/0.f, sm_scale, /*rope_scale=*/1.f,
                           /*rope_theta=*/1e4f);
    p.request_indices = GetPtrFromBaseOffset<int32_t>(ibuf, info.request_indices_offset);
    p.kv_tile_indices = GetPtrFromBaseOffset<int32_t>(ibuf, info.kv_tile_indices_offset);
    p.o_indptr = GetPtrFromBaseOffset<int32_t>(ibuf, info.o_indptr_offset);
    p.kv_chunk_size_ptr = GetPtrFromBaseOffset<int32_t>(ibuf, info.kv_chunk_size_ptr_offset);
    p.padded_batch_size = uint32_t(info.padded_batch_size);
    bf16* tmp_v = nullptr;
    float* tmp_s = nullptr;
    if (info.split_kv) {
      tmp_v = GetPtrFromBaseOffset<bf16>(fbuf, info.v_offset);
      tmp_s = GetPtrFromBaseOffset<float>(fbuf, info.s_offset);
    }
    CUDA_CHECK(detail::LaunchDecode(p, tmp_v, tmp_s, w.window_left >= 0, w.stream));
  }
}

void prefill(AttentionWorkspace& ws, PagedKVCache& cache, const std::vector<int>& seq_ids,
             const bf16* q, const std::vector<int32_t>& qo_indptr, const bf16* k, const bf16* v,
             const std::vector<int32_t>& kv_indptr, int window_left, int window_right, bf16* out,
             cudaStream_t stream) {
  plan_prefill(ws, cache, seq_ids, qo_indptr, kv_indptr, window_left, window_right, stream);
  run_planned(ws, q, k, v, out);
}

void prefill(AttentionWorkspace& ws, PagedKVCache& cache, const std::vector<int>& seq_ids,
             const bf16* q, const bf16* k, const bf16* v, const std::vector<int32_t>& qo_indptr,
             int window_left, bf16* out, cudaStream_t stream) {
  for (int n : segment_lengths(qo_indptr, int(seq_ids.size()), "qo_indptr")) {
    require(n >= 1, "every sequence in a prefill batch needs at least one token");
  }
  prefill(ws, cache, seq_ids, q, qo_indptr, k, v, qo_indptr, window_left, /*window_right=*/0, out,
          stream);
}

void decode(AttentionWorkspace& ws, PagedKVCache& cache, const std::vector<int>& seq_ids,
            const bf16* q, const bf16* k, const bf16* v, int window_left, int window_right,
            bf16* out, cudaStream_t stream) {
  plan_decode(ws, cache, seq_ids, window_left, window_right, stream);
  run_planned(ws, q, k, v, out);
}

void decode(AttentionWorkspace& ws, PagedKVCache& cache, const std::vector<int>& seq_ids,
            const bf16* q, const bf16* k, const bf16* v, int window_left, bf16* out,
            cudaStream_t stream) {
  decode(ws, cache, seq_ids, q, k, v, window_left, /*window_right=*/0, out, stream);
}

}  // namespace attn
