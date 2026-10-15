// Batched paged-KV attention (prefill / decode) on top of FlashInfer's header-only C++ API.
//
// Fixed model shape: 8 query heads, 8 KV heads (MHA), head_dim 64, bf16 everywhere.
// Tensors passed in are device pointers laid out as [tokens, heads, head_dim].
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Checks a cudaError_t-returning expression and throws std::runtime_error on failure.
#define CUDA_CHECK(expr)                                                              \
  do {                                                                                \
    cudaError_t err_ = (expr);                                                        \
    if (err_ != cudaSuccess) {                                                        \
      std::ostringstream oss_;                                                        \
      oss_ << "CUDA error '" << cudaGetErrorString(err_) << "' (" << int(err_) << ")" \
           << " at " << __FILE__ << ":" << __LINE__ << ": " #expr;                    \
      throw std::runtime_error(oss_.str());                                           \
    }                                                                                 \
  } while (0)

namespace attn {

using bf16 = __nv_bfloat16;

constexpr int kNumQoHeads = 8;
constexpr int kNumKvHeads = 8;
constexpr int kHeadDim = 64;

// window_left  = number of earlier tokens each query may attend to besides itself
//                (FlashInfer's convention): 0 = only itself, kNoWindow (any negative value) =
//                everything before it.
// window_right = number of later tokens each query may attend to (look-ahead, for streaming);
//                must be >= 0, 0 = causal. A query is only computed once its right context is in
//                the cache, so with window_right = R the queries trail the cache by R tokens
//                (see the streaming prefill() / decode() below).
constexpr int kNoWindow = -1;

// Default number of tokens per KV page (see README "page_size の決定"). The page size is a
// runtime argument of PagedKVCache; any positive value works.
constexpr int kDefaultPageSize = 16;

// Pool of fixed-size KV pages on the GPU plus a per-sequence page table on the host.
// Each pool (K and V) is NHD: [max_num_pages, page_size, kNumKvHeads, kHeadDim].
//
// Pages are reference counted so that sequences can share them (beam search): fork() shares a
// page table without copying, copy_on_write() gives a sequence a private copy of a shared,
// partially filled last page before it is appended to, and a page returns to the free list when
// its count drops to zero. GPU work issued by the cache (page copies) and by prefill()/decode()
// must go to the same stream so that a page freed here is not rewritten while still being read.
// Debug builds (NDEBUG undefined) assert check_invariants() after every page-table change.
class PagedKVCache {
 public:
  PagedKVCache(int page_size, int max_num_pages);
  ~PagedKVCache();
  PagedKVCache(const PagedKVCache&) = delete;
  PagedKVCache& operator=(const PagedKVCache&) = delete;

  // Registers an empty sequence and returns its id.
  int add_sequence();
  // Drops the sequence's references to its pages (pages nobody else uses go back to the free
  // list); the id becomes invalid and may be reused.
  void free_sequence(int seq_id);
  void release(int seq_id) { free_sequence(seq_id); }
  // Drops tokens past new_length (e.g. to roll back), releasing pages that become unused.
  void truncate(int seq_id, int new_length);

  int length(int seq_id) const;
  const std::vector<int32_t>& pages(int seq_id) const;
  int page_size() const { return page_size_; }
  int max_num_pages() const { return max_num_pages_; }
  int num_free_pages() const { return static_cast<int>(free_pages_.size()); }
  int num_used_pages() const { return max_num_pages_ - num_free_pages(); }
  int ref_count(int32_t page) const;
  bf16* k_pool() const { return k_pool_; }
  bf16* v_pool() const { return v_pool_; }

  // Grows each listed sequence by the matching count, allocating pages as needed.
  // All-or-nothing: throws without changing anything if the pool cannot hold the new tokens.
  // Throws std::logic_error if a sequence would append into a shared, partially filled last page
  // (call copy_on_write() first).
  void extend(const std::vector<int>& seq_ids, const std::vector<int>& num_new_tokens);

  // ---- Sharing (beam search) --------------------------------------------------------------------

  // Creates n sequences that share src_seq's page table (no K/V is copied; every page of src
  // gains n references) and returns their ids. src_seq stays valid.
  std::vector<int> fork(int src_seq, int n);

  // Makes the last page of each listed sequence private before K/V is appended to it: when that
  // page is partially filled and shared (ref count > 1), a fresh page is allocated, its used
  // slots are copied on the GPU, the page table entry is swapped and the old page loses a
  // reference. A full last page is left alone (the next token starts a new page). All copies of
  // one call are done by a single kernel launch on `stream`. All-or-nothing on page exhaustion.
  // Returns the number of pages copied.
  int copy_on_write(const std::vector<int>& seq_ids, cudaStream_t stream = nullptr);

  // Beam reorder: beam i of the group takes over the page table (and length) that beam
  // parent_idx[i] had before the call. References are all added before any is dropped, so a page
  // still needed by some child never reaches zero midway; parents chosen by no child lose their
  // references (and their private pages go back to the free list).
  void reorder(const std::vector<int>& beam_ids, const std::vector<int>& parent_idx);

  // Checks: every page's ref count equals its number of occurrences in the active page tables
  // (so the counts sum to the total page-table size), the free list holds exactly the pages
  // with count 0 and has no duplicates, and each table has ceil(length / page_size) pages.
  // Throws std::logic_error describing the first violation.
  void check_invariants() const;

  // Incremented by every change to the page table; plans remember it to detect staleness.
  uint64_t version() const { return version_; }

 private:
  struct Sequence {
    std::vector<int32_t> pages;
    int length = 0;
    bool active = false;
  };
  Sequence& seq(int seq_id);
  const Sequence& seq(int seq_id) const;
  int32_t take_free_page();     // count becomes 1
  void drop_reference(int32_t page);
  bool invariants_hold(std::string* why) const;
  void debug_check() const;     // asserts invariants_hold() unless NDEBUG

  int page_size_;
  int max_num_pages_;
  bf16* k_pool_ = nullptr;
  bf16* v_pool_ = nullptr;
  std::vector<int32_t> free_pages_;
  std::vector<int32_t> ref_counts_;
  std::vector<Sequence> seqs_;
  std::vector<int> free_seq_ids_;
  uint64_t version_ = 0;
};

// Scratch memory reused across steps (FlashInfer's float / int / page-locked plan workspaces and
// the per-step index arrays) together with the most recent plan.
// Use one workspace from one host thread and one CUDA stream at a time.
class AttentionWorkspace {
 public:
  // Defaults match FlashInfer's Python wrappers (128 MiB float, 8 MiB int).
  explicit AttentionWorkspace(size_t float_workspace_bytes = size_t(128) << 20,
                              size_t int_workspace_bytes = size_t(8) << 20);
  ~AttentionWorkspace();
  AttentionWorkspace(const AttentionWorkspace&) = delete;
  AttentionWorkspace& operator=(const AttentionWorkspace&) = delete;

  struct Impl;

 private:
  friend void plan_prefill(AttentionWorkspace&, PagedKVCache&, const std::vector<int>&,
                           const std::vector<int32_t>&, const std::vector<int32_t>&, int, int,
                           cudaStream_t);
  friend void plan_decode(AttentionWorkspace&, PagedKVCache&, const std::vector<int>&, int, int,
                          cudaStream_t);
  friend void run_planned(AttentionWorkspace&, const bf16*, const bf16*, const bf16*, bf16*);
  std::unique_ptr<Impl> impl_;
};

// ---- One-call API ------------------------------------------------------------------------------

// Prefill for a batch of sequences.
//   q, out    : [qo_indptr.back(), kNumQoHeads, kHeadDim]   (ragged, sequence i owns rows
//   k, v      : [qo_indptr.back(), kNumKvHeads, kHeadDim]    qo_indptr[i] .. qo_indptr[i+1]-1)
//   qo_indptr : host array of seq_ids.size() + 1 offsets starting at 0; every sequence gets >= 1 row
// The new K/V are appended to the cache, then each query attends causally to every cached token
// of its sequence (so a sequence that already holds tokens continues from that prefix), limited to
// the window_left preceding tokens unless window_left < 0.
void prefill(AttentionWorkspace& ws, PagedKVCache& cache, const std::vector<int>& seq_ids,
             const bf16* q, const bf16* k, const bf16* v, const std::vector<int32_t>& qo_indptr,
             int window_left, bf16* out, cudaStream_t stream = nullptr);

// One decode step: each sequence in seq_ids receives one token.
//   q, out : [seq_ids.size(), kNumQoHeads, kHeadDim]    k, v : [seq_ids.size(), kNumKvHeads, kHeadDim]
void decode(AttentionWorkspace& ws, PagedKVCache& cache, const std::vector<int>& seq_ids,
            const bf16* q, const bf16* k, const bf16* v, int window_left, bf16* out,
            cudaStream_t stream = nullptr);

// ---- Streaming API (window_right >= 0) ---------------------------------------------------------
//
// A query at position p attends to positions [p - window_left, p + window_right] of its sequence,
// and callers pass a query only once all of its right context is in the cache. After this call
// appends its K/V, sequence i (new length len_i) gets the last qo_len_i computable queries, i.e.
// positions [len_i - window_right - qo_len_i, len_i - window_right).
// With window_right = 0 and qo_indptr == kv_indptr this is the causal prefill above.

// Prefill / chunked streaming step.
//   q, out    : [qo_indptr.back(), kNumQoHeads, kHeadDim]   queries, ragged by qo_indptr
//   k, v      : [kv_indptr.back(), kNumKvHeads, kHeadDim]   tokens appended, ragged by kv_indptr
// Both indptrs are host arrays of seq_ids.size() + 1 offsets starting at 0. A sequence may get
// zero queries (e.g. it has not yet accumulated window_right tokens) or zero new tokens, but
// qo_len_i <= len_i - window_right must hold.
void prefill(AttentionWorkspace& ws, PagedKVCache& cache, const std::vector<int>& seq_ids,
             const bf16* q, const std::vector<int32_t>& qo_indptr, const bf16* k, const bf16* v,
             const std::vector<int32_t>& kv_indptr, int window_left, int window_right, bf16* out,
             cudaStream_t stream = nullptr);

// Streaming decode step: appends one token per sequence and computes the one query whose right
// context it completes, at position length - 1 - window_right (needs length >= window_right + 1
// after the append).
//   q, out : [seq_ids.size(), kNumQoHeads, kHeadDim]    k, v : [seq_ids.size(), kNumKvHeads, kHeadDim]
void decode(AttentionWorkspace& ws, PagedKVCache& cache, const std::vector<int>& seq_ids,
            const bf16* q, const bf16* k, const bf16* v, int window_left, int window_right,
            bf16* out, cudaStream_t stream = nullptr);

// ---- Two-phase API (what prefill() / decode() do internally) -----------------------------------

// Grows the sequences in the cache, uploads the page table and runs FlashInfer's scheduler once.
void plan_prefill(AttentionWorkspace& ws, PagedKVCache& cache, const std::vector<int>& seq_ids,
                  const std::vector<int32_t>& qo_indptr, const std::vector<int32_t>& kv_indptr,
                  int window_left, int window_right, cudaStream_t stream = nullptr);
void plan_decode(AttentionWorkspace& ws, PagedKVCache& cache, const std::vector<int>& seq_ids,
                 int window_left, int window_right, cudaStream_t stream = nullptr);
// Writes k/v into the slots reserved by the last plan and launches the attention kernel on the
// plan's stream. Can be repeated with the same plan (it rewrites the same slots), but throws if
// the cache's page table changed since the plan.
void run_planned(AttentionWorkspace& ws, const bf16* q, const bf16* k, const bf16* v, bf16* out);

}  // namespace attn
