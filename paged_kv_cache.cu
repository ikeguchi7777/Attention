// PagedKVCache: page pool, per-sequence page tables, reference counting, fork / copy-on-write /
// reorder for beam search.
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "attention.h"

namespace attn {
namespace {

void require(bool cond, const std::string& msg) {
  if (!cond) throw std::invalid_argument(msg);
}

int ceil_div(int a, int b) { return (a + b - 1) / b; }

constexpr size_t kTokenElems = size_t(kNumKvHeads) * kHeadDim;  // elements per token per pool

// Up to kMaxPageCopies (src page, dst page, used slots) triples travel in the kernel parameter
// block itself (6 KiB here; CUDA 12.1+ allows 32 KiB), so a copy-on-write needs no device-side
// staging buffer or host synchronization. Larger batches are split into several launches.
constexpr int kMaxPageCopies = 512;
struct PageCopyBatch {
  int32_t src[kMaxPageCopies];
  int32_t dst[kMaxPageCopies];
  int32_t tokens[kMaxPageCopies];
  int32_t count;
};

// blockIdx.x: copy index, blockIdx.y: 0 = K pool, 1 = V pool. Copies the first `tokens` slots of
// the source page; in the NHD layout those are one contiguous run of tokens * H * D elements.
__global__ void copy_page_prefixes(bf16* __restrict__ k_pool, bf16* __restrict__ v_pool,
                                   size_t page_elems, PageCopyBatch batch) {
  const int i = blockIdx.x;
  if (i >= batch.count) return;
  bf16* pool = blockIdx.y == 0 ? k_pool : v_pool;
  const uint4* src = reinterpret_cast<const uint4*>(pool + size_t(batch.src[i]) * page_elems);
  uint4* dst = reinterpret_cast<uint4*>(pool + size_t(batch.dst[i]) * page_elems);
  // kTokenElems * sizeof(bf16) = 1 KiB per token, a multiple of 16 bytes.
  const size_t n = size_t(batch.tokens[i]) * kTokenElems * sizeof(bf16) / sizeof(uint4);
  for (size_t j = threadIdx.x; j < n; j += blockDim.x) dst[j] = src[j];
}

}  // namespace

PagedKVCache::PagedKVCache(int page_size, int max_num_pages)
    : page_size_(page_size), max_num_pages_(max_num_pages) {
  require(page_size > 0, "page_size must be positive");
  require(max_num_pages > 0, "max_num_pages must be positive");
  const size_t pool_bytes = size_t(max_num_pages) * page_size * kTokenElems * sizeof(bf16);
  CUDA_CHECK(cudaMalloc(&k_pool_, pool_bytes));
  CUDA_CHECK(cudaMalloc(&v_pool_, pool_bytes));
  // Hand out low page ids first.
  free_pages_.resize(max_num_pages);
  for (int i = 0; i < max_num_pages; ++i) free_pages_[i] = max_num_pages - 1 - i;
  ref_counts_.assign(max_num_pages, 0);
}

PagedKVCache::~PagedKVCache() {
  cudaFree(k_pool_);
  cudaFree(v_pool_);
}

PagedKVCache::Sequence& PagedKVCache::seq(int seq_id) {
  require(seq_id >= 0 && seq_id < int(seqs_.size()) && seqs_[seq_id].active,
          "unknown sequence id " + std::to_string(seq_id));
  return seqs_[seq_id];
}

const PagedKVCache::Sequence& PagedKVCache::seq(int seq_id) const {
  require(seq_id >= 0 && seq_id < int(seqs_.size()) && seqs_[seq_id].active,
          "unknown sequence id " + std::to_string(seq_id));
  return seqs_[seq_id];
}

int32_t PagedKVCache::take_free_page() {
  const int32_t page = free_pages_.back();
  free_pages_.pop_back();
  ref_counts_[page] = 1;
  return page;
}

void PagedKVCache::drop_reference(int32_t page) {
  assert(ref_counts_[page] > 0);
  if (--ref_counts_[page] == 0) free_pages_.push_back(page);
}

int PagedKVCache::ref_count(int32_t page) const {
  require(page >= 0 && page < max_num_pages_, "page id out of range");
  return ref_counts_[page];
}

int PagedKVCache::add_sequence() {
  int id;
  if (!free_seq_ids_.empty()) {
    id = free_seq_ids_.back();
    free_seq_ids_.pop_back();
  } else {
    id = int(seqs_.size());
    seqs_.emplace_back();
  }
  seqs_[id] = Sequence{};
  seqs_[id].active = true;
  ++version_;
  return id;
}

void PagedKVCache::free_sequence(int seq_id) {
  Sequence& s = seq(seq_id);
  // Drop back to front so an unshared sequence's pages come back out in their original order.
  for (auto it = s.pages.rbegin(); it != s.pages.rend(); ++it) drop_reference(*it);
  s = Sequence{};
  free_seq_ids_.push_back(seq_id);
  ++version_;
  debug_check();
}

void PagedKVCache::truncate(int seq_id, int new_length) {
  Sequence& s = seq(seq_id);
  require(new_length >= 0 && new_length <= s.length, "truncate: new_length out of range");
  const size_t keep = size_t(ceil_div(new_length, page_size_));
  while (s.pages.size() > keep) {
    drop_reference(s.pages.back());
    s.pages.pop_back();
  }
  s.length = new_length;
  ++version_;
  debug_check();
}

int PagedKVCache::length(int seq_id) const { return seq(seq_id).length; }

const std::vector<int32_t>& PagedKVCache::pages(int seq_id) const { return seq(seq_id).pages; }

void PagedKVCache::extend(const std::vector<int>& seq_ids, const std::vector<int>& num_new_tokens) {
  require(seq_ids.size() == num_new_tokens.size(), "extend: size mismatch");
  size_t pages_needed = 0;
  for (size_t i = 0; i < seq_ids.size(); ++i) {
    const Sequence& s = seq(seq_ids[i]);
    require(num_new_tokens[i] >= 0, "extend: negative token count");
    require(int64_t(s.length) + num_new_tokens[i] <= INT32_MAX, "extend: sequence too long");
    if (num_new_tokens[i] > 0 && s.length % page_size_ != 0 &&
        ref_counts_[s.pages.back()] > 1) {
      throw std::logic_error("PagedKVCache: sequence " + std::to_string(seq_ids[i]) +
                             " would append into shared page " + std::to_string(s.pages.back()) +
                             "; call copy_on_write() first");
    }
    pages_needed += ceil_div(s.length + num_new_tokens[i], page_size_) - s.pages.size();
  }
  if (pages_needed > free_pages_.size()) {
    throw std::runtime_error("PagedKVCache: out of pages (need " + std::to_string(pages_needed) +
                             ", free " + std::to_string(free_pages_.size()) + ")");
  }
  for (size_t i = 0; i < seq_ids.size(); ++i) {
    Sequence& s = seq(seq_ids[i]);
    s.length += num_new_tokens[i];
    while (int(s.pages.size()) < ceil_div(s.length, page_size_)) {
      s.pages.push_back(take_free_page());
    }
  }
  ++version_;
  debug_check();
}

std::vector<int> PagedKVCache::fork(int src_seq, int n) {
  require(n >= 0, "fork: n must be non-negative");
  (void)seq(src_seq);  // validate before creating anything
  std::vector<int> ids;
  for (int i = 0; i < n; ++i) {
    const int id = add_sequence();
    const Sequence& src = seq(src_seq);  // add_sequence may reallocate seqs_
    seqs_[id].pages = src.pages;
    seqs_[id].length = src.length;
    for (int32_t page : src.pages) ++ref_counts_[page];
    ids.push_back(id);
  }
  ++version_;
  debug_check();
  return ids;
}

int PagedKVCache::copy_on_write(const std::vector<int>& seq_ids, cudaStream_t stream) {
  std::unordered_set<int> seen;
  for (int id : seq_ids) {
    (void)seq(id);
    require(seen.insert(id).second, "copy_on_write: duplicate sequence id");
  }
  // Which sequences need a copy depends on the order: once all but one holder of a shared last
  // page have copied it, the last holder owns it. Simulate on the counts first so that running
  // out of pages leaves everything unchanged.
  std::vector<int> needs_copy;
  {
    std::unordered_map<int32_t, int> counts;  // last page -> simulated ref count
    for (int id : seq_ids) {
      const Sequence& s = seq(id);
      if (s.length % page_size_ == 0) continue;  // empty, or last page full: nothing to protect
      auto [it, inserted] = counts.try_emplace(s.pages.back(), ref_counts_[s.pages.back()]);
      int& c = it->second;
      if (c > 1) {
        --c;
        needs_copy.push_back(id);
      }
    }
  }
  if (needs_copy.size() > free_pages_.size()) {
    throw std::runtime_error("PagedKVCache: out of pages for copy-on-write (need " +
                             std::to_string(needs_copy.size()) + ", free " +
                             std::to_string(free_pages_.size()) + ")");
  }
  if (needs_copy.empty()) return 0;

  const size_t page_elems = size_t(page_size_) * kTokenElems;
  PageCopyBatch batch;
  batch.count = 0;
  auto flush = [&] {
    if (batch.count == 0) return;
    copy_page_prefixes<<<dim3(batch.count, 2), 256, 0, stream>>>(k_pool_, v_pool_, page_elems,
                                                                   batch);
    CUDA_CHECK(cudaGetLastError());
    batch.count = 0;
  };
  for (int id : needs_copy) {
    Sequence& s = seq(id);
    const int32_t old_page = s.pages.back();
    const int32_t new_page = take_free_page();
    batch.src[batch.count] = old_page;
    batch.dst[batch.count] = new_page;
    batch.tokens[batch.count] = s.length % page_size_;
    if (++batch.count == kMaxPageCopies) flush();
    s.pages.back() = new_page;
    drop_reference(old_page);  // still >= 1: another holder keeps it
  }
  flush();
  ++version_;
  debug_check();
  return int(needs_copy.size());
}

void PagedKVCache::reorder(const std::vector<int>& beam_ids, const std::vector<int>& parent_idx) {
  const size_t W = beam_ids.size();
  require(parent_idx.size() == W, "reorder: parent_idx must have one entry per beam");
  std::unordered_set<int> seen;
  for (int id : beam_ids) {
    (void)seq(id);
    require(seen.insert(id).second, "reorder: duplicate beam id");
  }
  for (int p : parent_idx) require(p >= 0 && size_t(p) < W, "reorder: parent index out of range");

  // Snapshot the old tables: children read their parent's table as it was before the call.
  std::vector<Sequence> old(W);
  for (size_t i = 0; i < W; ++i) old[i] = seq(beam_ids[i]);
  // 1) Add every child's references first ...
  for (size_t i = 0; i < W; ++i) {
    for (int32_t page : old[parent_idx[i]].pages) ++ref_counts_[page];
  }
  // 2) ... then drop each beam's old references. Doing this per beam in one pass instead would
  //    free pages of a parent that a later child still has to pick up.
  for (size_t i = 0; i < W; ++i) {
    for (int32_t page : old[i].pages) drop_reference(page);
  }
  for (size_t i = 0; i < W; ++i) {
    Sequence& s = seq(beam_ids[i]);
    s.pages = old[parent_idx[i]].pages;
    s.length = old[parent_idx[i]].length;
  }
  ++version_;
  debug_check();
}

bool PagedKVCache::invariants_hold(std::string* why) const {
  auto fail = [&](const std::string& msg) {
    if (why) *why = msg;
    return false;
  };
  std::vector<int64_t> occurrences(max_num_pages_, 0);
  int64_t table_entries = 0;
  for (size_t id = 0; id < seqs_.size(); ++id) {
    const Sequence& s = seqs_[id];
    if (!s.active) continue;
    if (int(s.pages.size()) != ceil_div(s.length, page_size_)) {
      return fail("sequence " + std::to_string(id) + " has " + std::to_string(s.pages.size()) +
                  " pages for length " + std::to_string(s.length));
    }
    for (int32_t page : s.pages) {
      if (page < 0 || page >= max_num_pages_) return fail("page id out of range");
      ++occurrences[page];
    }
    table_entries += int64_t(s.pages.size());
  }
  int64_t ref_sum = 0;
  int used = 0;
  for (int p = 0; p < max_num_pages_; ++p) {
    if (ref_counts_[p] != occurrences[p]) {
      return fail("page " + std::to_string(p) + " has ref count " +
                  std::to_string(ref_counts_[p]) + " but appears " +
                  std::to_string(occurrences[p]) + " times in page tables");
    }
    ref_sum += ref_counts_[p];
    used += ref_counts_[p] > 0;
  }
  if (ref_sum != table_entries) {
    return fail("ref counts sum to " + std::to_string(ref_sum) + " but page tables hold " +
                std::to_string(table_entries) + " entries");
  }
  std::vector<bool> in_free(max_num_pages_, false);
  for (int32_t page : free_pages_) {
    if (page < 0 || page >= max_num_pages_) return fail("free list holds an invalid page id");
    if (in_free[page]) return fail("page " + std::to_string(page) + " is twice in the free list");
    if (ref_counts_[page] != 0) {
      return fail("page " + std::to_string(page) + " is in the free list but in use");
    }
    in_free[page] = true;
  }
  if (used + int(free_pages_.size()) != max_num_pages_) {
    return fail(std::to_string(max_num_pages_ - used - int(free_pages_.size())) +
                " unused pages are missing from the free list (leak)");
  }
  return true;
}

void PagedKVCache::check_invariants() const {
  std::string why;
  if (!invariants_hold(&why)) throw std::logic_error("PagedKVCache invariant violated: " + why);
}

void PagedKVCache::debug_check() const {
#ifndef NDEBUG
  std::string why;
  const bool ok = invariants_hold(&why);
  if (!ok) std::fprintf(stderr, "PagedKVCache invariant violated: %s\n", why.c_str());
  assert(ok);
#endif
}

}  // namespace attn
