// Beam search over the paged KV cache: fork, copy-on-write, reorder and release.
//
// No model is involved: Q/K/V are random and beam scores are pseudo log-probabilities, which is
// enough to reproduce the cache traffic of a real beam search. Every step
//   1. copy_on_write() + decode() for all R x W beams (decode() is the unchanged library call),
//   2. picks the top-W continuations per request from pseudo scores -> parent_idx,
//   3. reorder()s each request's beams.
// The reference keeps every beam's K/V history as a plain array and copies a parent's history
// wholesale on reorder; each decode output is compared against it. After every step the cache's
// invariants are checked and the page usage is printed next to what unshared (fully copied)
// beams would need. At the end every sequence is released and the pool must be whole again.
//
// Usage: beam_search_sample          (runs the built-in configurations, exit code 0 = all pass)
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "attention.h"
#include "ref_attention.h"

using attn::bf16;
using attn::kHeadDim;
using attn::kNoWindow;
using attn::kNumKvHeads;
using attn::kNumQoHeads;

namespace {

constexpr int kQoRow = kNumQoHeads * kHeadDim;
constexpr int kKvRow = kNumKvHeads * kHeadDim;

struct Config {
  std::string name;
  int num_requests;           // R
  int beam_width;             // W
  int steps;                  // T
  int page_size;
  std::vector<int> prompt_lens;
  bool all_same_parent;       // force every step to pick all children from one parent
};

struct Beam {
  int id = -1;
  std::vector<float> k, v;  // reference history [length, kNumKvHeads, kHeadDim]
  double score = 0;
  int length() const { return int(k.size() / kKvRow); }
};

template <typename T>
struct DeviceBuffer {
  T* ptr = nullptr;
  explicit DeviceBuffer(size_t n) { CUDA_CHECK(cudaMalloc(&ptr, std::max<size_t>(n, 1) * sizeof(T))); }
  ~DeviceBuffer() { cudaFree(ptr); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

class Runner {
 public:
  Runner(attn::AttentionWorkspace& ws, const Config& cfg, int window_left)
      : ws_(ws), cfg_(cfg), window_left_(window_left), rng_(std::hash<std::string>{}(cfg.name) +
                                                              unsigned(window_left)) {}

  bool run() {
    const int R = cfg_.num_requests, W = cfg_.beam_width, P = cfg_.page_size;
    // Unshared worst case plus slack for copy-on-write and the prompts themselves.
    int max_pages = 0;
    for (int len : cfg_.prompt_lens) max_pages += (W + 1) * ((len + cfg_.steps + P - 1) / P + 1);
    attn::PagedKVCache cache(P, max_pages);
    cache_ = &cache;

    std::printf("\n=== %s | R=%d W=%d T=%d page_size=%d window_left=%s | prompts:", cfg_.name.c_str(),
                R, W, cfg_.steps, P, window_left_ < 0 ? "off" : std::to_string(window_left_).c_str());
    for (int len : cfg_.prompt_lens) std::printf(" %d", len);
    std::printf("\n");

    // 1) Prefill the prompts (one sequence per request).
    std::vector<Beam> prompts(R);
    std::vector<int> prompt_ids;
    std::vector<int32_t> qo_indptr{0};
    for (int r = 0; r < R; ++r) {
      prompts[r].id = cache.add_sequence();
      prompt_ids.push_back(prompts[r].id);
      qo_indptr.push_back(qo_indptr.back() + cfg_.prompt_lens[r]);
    }
    {
      const int n = qo_indptr.back();
      auto q = random(size_t(n) * kQoRow, 2.f), k = random(size_t(n) * kKvRow, 2.f),
           v = random(size_t(n) * kKvRow, 1.f);
      for (int r = 0; r < R; ++r) append(prompts[r], k, v, qo_indptr[r], cfg_.prompt_lens[r]);
      auto out = run_on_gpu(n, q, k, v, [&](const bf16* dq, const bf16* dk, const bf16* dv,
                                            bf16* dout) {
        attn::prefill(ws_, cache, prompt_ids, dq, dk, dv, qo_indptr, window_left_, dout);
      });
      for (int r = 0; r < R; ++r) {
        for (int i = 0; i < cfg_.prompt_lens[r]; ++i) {
          const size_t row = size_t(qo_indptr[r] + i);
          check(q.data() + row * kQoRow, prompts[r], i, out.data() + row * kQoRow);
        }
      }
    }

    // 2) Fork each prompt into W beams, then drop the prompt sequence itself.
    beams_.assign(R, std::vector<Beam>(W));
    for (int r = 0; r < R; ++r) {
      const std::vector<int> ids = cache.fork(prompts[r].id, W);
      for (int b = 0; b < W; ++b) {
        beams_[r][b].id = ids[b];
        beams_[r][b].k = prompts[r].k;
        beams_[r][b].v = prompts[r].v;
      }
      cache.release(prompts[r].id);
    }
    cache.check_invariants();
    std::printf("after fork: used pages %d, unshared would need %d\n", cache.num_used_pages(),
                unshared_pages());
    std::printf("%5s %5s %6s %8s %10s %13s  %s\n", "step", "len", "CoW", "used", "unshared",
                "max_abs_err", "parents of request 0");

    // 3) Decode loop.
    for (int t = 0; t < cfg_.steps; ++t) {
      step(t);
      if (failed_) break;
    }

    // 4) Release everything; the pool must be whole again.
    for (auto& req : beams_) {
      for (Beam& b : req) cache.release(b.id);
    }
    cache.check_invariants();
    const bool no_leak = cache.num_free_pages() == cache.max_num_pages();
    std::printf("after release: used pages %d, free %d / %d -> %s\n", cache.num_used_pages(),
                cache.num_free_pages(), cache.max_num_pages(), no_leak ? "no leak" : "LEAK");
    if (!no_leak) fail("pages leaked");

    // Coverage of the selection patterns this configuration must exercise.
    if (W > 1 && steps_with_shared_parent_ == 0) fail("no step chose one parent twice");
    if (W > 1 && steps_with_dropped_parent_ == 0) fail("no step dropped a parent");
    if (cfg_.all_same_parent && steps_all_same_ != cfg_.steps) fail("not every step was all-same");
    if (P == 1 && total_copies_ != 0) fail("page_size 1 must never copy on write");
    std::printf("summary: rows=%ld bad=%ld max_abs_err=%.3e CoW pages=%d | steps with a parent "
                "chosen >1x: %d, with an unchosen parent: %d, all from one parent: %d -> %s\n",
                rows_checked_, rows_bad_, max_err_, total_copies_, steps_with_shared_parent_,
                steps_with_dropped_parent_, steps_all_same_, failed_ ? "FAIL" : "PASS");
    return !failed_;
  }

 private:
  void step(int t) {
    attn::PagedKVCache& cache = *cache_;
    const int R = cfg_.num_requests, W = cfg_.beam_width, B = R * W;
    std::vector<int> ids;
    for (auto& req : beams_) {
      for (Beam& b : req) ids.push_back(b.id);
    }
    auto q = random(size_t(B) * kQoRow, 2.f), k = random(size_t(B) * kKvRow, 2.f),
         v = random(size_t(B) * kKvRow, 1.f);

    // Expected copies: each partially filled last page held by n beams is copied n - 1 times.
    std::unordered_map<int32_t, int> holders;
    for (int id : ids) {
      if (cache.length(id) % cfg_.page_size != 0) ++holders[cache.pages(id).back()];
    }
    int expected_copies = 0;
    for (auto& [page, n] : holders) expected_copies += n - 1;

    // Appending without copy-on-write into a shared page must be refused (and change nothing).
    if (expected_copies > 0 && !guard_checked_) {
      guard_checked_ = true;
      const uint64_t version = cache.version();
      bool refused = false;
      try {
        DeviceBuffer<bf16> scratch(size_t(B) * kQoRow);
        attn::decode(ws_, cache, ids, scratch.ptr, scratch.ptr, scratch.ptr, window_left_,
                     scratch.ptr);
      } catch (const std::logic_error&) {
        refused = true;
      }
      if (!refused || cache.version() != version) {
        fail("decode into a shared page was not refused");
      } else {
        std::printf("      (step %d: decode() without copy_on_write() into a shared page was "
                    "refused, cache unchanged)\n", t);
      }
    }

    // 3.1) Copy-on-write, then the unchanged decode() for all R x W beams.
    const int copies = cache.copy_on_write(ids);
    total_copies_ += copies;
    const bool prompts_page_aligned =
        std::all_of(cfg_.prompt_lens.begin(), cfg_.prompt_lens.end(),
                    [&](int len) { return len % cfg_.page_size == 0; });
    if (t == 0 && prompts_page_aligned && copies != 0) {
      fail("forked from full last pages, yet step 0 copied " + std::to_string(copies) + " pages");
    }
    if (copies != expected_copies) {
      fail("step " + std::to_string(t) + ": " + std::to_string(copies) + " copies, expected " +
           std::to_string(expected_copies));
    }
    int idx = 0;
    for (auto& req : beams_) {
      for (Beam& b : req) append(b, k, v, idx++, 1);
    }
    step_err_ = 0;
    auto out = run_on_gpu(B, q, k, v, [&](const bf16* dq, const bf16* dk, const bf16* dv,
                                          bf16* dout) {
      attn::decode(ws_, cache, ids, dq, dk, dv, window_left_, dout);
    });
    idx = 0;
    for (auto& req : beams_) {
      for (Beam& b : req) {
        check(q.data() + size_t(idx) * kQoRow, b, b.length() - 1, out.data() + size_t(idx) * kQoRow);
        ++idx;
      }
    }

    // 3.2) Pseudo scores -> top-W continuations per request -> parent_idx.
    std::vector<std::vector<int>> parents(R);
    for (int r = 0; r < R; ++r) parents[r] = select_top_w(t, r);

    // 3.3) Reorder the cache and, in the reference, copy each parent's history to its children.
    for (int r = 0; r < R; ++r) {
      std::vector<int> beam_ids;
      for (Beam& b : beams_[r]) beam_ids.push_back(b.id);
      cache.reorder(beam_ids, parents[r]);
      std::vector<Beam> old = beams_[r];
      for (int i = 0; i < W; ++i) {
        beams_[r][i].k = old[parents[r][i]].k;
        beams_[r][i].v = old[parents[r][i]].v;
        beams_[r][i].score = new_scores_[r][i];
      }
    }

    try {
      cache.check_invariants();
    } catch (const std::logic_error& e) {
      fail(e.what());
    }
    std::string parent_str = "[";
    for (int i = 0; i < W; ++i) parent_str += (i ? " " : "") + std::to_string(parents[0][i]);
    parent_str += "]";
    std::printf("%5d %5d %6d %8d %10d %13.3e  %s\n", t, beams_[0][0].length(), copies,
                cache.num_used_pages(), unshared_pages(), step_err_, parent_str.c_str());
  }

  // Top-W of the W x W (parent, token) candidates of request r by cumulative pseudo
  // log-probability. Records the selection pattern and leaves the new scores in new_scores_.
  std::vector<int> select_top_w(int t, int r) {
    const int W = cfg_.beam_width;
    struct Cand {
      double score;
      int parent;
    };
    std::vector<Cand> cands;
    std::exponential_distribution<double> neg_logprob(1.0);
    // Every 5th step one parent dominates (all children from it) unless the config forces it
    // every step; the parent varies so different beams get dropped.
    const bool force_same = cfg_.all_same_parent || t % 5 == 4;
    const int favored = (t + r) % W;
    for (int b = 0; b < W; ++b) {
      for (int c = 0; c < W; ++c) {
        double s = beams_[r][b].score - neg_logprob(rng_);
        if (force_same && b == favored) s += 1e6;
        cands.push_back({s, b});
      }
    }
    std::stable_sort(cands.begin(), cands.end(),
                     [](const Cand& a, const Cand& b) { return a.score > b.score; });
    std::vector<int> parent_idx(W);
    new_scores_.resize(cfg_.num_requests);
    new_scores_[r].resize(W);
    std::vector<int> times_chosen(W, 0);
    for (int i = 0; i < W; ++i) {
      parent_idx[i] = cands[i].parent;
      new_scores_[r][i] = force_same ? beams_[r][cands[i].parent].score : cands[i].score;
      ++times_chosen[cands[i].parent];
    }
    if (r == 0) {  // count patterns once per step (request 0), plus any request below
      step_flags_ = {false, false, true};
    }
    const int max_chosen = *std::max_element(times_chosen.begin(), times_chosen.end());
    const int unchosen = int(std::count(times_chosen.begin(), times_chosen.end(), 0));
    step_flags_.shared |= max_chosen > 1;
    step_flags_.dropped |= unchosen > 0;
    step_flags_.all_same &= max_chosen == W;
    if (r == cfg_.num_requests - 1) {
      steps_with_shared_parent_ += step_flags_.shared;
      steps_with_dropped_parent_ += step_flags_.dropped;
      steps_all_same_ += step_flags_.all_same && W > 1;
    }
    return parent_idx;
  }

  int unshared_pages() const {
    int n = 0;
    for (auto& req : beams_) {
      for (const Beam& b : req) n += (b.length() + cfg_.page_size - 1) / cfg_.page_size;
    }
    return n;
  }

  std::vector<float> random(size_t n, float amp) {
    std::uniform_real_distribution<float> dist(-amp, amp);
    std::vector<float> x(n);
    for (float& e : x) e = __bfloat162float(__float2bfloat16(dist(rng_)));
    return x;
  }

  static void append(Beam& b, const std::vector<float>& k, const std::vector<float>& v,
                     int first_row, int rows) {
    b.k.insert(b.k.end(), k.begin() + size_t(first_row) * kKvRow,
               k.begin() + size_t(first_row + rows) * kKvRow);
    b.v.insert(b.v.end(), v.begin() + size_t(first_row) * kKvRow,
               v.begin() + size_t(first_row + rows) * kKvRow);
  }

  template <typename Op>
  std::vector<float> run_on_gpu(int rows, const std::vector<float>& q, const std::vector<float>& k,
                                const std::vector<float>& v, Op op) {
    DeviceBuffer<bf16> dq(q.size()), dk(k.size()), dv(v.size()), dout(size_t(rows) * kQoRow);
    auto upload = [](bf16* dst, const std::vector<float>& src) {
      std::vector<bf16> h(src.size());
      for (size_t i = 0; i < src.size(); ++i) h[i] = __float2bfloat16(src[i]);
      CUDA_CHECK(cudaMemcpy(dst, h.data(), h.size() * sizeof(bf16), cudaMemcpyHostToDevice));
    };
    upload(dq.ptr, q);
    upload(dk.ptr, k);
    upload(dv.ptr, v);
    CUDA_CHECK(cudaMemset(dout.ptr, 0xff, size_t(rows) * kQoRow * sizeof(bf16)));  // NaN poison
    op(dq.ptr, dk.ptr, dv.ptr, dout.ptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<bf16> h(size_t(rows) * kQoRow);
    CUDA_CHECK(cudaMemcpy(h.data(), dout.ptr, h.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
    std::vector<float> out(h.size());
    for (size_t i = 0; i < h.size(); ++i) out[i] = __bfloat162float(h[i]);
    return out;
  }

  void check(const float* q, const Beam& b, int p, const float* got) {
    float ref[kQoRow];
    attn_ref::reference_attention(q, b.k.data(), b.v.data(), b.length(), p, window_left_, 0, ref);
    bool ok = true;
    for (int i = 0; i < kQoRow; ++i) {
      if (!attn_ref::within_tolerance(got[i], ref[i])) ok = false;
      const float err = std::isnan(got[i]) ? INFINITY : std::fabs(got[i] - ref[i]);
      max_err_ = std::max(max_err_, err);
      step_err_ = std::max(step_err_, err);
    }
    ++rows_checked_;
    if (!ok) {
      if (rows_bad_ == 0) fail("attention output differs from the reference");
      ++rows_bad_;
    }
  }

  void fail(const std::string& msg) {
    std::printf("  ERROR: %s\n", msg.c_str());
    failed_ = true;
  }

  attn::AttentionWorkspace& ws_;
  const Config& cfg_;
  const int window_left_;
  std::mt19937 rng_;
  attn::PagedKVCache* cache_ = nullptr;
  std::vector<std::vector<Beam>> beams_;  // [request][beam]
  std::vector<std::vector<double>> new_scores_;
  struct {
    bool shared, dropped, all_same;
  } step_flags_{};
  int steps_with_shared_parent_ = 0, steps_with_dropped_parent_ = 0, steps_all_same_ = 0;
  int total_copies_ = 0;
  long rows_checked_ = 0, rows_bad_ = 0;
  float max_err_ = 0, step_err_ = 0;
  bool guard_checked_ = false;
  bool failed_ = false;
};

}  // namespace

int main() {
  const std::vector<Config> configs = {
      {"basic", 3, 4, 20, 16, {37, 16, 5}, false},
      {"W=1", 2, 1, 20, 16, {21, 8}, false},
      {"all beams from one parent", 2, 4, 12, 4, {10, 3}, true},
      {"last page full at fork", 2, 4, 10, 16, {16, 32}, false},
      {"page_size=1", 2, 3, 20, 1, {7, 12}, false},
  };
  attn::AttentionWorkspace ws;
  int passed = 0, failed = 0;
  std::vector<std::string> failures;
  for (const Config& cfg : configs) {
    for (int window_left : {kNoWindow, 12}) {
      Runner runner(ws, cfg, window_left);
      bool ok = false;
      try {
        ok = runner.run();
      } catch (const std::exception& e) {
        std::printf("  EXCEPTION: %s\n", e.what());
      }
      (ok ? passed : failed)++;
      if (!ok) failures.push_back(cfg.name + " window_left=" + std::to_string(window_left));
    }
  }
  std::printf("\n%d passed, %d failed\n", passed, failed);
  for (const auto& f : failures) std::printf("  FAILED: %s\n", f.c_str());
  return failed == 0 ? 0 : 1;
}
