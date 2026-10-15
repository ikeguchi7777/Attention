// Correctness tests: compares prefill()/decode() against an fp32 CPU reference of
// softmax(q k^T / sqrt(d)) v with causal + sliding-window masking.
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "attention.h"
#include "ref_attention.h"

using attn::bf16;
using attn::kHeadDim;
using attn::kNoWindow;
using attn::kNumKvHeads;
using attn::kNumQoHeads;

namespace {

using attn_ref::kAtol;
using attn_ref::kRtol;
constexpr int kQoRow = kNumQoHeads * kHeadDim;
constexpr int kKvRow = kNumKvHeads * kHeadDim;

template <typename T>
struct DeviceBuffer {
  T* ptr = nullptr;
  explicit DeviceBuffer(size_t n) { CUDA_CHECK(cudaMalloc(&ptr, std::max<size_t>(n, 1) * sizeof(T))); }
  ~DeviceBuffer() { cudaFree(ptr); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

std::vector<bf16> to_bf16(const std::vector<float>& x) {
  std::vector<bf16> y(x.size());
  for (size_t i = 0; i < x.size(); ++i) y[i] = __float2bfloat16(x[i]);
  return y;
}

// One sequence as the test sees it: its id in the cache and every K/V row appended so far
// (already rounded to bf16, stored as float).
struct Seq {
  int id = -1;
  std::vector<float> k, v;  // [length, kNumKvHeads, kHeadDim]
  int length() const { return int(k.size() / kKvRow); }
};

// fp32 reference for one query token at absolute position p of `s` (see ref_attention.h).
void reference(const float* q, const Seq& s, int p, int window_left, int window_right,
               float* out) {
  attn_ref::reference_attention(q, s.k.data(), s.v.data(), s.length(), p, window_left,
                                window_right, out);
}

struct CaseResult {
  long checked = 0;
  long failed = 0;
  float max_abs_err = 0.f;
};

class Harness {
 public:
  Harness(attn::AttentionWorkspace& ws, attn::PagedKVCache& cache, unsigned seed)
      : ws_(ws), cache_(cache), rng_(seed) {}

  Seq new_seq() {
    Seq s;
    s.id = cache_.add_sequence();
    return s;
  }

  void free_seq(Seq& s) {
    cache_.free_sequence(s.id);
    s = Seq{};
  }

  // Prefills lens[i] new tokens onto batch[i] and checks every output row.
  void prefill(std::vector<Seq*> batch, const std::vector<int>& lens, int window_left) {
    std::vector<int32_t> qo_indptr{0};
    for (int n : lens) qo_indptr.push_back(qo_indptr.back() + n);
    const int nnz = qo_indptr.back();
    std::vector<float> q = random(size_t(nnz) * kQoRow, 2.f);
    std::vector<float> k = random(size_t(nnz) * kKvRow, 2.f);
    std::vector<float> v = random(size_t(nnz) * kKvRow, 1.f);
    std::vector<int> ids, old_len;
    for (size_t b = 0; b < batch.size(); ++b) {
      ids.push_back(batch[b]->id);
      old_len.push_back(batch[b]->length());
      append_history(*batch[b], k, v, qo_indptr[b], lens[b]);
    }
    std::vector<float> out = run(nnz, q, k, v, [&](const bf16* dq, const bf16* dk, const bf16* dv,
                                                     bf16* dout) {
      attn::prefill(ws_, cache_, ids, dq, dk, dv, qo_indptr, window_left, dout);
    });
    for (size_t b = 0; b < batch.size(); ++b) {
      for (int i = 0; i < lens[b]; ++i) {
        const size_t row = size_t(qo_indptr[b] + i);
        check(q.data() + row * kQoRow, *batch[b], old_len[b] + i, window_left, 0,
              out.data() + row * kQoRow);
      }
    }
  }

  // One decode step for every sequence in batch.
  void decode(std::vector<Seq*> batch, int window_left) {
    const int B = int(batch.size());
    std::vector<float> q = random(size_t(B) * kQoRow, 2.f);
    std::vector<float> k = random(size_t(B) * kKvRow, 2.f);
    std::vector<float> v = random(size_t(B) * kKvRow, 1.f);
    std::vector<int> ids;
    for (int b = 0; b < B; ++b) {
      ids.push_back(batch[b]->id);
      append_history(*batch[b], k, v, b, 1);
    }
    std::vector<float> out = run(B, q, k, v, [&](const bf16* dq, const bf16* dk, const bf16* dv,
                                                   bf16* dout) {
      attn::decode(ws_, cache_, ids, dq, dk, dv, window_left, dout);
    });
    for (int b = 0; b < B; ++b) {
      check(q.data() + size_t(b) * kQoRow, *batch[b], batch[b]->length() - 1, window_left, 0,
            out.data() + size_t(b) * kQoRow);
    }
  }

  // Streaming step: appends new_tokens[i] tokens to batch[i] and computes every query whose
  // right context became complete (positions up to length - 1 - window_right).
  void stream_prefill(std::vector<Seq*> batch, const std::vector<int>& new_tokens,
                      int window_left, int window_right) {
    const int B = int(batch.size());
    std::vector<int32_t> qo_indptr{0}, kv_indptr{0};
    std::vector<int> first_q(B);
    for (int b = 0; b < B; ++b) {
      const int old_len = batch[b]->length(), new_len = old_len + new_tokens[b];
      const int done = std::max(0, old_len - window_right);
      const int ready = std::max(0, new_len - window_right);
      first_q[b] = done;
      qo_indptr.push_back(qo_indptr.back() + (ready - done));
      kv_indptr.push_back(kv_indptr.back() + new_tokens[b]);
    }
    std::vector<float> q = random(size_t(qo_indptr.back()) * kQoRow, 2.f);
    std::vector<float> k = random(size_t(kv_indptr.back()) * kKvRow, 2.f);
    std::vector<float> v = random(size_t(kv_indptr.back()) * kKvRow, 1.f);
    std::vector<int> ids;
    for (int b = 0; b < B; ++b) {
      ids.push_back(batch[b]->id);
      append_history(*batch[b], k, v, kv_indptr[b], new_tokens[b]);
    }
    std::vector<float> out =
        run(qo_indptr.back(), q, k, v, [&](const bf16* dq, const bf16* dk, const bf16* dv,
                                           bf16* dout) {
          attn::prefill(ws_, cache_, ids, dq, qo_indptr, dk, dv, kv_indptr, window_left,
                        window_right, dout);
        });
    for (int b = 0; b < B; ++b) {
      for (int i = 0; i < qo_indptr[b + 1] - qo_indptr[b]; ++i) {
        const size_t row = size_t(qo_indptr[b] + i);
        check(q.data() + row * kQoRow, *batch[b], first_q[b] + i, window_left, window_right,
              out.data() + row * kQoRow);
      }
    }
  }

  // Streaming decode step: one token per sequence; checks the query at length - 1 - window_right.
  void stream_decode(std::vector<Seq*> batch, int window_left, int window_right) {
    const int B = int(batch.size());
    std::vector<float> q = random(size_t(B) * kQoRow, 2.f);
    std::vector<float> k = random(size_t(B) * kKvRow, 2.f);
    std::vector<float> v = random(size_t(B) * kKvRow, 1.f);
    std::vector<int> ids;
    for (int b = 0; b < B; ++b) {
      ids.push_back(batch[b]->id);
      append_history(*batch[b], k, v, b, 1);
    }
    std::vector<float> out = run(B, q, k, v, [&](const bf16* dq, const bf16* dk, const bf16* dv,
                                                   bf16* dout) {
      attn::decode(ws_, cache_, ids, dq, dk, dv, window_left, window_right, dout);
    });
    for (int b = 0; b < B; ++b) {
      check(q.data() + size_t(b) * kQoRow, *batch[b], batch[b]->length() - 1 - window_right,
            window_left, window_right, out.data() + size_t(b) * kQoRow);
    }
  }

  void truncate(Seq& s, int new_length) {
    cache_.truncate(s.id, new_length);
    s.k.resize(size_t(new_length) * kKvRow);
    s.v.resize(size_t(new_length) * kKvRow);
  }

  CaseResult take_result() {
    CaseResult r = result_;
    result_ = CaseResult{};
    return r;
  }

 private:
  // Random values rounded to bf16 (returned as float so the reference sees the exact inputs).
  std::vector<float> random(size_t n, float amp) {
    std::uniform_real_distribution<float> dist(-amp, amp);
    std::vector<float> x(n);
    for (float& e : x) e = __bfloat162float(__float2bfloat16(dist(rng_)));
    return x;
  }

  static void append_history(Seq& s, const std::vector<float>& k, const std::vector<float>& v,
                             int first_row, int rows) {
    s.k.insert(s.k.end(), k.begin() + size_t(first_row) * kKvRow,
               k.begin() + size_t(first_row + rows) * kKvRow);
    s.v.insert(s.v.end(), v.begin() + size_t(first_row) * kKvRow,
               v.begin() + size_t(first_row + rows) * kKvRow);
  }

  // Uploads q/k/v, runs `op`, returns the output as float.
  template <typename Op>
  std::vector<float> run(int rows, const std::vector<float>& q, const std::vector<float>& k,
                         const std::vector<float>& v, Op op) {
    DeviceBuffer<bf16> dq(q.size()), dk(k.size()), dv(v.size()), dout(size_t(rows) * kQoRow);
    auto hq = to_bf16(q), hk = to_bf16(k), hv = to_bf16(v);
    CUDA_CHECK(cudaMemcpy(dq.ptr, hq.data(), hq.size() * sizeof(bf16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dk.ptr, hk.data(), hk.size() * sizeof(bf16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv.ptr, hv.data(), hv.size() * sizeof(bf16), cudaMemcpyHostToDevice));
    // Poison the output so rows the kernel fails to write cannot pass by accident.
    CUDA_CHECK(cudaMemset(dout.ptr, 0xff, size_t(rows) * kQoRow * sizeof(bf16)));
    op(dq.ptr, dk.ptr, dv.ptr, dout.ptr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<bf16> hout(size_t(rows) * kQoRow);
    CUDA_CHECK(cudaMemcpy(hout.data(), dout.ptr, hout.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
    std::vector<float> out(hout.size());
    for (size_t i = 0; i < hout.size(); ++i) out[i] = __bfloat162float(hout[i]);
    return out;
  }

  void check(const float* q, const Seq& s, int p, int window_left, int window_right,
             const float* got) {
    float ref[kQoRow];
    reference(q, s, p, window_left, window_right, ref);
    bool ok = true;
    for (int i = 0; i < kQoRow; ++i) {
      const float err = std::fabs(got[i] - ref[i]);
      if (!attn_ref::within_tolerance(got[i], ref[i])) ok = false;
      if (!(err <= result_.max_abs_err)) result_.max_abs_err = std::isnan(err) ? INFINITY : err;
    }
    ++result_.checked;
    if (!ok) ++result_.failed;
  }

  attn::AttentionWorkspace& ws_;
  attn::PagedKVCache& cache_;
  std::mt19937 rng_;
  CaseResult result_;
};

std::string window_name(int w) { return w < 0 ? "off" : std::to_string(w); }

struct Totals {
  int passed = 0, failed = 0;
};

void report(Totals& totals, const std::string& name, bool ok, const std::string& detail) {
  std::printf("[%s] %-72s %s\n", ok ? "PASS" : "FAIL", name.c_str(), detail.c_str());
  std::fflush(stdout);
  (ok ? totals.passed : totals.failed)++;
}

// Runs one numerical case on a fresh cache; `body` drives the harness.
void numeric_case(Totals& totals, attn::AttentionWorkspace& ws, const std::string& name,
                  int page_size, int max_pages, const std::function<void(Harness&)>& body) {
  bool ok;
  std::string detail;
  try {
    attn::PagedKVCache cache(page_size, max_pages);
    Harness h(ws, cache, std::hash<std::string>{}(name));
    body(h);
    CaseResult r = h.take_result();
    ok = r.failed == 0 && r.checked > 0;
    char buf[160];
    std::snprintf(buf, sizeof(buf), "max_abs_err=%.3e  rows=%ld  bad_rows=%ld", r.max_abs_err,
                  r.checked, r.failed);
    detail = buf;
  } catch (const std::exception& e) {
    ok = false;
    detail = std::string("exception: ") + e.what();
  }
  report(totals, name, ok, detail);
}

// Expects `body` to throw an exception of type E.
template <typename E>
void error_case(Totals& totals, const std::string& name, const std::function<void()>& body,
                const std::function<bool()>& postcondition = [] { return true; }) {
  bool threw = false;
  std::string what;
  try {
    body();
  } catch (const E& e) {
    threw = true;
    what = e.what();
  } catch (const std::exception& e) {
    what = std::string("wrong exception type: ") + e.what();
  }
  const bool ok = threw && postcondition();
  report(totals, name, ok, threw ? "threw as expected" : what.empty() ? "did not throw" : what);
}

// Lengths that hit page boundaries for page size p, plus length 1.
std::vector<int> boundary_lengths(int p, int count) {
  const int base[] = {1, p > 1 ? p - 1 : 2, p, p + 1, 2 * p + 1, 3 * p - 1, 37, 129,
                      5 * p,  250,           1,  64,    65,        17,        200, 2};
  std::vector<int> lens;
  for (int i = 0; i < count; ++i) lens.push_back(base[i % 16]);
  return lens;
}

}  // namespace

int main() {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  std::printf("GPU: %s (sm_%d%d)   tolerance: atol=%.0e rtol=%.0e\n\n", prop.name, prop.major,
              prop.minor, kAtol, kRtol);

  attn::AttentionWorkspace ws;  // shared by every case, as in a serving loop
  Totals totals;

  // 1) prefill -> decode, all batch sizes x page sizes x windows.
  //    Ragged lengths around page boundaries, including 1; decode re-checks every step against a
  //    reference recomputed over the full history.
  for (int page_size : {1, 2, 4, 8, 16, 32}) {
    for (int batch : {1, 4, 16}) {
      for (int window : {kNoWindow, 0, 1, 16, 100, 100000}) {
        const std::string name = "prefill+decode  page=" + std::to_string(page_size) +
                                 " batch=" + std::to_string(batch) + " window_left=" +
                                 window_name(window);
        numeric_case(totals, ws, name, page_size, 1 << 14, [&](Harness& h) {
          std::vector<Seq> seqs(batch);
          std::vector<Seq*> ptrs;
          for (auto& s : seqs) {
            s = h.new_seq();
            ptrs.push_back(&s);
          }
          std::vector<int> lens = boundary_lengths(page_size, batch);
          if (batch == 1) lens = {page_size + 1};
          h.prefill(ptrs, lens, window);
          for (int step = 0; step < 2 * page_size + 3; ++step) h.decode(ptrs, window);
        });
      }
    }
  }

  // 2) Window boundaries relative to the sequence length L: window_left = L-2, L-1 (exactly the
  //    whole sequence for the last query), L, and far beyond.
  for (int page_size : {1, 16}) {
    const int L = 3 * page_size + 5;
    for (int window : {L - 2, L - 1, L, 10 * L}) {
      const std::string name = "window vs length  page=" + std::to_string(page_size) +
                               " L=" + std::to_string(L) + " window_left=" + window_name(window);
      numeric_case(totals, ws, name, page_size, 4096, [&](Harness& h) {
        Seq s = h.new_seq();
        h.prefill({&s}, {L}, window);
        for (int step = 0; step < 4; ++step) h.decode({&s}, window);
      });
    }
  }

  // 3) Long sequences: exercise FlashInfer's split-KV paths (few CTAs per request) for both
  //    decode and a short chunked prefill on top of a long prefix.
  for (int page_size : {1, 16}) {
    for (int window : {kNoWindow, 0, 255, 1000}) {
      const std::string name = "long + chunked prefill  page=" + std::to_string(page_size) +
                               " window_left=" + window_name(window);
      numeric_case(totals, ws, name, page_size, 1 << 13, [&](Harness& h) {
        Seq a = h.new_seq(), b = h.new_seq();
        h.prefill({&a, &b}, {1500, 3}, window);
        h.prefill({&a, &b}, {5, 1200}, window);  // continue both from their existing prefixes
        for (int step = 0; step < 3; ++step) h.decode({&a, &b}, window);
        h.decode({&a}, window);  // sub-batch
      });
    }
  }

  // 4) Cache management: freeing / re-adding sequences reuses (fragmented) pages, sequences join
  //    mid-stream, truncate rolls a sequence back.
  numeric_case(totals, ws, "free/add/truncate with page reuse  page=4", 4, 512, [&](Harness& h) {
    std::vector<Seq> s(6);
    for (auto& x : s) x = h.new_seq();
    h.prefill({&s[0], &s[1], &s[2], &s[3]}, {9, 1, 4, 30}, kNoWindow);
    for (int i = 0; i < 5; ++i) h.decode({&s[0], &s[1], &s[2], &s[3]}, kNoWindow);
    h.free_seq(s[1]);
    h.free_seq(s[3]);
    s[1] = h.new_seq();
    h.prefill({&s[4], &s[1]}, {23, 6}, 7);  // new sequences land on recycled pages
    for (int i = 0; i < 6; ++i) h.decode({&s[0], &s[1], &s[2], &s[4]}, 7);
    h.truncate(s[0], 10);
    h.truncate(s[4], 5);
    for (int i = 0; i < 6; ++i) h.decode({&s[4], &s[0], &s[2]}, kNoWindow);
  });

  // 5) Streaming with look-ahead (window_right): chunks of uneven size arrive (including empty
  //    chunks and chunks shorter than window_right, so some steps have no computable query), each
  //    step computes exactly the queries whose right context became complete, then streaming
  //    decode continues. The reference applies [p - window_left, p + window_right] explicitly.
  {
    struct Win {
      int left, right;
    };
    const Win wins[] = {{kNoWindow, 0}, {kNoWindow, 1}, {kNoWindow, 7}, {0, 0}, {0, 1},
                        {3, 5},         {16, 33},       {50, 20}};
    for (int page_size : {1, 16}) {
      for (int batch : {1, 4}) {
        for (const Win& w : wins) {
          const std::string name = "streaming  page=" + std::to_string(page_size) +
                                   " batch=" + std::to_string(batch) + " window_left=" +
                                   window_name(w.left) + " window_right=" + std::to_string(w.right);
          numeric_case(totals, ws, name, page_size, 1 << 14, [&](Harness& h) {
            std::vector<Seq> seqs(batch);
            std::vector<Seq*> ptrs;
            for (auto& x : seqs) {
              x = h.new_seq();
              ptrs.push_back(&x);
            }
            const int R = w.right;
            const int sizes[] = {0, 1, std::max(R - 1, 0), R, R + 1, 2, 17, 40, 3};
            for (int step = 0; step < 7; ++step) {
              std::vector<int> chunk(batch);
              for (int b = 0; b < batch; ++b) chunk[b] = sizes[(step * 3 + b * 5) % 9];
              h.stream_prefill(ptrs, chunk, w.left, R);
            }
            std::vector<int> top_up(batch);  // make every stream long enough to decode
            for (int b = 0; b < batch; ++b) top_up[b] = std::max(0, R - seqs[b].length());
            h.stream_prefill(ptrs, top_up, w.left, R);
            for (int step = 0; step < 2 * R + 3; ++step) h.stream_decode(ptrs, w.left, R);
            h.stream_prefill(ptrs, std::vector<int>(batch, 5), w.left, R);
            h.stream_decode(ptrs, w.left, R);
          });
        }
      }
    }
    // Long streams: FlashInfer's split-KV paths with a delayed, windowed query block.
    for (int page_size : {1, 16}) {
      const std::string name = "streaming long  page=" + std::to_string(page_size) +
                               " window_left=300 window_right=64";
      numeric_case(totals, ws, name, page_size, 1 << 13, [&](Harness& h) {
        Seq a = h.new_seq(), b = h.new_seq();
        h.stream_prefill({&a, &b}, {1500, 70}, 300, 64);
        h.stream_prefill({&a, &b}, {5, 1200}, 300, 64);
        for (int step = 0; step < 3; ++step) h.stream_decode({&a, &b}, 300, 64);
      });
    }
  }

  // 6) Error handling.
  {
    attn::PagedKVCache cache(4, 4);
    int id = cache.add_sequence();
    DeviceBuffer<bf16> buf(size_t(32) * kQoRow);
    error_case<std::runtime_error>(totals,
        "error: out of pages leaves cache untouched",
        [&] {
          attn::prefill(ws, cache, {id}, buf.ptr, buf.ptr, buf.ptr, {0, 17}, kNoWindow, buf.ptr);
        },
        [&] { return cache.length(id) == 0 && cache.num_free_pages() == 4; });
    error_case<std::invalid_argument>(totals, "error: duplicate sequence in batch", [&] {
      attn::decode(ws, cache, {id, id}, buf.ptr, buf.ptr, buf.ptr, kNoWindow, buf.ptr);
    });
    error_case<std::invalid_argument>(totals, "error: unknown sequence id", [&] {
      attn::decode(ws, cache, {id + 7}, buf.ptr, buf.ptr, buf.ptr, kNoWindow, buf.ptr);
    });
    error_case<std::invalid_argument>(totals, "error: empty prefill row", [&] {
      attn::prefill(ws, cache, {id}, buf.ptr, buf.ptr, buf.ptr, {0, 0}, kNoWindow, buf.ptr);
    });
    error_case<std::invalid_argument>(totals, "error: negative window_right", [&] {
      attn::decode(ws, cache, {id}, buf.ptr, buf.ptr, buf.ptr, kNoWindow, -1, buf.ptr);
    });
    error_case<std::invalid_argument>(
        totals, "error: streaming decode before window_right tokens exist",
        [&] { attn::decode(ws, cache, {id}, buf.ptr, buf.ptr, buf.ptr, kNoWindow, 3, buf.ptr); },
        [&] { return cache.length(id) == 0; });
    error_case<std::invalid_argument>(
        totals, "error: streaming prefill with a query lacking right context",
        [&] {
          // 4 new tokens, window_right 2 -> only 2 computable queries, 3 requested
          attn::prefill(ws, cache, {id}, buf.ptr, {0, 3}, buf.ptr, buf.ptr, {0, 4}, kNoWindow, 2,
                        buf.ptr);
        },
        [&] { return cache.length(id) == 0; });
    error_case<std::invalid_argument>(totals, "error: run_planned after the cache changed", [&] {
      attn::plan_decode(ws, cache, {id}, kNoWindow, 0);
      cache.add_sequence();
      attn::run_planned(ws, buf.ptr, buf.ptr, buf.ptr, buf.ptr);
    });
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  std::printf("\n%d passed, %d failed\n", totals.passed, totals.failed);
  return totals.failed == 0 ? 0 : 1;
}
