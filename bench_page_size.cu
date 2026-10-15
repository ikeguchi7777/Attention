// Compares page sizes for prefill and decode.
//
// For each (phase, batch, kv_len) and page size it reports
//   gpu      : GPU time of run_planned() (K/V append + attention) with a warm L2. Each sample
//              times a batch of back-to-back runs and divides (CUDA events only resolve ~0.5 us,
//              too coarse for decode steps of a few us); median over samples.
//   gpu_cold : the same with L2 flushed before every run (one run per sample; median). Shows the
//              DRAM-bound cost when the KV of this layer is not already cached.
//   e2e      : wall time of a full prefill()/decode() call including the host-side plan, followed
//              by a stream synchronize (median).
// While a sample's work is being enqueued the stream is held by a spin kernel, so GPU timings
// exclude host launch gaps. The page size is chosen on `gpu`: the smallest page size within 5% of
// the fastest one in every condition (falling back to the best geometric-mean ratio otherwise).
//
// Usage: bench_page_size [window_left]   (default: no sliding window)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "attention.h"

using attn::bf16;
using attn::kHeadDim;
using attn::kNoWindow;
using attn::kNumKvHeads;
using attn::kNumQoHeads;

namespace {

constexpr int kWarmup = 5;
constexpr int kSamples = 30;
constexpr int kE2eIters = 15;
constexpr int kMaxRepsPerSample = 100;    // keeps queued launches far below the launch-queue depth
constexpr double kTargetSampleMs = 1.0;
constexpr size_t kFlushBytes = size_t(128) << 20;  // > 4x the 32 MiB L2 of the RTX 4060 Ti
constexpr double kTolerance = 1.05;

__global__ void fill_random(bf16* x, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x) {
    uint32_t h = uint32_t(i) * 2654435761u ^ seed;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    x[i] = __float2bfloat16(float(h & 0xffff) / 32768.f - 1.f);
  }
}

// Spins until the host sets *flag, so the work queued behind it runs back to back.
__global__ void wait_for_flag(volatile int* flag) {
  while (*flag == 0) {
  }
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

struct DeviceTensors {
  bf16 *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr;
  explicit DeviceTensors(size_t rows) {
    const size_t qn = rows * kNumQoHeads * kHeadDim, kn = rows * kNumKvHeads * kHeadDim;
    CUDA_CHECK(cudaMalloc(&q, qn * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&k, kn * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&v, kn * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&o, qn * sizeof(bf16)));
    fill_random<<<1024, 256>>>(q, qn, 1);
    fill_random<<<1024, 256>>>(k, kn, 2);
    fill_random<<<1024, 256>>>(v, kn, 3);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  ~DeviceTensors() {
    cudaFree(q);
    cudaFree(k);
    cudaFree(v);
    cudaFree(o);
  }
};

class Bench {
 public:
  explicit Bench(cudaStream_t stream) : stream_(stream) {
    CUDA_CHECK(cudaHostAlloc(&flag_, sizeof(int), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&flag_dev_, flag_, 0));
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
    CUDA_CHECK(cudaMalloc(&flush_, kFlushBytes));
  }
  ~Bench() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
    cudaFreeHost(flag_);
    cudaFree(flush_);
  }

  // Warm-L2 GPU time (ms) of one run_planned() for the workspace's current plan.
  double gpu_ms(attn::AttentionWorkspace& ws, const DeviceTensors& t) {
    for (int i = 0; i < kWarmup; ++i) attn::run_planned(ws, t.q, t.k, t.v, t.o);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    const double estimate = time_blocked(ws, t, 1, /*flush=*/false);
    const int reps =
        int(std::clamp(kTargetSampleMs / std::max(estimate, 1e-4), 1.0, double(kMaxRepsPerSample)));
    std::vector<double> ms(kSamples);
    for (double& m : ms) m = time_blocked(ws, t, reps, /*flush=*/false) / reps;
    return median(ms);
  }

  // Cold-L2 GPU time (ms) of one run_planned().
  double gpu_cold_ms(attn::AttentionWorkspace& ws, const DeviceTensors& t) {
    std::vector<double> ms(kSamples);
    for (double& m : ms) m = time_blocked(ws, t, 1, /*flush=*/true);
    return median(ms);
  }

  // Median wall time (ms) of `step` after `reset` (which restores the cache for the next call),
  // synchronizing the stream after each call.
  template <typename Step, typename Reset>
  double e2e_ms(Step step, Reset reset) {
    std::vector<double> ms;
    for (int i = 0; i < kWarmup + kE2eIters; ++i) {
      reset();
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      const auto t0 = std::chrono::steady_clock::now();
      step();
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      const auto t1 = std::chrono::steady_clock::now();
      if (i >= kWarmup) ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return median(ms);
  }

 private:
  // Holds the stream with a spin kernel while [L2 flush,] start event, `reps` runs and stop event
  // are enqueued, then releases it. Returns the time between the events in ms.
  double time_blocked(attn::AttentionWorkspace& ws, const DeviceTensors& t, int reps, bool flush) {
    *reinterpret_cast<volatile int*>(flag_) = 0;
    wait_for_flag<<<1, 1, 0, stream_>>>(flag_dev_);
    CUDA_CHECK(cudaGetLastError());
    if (flush) CUDA_CHECK(cudaMemsetAsync(flush_, ++flush_value_ & 0xff, kFlushBytes, stream_));
    CUDA_CHECK(cudaEventRecord(start_, stream_));
    for (int r = 0; r < reps; ++r) attn::run_planned(ws, t.q, t.k, t.v, t.o);
    CUDA_CHECK(cudaEventRecord(stop_, stream_));
    *reinterpret_cast<volatile int*>(flag_) = 1;
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }

  cudaStream_t stream_;
  int* flag_ = nullptr;
  int* flag_dev_ = nullptr;
  cudaEvent_t start_ = nullptr, stop_ = nullptr;
  void* flush_ = nullptr;
  int flush_value_ = 0;
};

struct Condition {
  std::string phase;
  int batch;
  int kv_len;
  std::string label() const {
    return phase + " b=" + std::to_string(batch) + " kv=" + std::to_string(kv_len);
  }
};

struct Result {
  double gpu = 0, gpu_cold = 0, e2e = 0;
};

}  // namespace

int main(int argc, char** argv) {
  const int window_left = argc > 1 ? std::atoi(argv[1]) : kNoWindow;
  const std::vector<int> page_sizes = {1, 2, 4, 8, 16, 32};
  const std::vector<int> batches = {1, 8, 32};
  const std::vector<int> kv_lens = {512, 4096};

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s, %d SMs, L2 %d MiB | window_left=%d | samples=%d e2e_iters=%d\n\n",
              prop.name, prop.multiProcessorCount, prop.l2CacheSize >> 20, window_left, kSamples,
              kE2eIters);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  Bench bench(stream);
  attn::AttentionWorkspace ws;

  std::vector<Condition> conds;
  for (const char* phase : {"prefill", "decode"}) {
    for (int b : batches) {
      for (int L : kv_lens) conds.push_back({phase, b, L});
    }
  }
  std::map<std::string, std::map<int, Result>> results;

  // The GPU idles at a low clock; keep it busy for ~1 s so the first condition is not measured
  // while clocks are still ramping up.
  {
    const size_t n = size_t(64) << 20;  // 128 MiB of bf16
    bf16* warm = nullptr;
    CUDA_CHECK(cudaMalloc(&warm, n * sizeof(bf16)));
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(1)) {
      fill_random<<<1024, 256, 0, stream>>>(warm, n, 0);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaFree(warm));
  }

  for (const Condition& c : conds) {
    const bool is_prefill = c.phase == "prefill";
    // Prefill feeds kv_len tokens per sequence; decode adds one token to kv_len cached ones.
    DeviceTensors t(size_t(c.batch) * (is_prefill ? c.kv_len : 1));
    std::vector<int32_t> qo_indptr{0};
    for (int b = 0; b < c.batch; ++b) qo_indptr.push_back(qo_indptr.back() + c.kv_len);
    for (int P : page_sizes) {
      attn::PagedKVCache cache(P, c.batch * ((c.kv_len + 1 + P - 1) / P));
      std::vector<int> ids;
      for (int b = 0; b < c.batch; ++b) ids.push_back(cache.add_sequence());
      Result r;
      if (is_prefill) {
        attn::plan_prefill(ws, cache, ids, qo_indptr, qo_indptr, window_left,
                           /*window_right=*/0, stream);
        r.gpu = bench.gpu_ms(ws, t);
        r.gpu_cold = bench.gpu_cold_ms(ws, t);
        r.e2e = bench.e2e_ms(
            [&] { attn::prefill(ws, cache, ids, t.q, t.k, t.v, qo_indptr, window_left, t.o, stream); },
            [&] {
              for (int id : ids) cache.truncate(id, 0);
            });
      } else {
        {  // Fill the cache with kv_len tokens per sequence (contents do not matter for timing).
          DeviceTensors fill(size_t(c.batch) * c.kv_len);
          attn::prefill(ws, cache, ids, fill.q, fill.k, fill.v, qo_indptr, kNoWindow, fill.o,
                        stream);
          CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        attn::plan_decode(ws, cache, ids, window_left, /*window_right=*/0, stream);
        r.gpu = bench.gpu_ms(ws, t);
        r.gpu_cold = bench.gpu_cold_ms(ws, t);
        r.e2e = bench.e2e_ms(
            [&] { attn::decode(ws, cache, ids, t.q, t.k, t.v, window_left, t.o, stream); },
            [&] {
              for (int id : ids) cache.truncate(id, c.kv_len);
            });
      }
      results[c.label()][P] = r;
      std::printf("  %-22s page=%-2d  gpu=%9.4f ms  gpu_cold=%9.4f ms  e2e=%9.4f ms\n",
                  c.label().c_str(), P, r.gpu, r.gpu_cold, r.e2e);
      std::fflush(stdout);
    }
  }

  // Markdown tables.
  auto print_table = [&](const char* title, double Result::*metric) {
    std::printf("\n### %s (ms, median; ratio to fastest in parentheses)\n\n| condition |", title);
    for (int P : page_sizes) std::printf(" P=%d |", P);
    std::printf("\n|---|");
    for (size_t i = 0; i < page_sizes.size(); ++i) std::printf("---|");
    std::printf("\n");
    for (const Condition& c : conds) {
      const auto& row = results[c.label()];
      double best = 1e30;
      for (int P : page_sizes) best = std::min(best, row.at(P).*metric);
      std::printf("| %s |", c.label().c_str());
      for (int P : page_sizes) std::printf(" %.4f (%.2f) |", row.at(P).*metric, row.at(P).*metric / best);
      std::printf("\n");
    }
  };
  print_table("gpu: append + attention, warm L2 (decision metric)", &Result::gpu);
  print_table("gpu_cold: append + attention, L2 flushed", &Result::gpu_cold);
  print_table("e2e: prefill()/decode() incl. plan + sync", &Result::e2e);

  // Decision on the warm-L2 GPU time.
  std::printf("\n### Decision (gpu, tolerance %.0f%%)\n\n", (kTolerance - 1) * 100);
  std::printf("| page size | worst ratio | geomean ratio | worst ratio (gpu_cold) |\n|---|---|---|---|\n");
  int chosen = -1, best_geo_p = -1;
  double best_geo = 1e30;
  for (int P : page_sizes) {
    double worst = 0, worst_cold = 0, log_sum = 0;
    for (const Condition& c : conds) {
      const auto& row = results[c.label()];
      double best = 1e30, best_cold = 1e30;
      for (int Q : page_sizes) {
        best = std::min(best, row.at(Q).gpu);
        best_cold = std::min(best_cold, row.at(Q).gpu_cold);
      }
      worst = std::max(worst, row.at(P).gpu / best);
      worst_cold = std::max(worst_cold, row.at(P).gpu_cold / best_cold);
      log_sum += std::log(row.at(P).gpu / best);
    }
    const double geo = std::exp(log_sum / conds.size());
    std::printf("| %d | %.3f | %.3f | %.3f |\n", P, worst, geo, worst_cold);
    if (chosen < 0 && worst <= kTolerance) chosen = P;
    if (geo < best_geo) {
      best_geo = geo;
      best_geo_p = P;
    }
  }
  if (chosen > 0) {
    std::printf("\nChosen page_size = %d (smallest within %.0f%% of the fastest in every condition)\n",
                chosen, (kTolerance - 1) * 100);
  } else {
    std::printf("\nNo page size is within %.0f%% everywhere; best geometric mean: page_size = %d\n",
                (kTolerance - 1) * 100, best_geo_p);
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
