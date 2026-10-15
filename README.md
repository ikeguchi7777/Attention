# FlashInfer (C++/CUDA) によるページ KV キャッシュ Attention

FlashInfer のヘッダオンリー C++ API を直接使い、ページ方式の KV キャッシュ上で
バッチ prefill → decode のインクリメンタル推論を行う単体プログラムです。
Python / PyTorch / JIT は使いません。

- 形状: `num_qo_heads = 8`, `num_kv_heads = 8`(MHA), `head_dim = 64`。dtype は Q/K/V/出力とも bf16
- バッチサイズ 1 以上、シーケンス長はバラバラでよい(ragged)
- Sliding Window Attention: `window_left` 引数で指定(下記)。`kNoWindow`(-1)で無効
- ストリーミング用の右ウィンドウ(先読み)`window_right` にも対応(下記「ストリーミング」)
- prefill と decode は別関数。plan はステップごとに 1 回、workspace は使い回す
- softmax のスケール: `1/sqrt(head_dim)`

API 調査のメモは [API_NOTES.md](API_NOTES.md) にあります。

## ファイル構成

| ファイル | 内容 |
|---|---|
| `attention.h` | 公開 API(`PagedKVCache`, `AttentionWorkspace`, `prefill`, `decode`, `CUDA_CHECK`)。FlashInfer のヘッダは含まない |
| `attention.cu` | ページ表の構築と転送、plan(`PrefillPlan` / `DecodePlan`)、K/V 追記、カーネル起動 |
| `paged_kv_cache.cu` | `PagedKVCache` の実装(ページプール、参照カウント、fork / コピーオンライト / reorder / release、不変条件チェック) |
| `fi_kernels.cuh` | FlashInfer のテンプレートを包む内部宣言 |
| `fi_prefill.cuh`, `fi_prefill_{nowindow,window}.cu` | `BatchPrefillWithPagedKVCacheDispatched` の明示的実体化(window なし/あり × CTA_TILE_Q 16/64/128) |
| `fi_decode.cu` | `DecodePlan` と `BatchDecodeWithPagedKVCacheDispatched` の実体化 |
| `test_attention.cu` | 正しさのテスト(fp32 CPU リファレンスとの比較) |
| `bench_page_size.cu` | page_size の比較ベンチマーク |
| `beam_search_sample.cu` | ビームサーチでの fork / コピーオンライト / reorder / release のサンプル兼検証 |
| `ref_attention.h` | fp32 CPU リファレンスと許容誤差(テストとサンプルで共用) |
| `API_NOTES.md` | FlashInfer API の調査結果 |

## 動作確認した環境

| 項目 | 値 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti (8 GB, 34 SM, L2 32 MiB), Compute Capability 8.9 |
| ドライバ | 595.104.02 |
| CUDA Toolkit | 13.2 (nvcc V13.2.51) |
| ホストコンパイラ | GCC 15.2.0 |
| CMake | 4.2.3 (Unix Makefiles) |
| FlashInfer | `3637083a3a3d08538cce60937cd876b17779c6f0`(v0.7.2, 2026-10-15) |
| FlashInfer サブモジュール | cccl `16bd510c`, cutlass `b46b16d0`(cutlass は今回のカーネルでは未使用) |

## ビルドと実行

```bash
# FlashInfer をこのディレクトリ直下に取得 (サブモジュール込み)
git clone --recurse-submodules https://github.com/flashinfer-ai/flashinfer.git
git -C flashinfer checkout 3637083a3a3d08538cce60937cd876b17779c6f0
git -C flashinfer submodule update --init --recursive

cmake -S . -B build                 # 既定: Release, CMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j 8            # 16 コアで約 15 秒 (テンプレート実体化の 4 ファイルを並列ビルド)

./build/test_attention              # 正しさのテスト (全ケースの PASS/FAIL と最大誤差を表示)
./build/bench_page_size             # page_size 比較 (引数で window_left を指定可。既定は window なし)
./build/beam_search_sample          # ビームサーチのサンプル (全構成を検証し、終了コード 0 で全 PASS)

# assert を有効にしたデバッグビルド (キャッシュ操作のたびに不変条件を assert で確認)
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build-debug -j 8
```

- FlashInfer を別の場所に置いた場合: `-DFLASHINFER_DIR=/path/to/flashinfer`
- 別の GPU で使う場合: `-DCMAKE_CUDA_ARCHITECTURES=80` など
- FlashInfer はヘッダオンリーとして使っています。include パスは `include/` と同梱 CCCL
  (`3rdparty/cccl/{libcudacxx/include,cub,thrust}`)の 2 系統です。FlashInfer 自身の JIT と同様に、同梱 CCCL を CUDA Toolkit 付属の版より優先させています。
- C++17 / `--expt-relaxed-constexpr` でビルドします。

## 使い方 (prefill → decode)

```cpp
#include <vector>
#include "attention.h"

int main() {
  using namespace attn;
  PagedKVCache cache(kDefaultPageSize, /*max_num_pages=*/4096);  // page_size = 16
  AttentionWorkspace ws;  // 一度作って全ステップで使い回す

  const std::vector<int> batch = {cache.add_sequence(), cache.add_sequence()};
  const int window_left = 255;  // 自分 + 直前 255 トークンを見る。kNoWindow で無効

  // --- prefill: 1 本目に 5 トークン、2 本目に 3 トークン (ragged) ---
  const std::vector<int32_t> qo_indptr = {0, 5, 8};  // ホスト配列
  bf16 *q, *k, *v, *o;                               // デバイス上 [8, heads, 64]
  CUDA_CHECK(cudaMalloc(&q, 8 * kNumQoHeads * kHeadDim * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&k, 8 * kNumKvHeads * kHeadDim * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&v, 8 * kNumKvHeads * kHeadDim * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&o, 8 * kNumQoHeads * kHeadDim * sizeof(bf16)));
  // ... q/k/v を埋める ...
  prefill(ws, cache, batch, q, k, v, qo_indptr, window_left, o);

  // --- decode: 各シーケンスに 1 トークンずつ。q/k/v/o は [2, heads, 64] ---
  for (int step = 0; step < 16; ++step) {
    // ... 前ステップの出力から次トークンの q/k/v を作る ...
    decode(ws, cache, batch, q, k, v, window_left, o);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cache.free_sequence(batch[0]);  // 終わったシーケンスのページを返す
  cache.free_sequence(batch[1]);
  cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o);
}
```
(このコードはビルドして実行できることを確認済みです。)

### API の要点
- **`PagedKVCache(page_size, max_num_pages)`**(page_size の既定値は `kDefaultPageSize` = 16)
  - K / V それぞれの GPU ページプールを確保する。レイアウトは NHD `[max_num_pages, page_size, 8, 64]`。
  - `add_sequence()` / `free_sequence(id)` / `truncate(id, len)` / `length(id)` / `pages(id)` を持つ。
  - 空きページはフリーリストで管理する。解放したページは別のシーケンスで再利用される。
- **`prefill(ws, cache, seq_ids, q, k, v, qo_indptr, window_left, out, stream)`**
  - q/k/v/out は ragged `[qo_indptr.back(), heads, 64]`。`qo_indptr` はホスト配列。
  - 処理: 新しい K/V を `AppendPagedKVCache` でキャッシュに追記し、causal + sliding window の attention を計算する。
  - すでにトークンを持つシーケンスに対しては、その prefix の続きとして計算する(chunked prefill)。
- **`decode(ws, cache, seq_ids, q, k, v, window_left, out, stream)`**
  - q/k/v/out は `[batch, heads, 64]`。
  - 処理: 長さを +1 → `AppendPagedKVCacheDecode` → キャッシュ全体に対して window を適用した attention。
  - `seq_ids` にキャッシュ内の一部のシーケンスだけを渡してもよい。
- **`window_left`**: 左側の文脈トークン数(FlashInfer の定義どおり)。
  - 位置 p のクエリは `[p - window_left, p]` を参照する。`0` なら自分だけを見る。
  - `kNoWindow`(負値)なら全トークンを参照する。
- **2 段階 API**: `plan_prefill(…, qo_indptr, kv_indptr, window_left, window_right)` / `plan_decode(…, window_left, window_right)` → `run_planned`。`prefill()` / `decode()` はこの 2 つを順に呼ぶだけ。
  - `run_planned` は同じ plan で繰り返し実行できる(同じスロットに書き直すだけ)。ベンチマークはこれを利用している。
  - plan の後にキャッシュのページ表が変わった場合、`run_planned` は例外を投げる。
- **エラー処理**
  - CUDA API のエラーは `CUDA_CHECK` で `std::runtime_error` にする。FlashInfer 内部のエラーは `flashinfer::Error`(`std::exception` の派生)。
  - 引数の誤りは `std::invalid_argument`(未知 / 重複した ID、長さ 0 の prefill 行など)。
  - ページ不足は `std::runtime_error`。この場合キャッシュは変更されない。

### ストリーミング(右ウィンドウ `window_right`)
位置 p のクエリは `[p - window_left, p + window_right]` を参照します。
クエリは右コンテキスト(`p + window_right` まで)がキャッシュに入ってから渡す前提です。つまりクエリは、キャッシュの末尾から `window_right` 個遅れて入力されます。

```cpp
// prefill: K/V とクエリの数を別々に指定する
prefill(ws, cache, seq_ids,
        q, qo_indptr,      // 今回計算可能になったクエリ [qo_indptr.back(), 8, 64]
        k, v, kv_indptr,   // 今回到着したトークン       [kv_indptr.back(), 8, 64]
        window_left, window_right, out);
// decode: 1 トークン追記し、位置 length-1-window_right のクエリを 1 つ計算する
decode(ws, cache, seq_ids, q, k, v, window_left, window_right, out);
```
- **prefill**
  - シーケンス i の追記後の長さを `len_i` とすると、渡すクエリは位置 `[len_i - R - qo_len_i, len_i - R)` に当たります(R = window_right)。
  - 例えば毎回「新たに計算可能になったクエリをすべて」渡すなら、`qo_len_i = max(0, len_i - R) - max(0, 旧len_i - R)` です。
  - 先読み分がまだ溜まっていないシーケンスはクエリ数 0 にできます。到着トークンが 0 のシーケンスがあっても構いません。
  - `qo_len_i > len_i - R` の場合(右コンテキストが揃っていないクエリがある場合)は `std::invalid_argument` になります。
- **decode**: 追記前の長さが `window_right` 以上である必要があります。足りない間は、クエリ数 0 の prefill で K/V だけ追記します。
- `window_right = 0` は通常の causal と同じです。`window_right < 0`(右側無制限)はストリーミングでは意味を持たないので受け付けません。
- **実装**: カスタムカーネルや Variant は不要でした。理由は次のとおりです。
  - クエリが末尾から R 遅れているので、右端 `p + R` は FlashInfer の causal マスクの右端(末尾揃え)とちょうど一致します。
  - 左端は FlashInfer に `window_left = L + R` を渡すと `p - L` になります。
  - したがって、既存の `MaskMode::kCausal` + sliding window のカーネル(prefill / decode とも)で厳密に同じ計算になり、FlashInfer の改変や `custom/` 以下の追加コードはありません。
  - 換算は `attention.cu` の `flashinfer_window()` で行っています。導出は API_NOTES.md §9 にあります。

### 1 ステップの流れ(実装)
1. 前のステップのホスト→デバイス転送の完了を待つ。ピン留めバッファを書き換えてよい状態にするため(`cudaEventSynchronize`)。
2. `PagedKVCache::extend` でページを確保する(全部確保できるか、何も変えないかのどちらか)。
3. 次の配列をピン留めバッファに構築し、1 回の `cudaMemcpyAsync` で転送する:
   `kv_indptr`(ページ単位)/ `kv_last_page_len` / `kv_indices`、prefill ではさらに `qo_indptr` / `batch_indices` / `positions`。
4. FlashInfer の plan(`PrefillPlan` / `DecodePlan`)を**1 回**呼ぶ。workspace は float 128 MiB / int 8 MiB / pinned 8 MiB(Python 版と同じ既定値)を使い回す。
5. K/V を追記する(`AppendPagedKVCache` / `AppendPagedKVCacheDecode`)。
6. attention カーネルを起動する。
   - `window_left >= 0` のときは `DefaultAttention<…, use_sliding_window=true, …>` の実体化を使う。
   - prefill の CTA_TILE_Q は plan の結果に従って 16 / 64 / 128 から選ぶ。

## テスト

`./build/test_attention` を実行すると、全 167 ケースが PASS し、最大絶対誤差は **3.8e-3** でした(判定基準は atol = rtol = 1e-2)。

- **リファレンス**: CPU 上の fp32 実装で `softmax(q kᵀ / sqrt(64)) v` に causal マスクと sliding window マスクを適用したもの。
  右ウィンドウの場合は、参照範囲 `[p - window_left, p + window_right]` を明示的に適用する(FlashInfer に渡す値への換算は使わない)。
  入力は bf16 に丸めた値を使う。
- **判定**: `|out - ref| <= 1e-2 + 1e-2·|ref|` を全要素で満たすこと。NaN は不合格。出力バッファは事前に NaN で埋めておき、書き込まれなかった行を検出する。
- **ケース**
  1. prefill → decode(108 ケース)
     - page_size {1, 2, 4, 8, 16, 32} × batch {1, 4, 16} × window_left {無効, 0, 1, 16, 100, 100000}
     - 長さはページ境界 ±1、長さ 1、37、129、250 などを混ぜた ragged
     - 続けて decode を `2·page_size + 3` ステップ行い、各ステップを全履歴で再計算したリファレンスと比較する
  2. window とシーケンス長 L の境界(8 ケース): window_left = L−2, L−1, L, 10L(page 1 と 16)
  3. 長いシーケンスと chunked prefill(8 ケース)
     - 長さ 1500 と 3 で prefill した後、それぞれに 5 と 1200 トークンを追加で prefill し、decode とサブバッチ decode を行う
     - FlashInfer の KV 分割(split-KV)とマージの経路を通ることを nsys で確認済み
  4. キャッシュ管理(1 ケース): free → add によるページ再利用(断片化したページ表)、途中参加、`truncate` による巻き戻し
  5. ストリーミング(34 ケース)
     - page {1, 16} × batch {1, 4} × (window_left, window_right) ∈ {(無効,0), (無効,1), (無効,7), (0,0), (0,1), (3,5), (16,33), (50,20)}
     - 到着チャンクの大きさを 0, 1, R−1, R, R+1, 2, 17, 40, 3 と変える。先読み分が溜まらずクエリ数 0 になるステップや、全シーケンスがクエリ 0 のステップも含む
     - 毎回、計算可能になったクエリをすべて計算し、続けて streaming decode を `2R+3` ステップ行う
     - 長いストリーム(1500 トークン、L=300, R=64)では KV 分割の経路も通す
  6. エラー系(8 ケース)
     - ページ不足(キャッシュが変化しないこと)、バッチ内の重複 ID、未知の ID、長さ 0 の prefill 行、plan 後にキャッシュが変わった状態での `run_planned`
     - 負の window_right、先読み分が溜まる前の decode、右コンテキストが揃っていないクエリ(いずれもキャッシュが変化しないこと)
- **テスト自体の検証**
  - リファレンスの window を 1 ずらした改変版では 67 ケースが FAIL する。テストが window の境界を検出できることを確認した。
  - ストリーミングのリファレンスで右端を 1 縮めると、R>0 の 26 ケースがすべて FAIL する。左端を 1 広げると、L≥0 の 22 ケースがすべて FAIL する。
- **compute-sanitizer**
  - memcheck / initcheck: エラー 0。
  - racecheck: FlashInfer の prefill カーネルの出力処理(`threadblock_sync_mdo_states` → `write_o_reg_gmem`、CTA_TILE_Q=16/64)で共有メモリのハザードが報告される。
    出力用の `q_smem` はワープ間同期用の `cta_sync_o_smem` と共用体で重なっていて、KV 方向のワープ 0 が bf16 出力を書き込むとき、他の KV ワープがまだそこを読んでいる可能性がある。
    ただし global に出力するのはワープ 0 だけで、ワープ 0 は自分の読み込みを終えてから書き込む。他のワープが計算した値は捨てられるので、結果には影響しないと判断した(数値テストもすべて一致)。
    FlashInfer 本体の挙動なので手を入れていない。

## page_size の決定

### 方法
- **候補**: 1, 2, 4, 8, 16, 32。
  - FlashInfer の `page_size` は実行時の値(`uint_fastdiv`)で、任意の正整数を受け付けるので、すべて計測した。
- **条件**: prefill / decode × batch {1, 8, 32} × KV 長 {512, 4096}。sliding window なし。
  - prefill は各シーケンスに KV 長と同じ数のトークンを入れる。
  - decode は KV 長ぶん埋めたキャッシュに 1 トークンを足す。
- **指標**
  - **gpu(判定に使う)**: `run_planned`(K/V 追記 + attention)の GPU 時間。
    - CUDA イベントの分解能は約 0.5 µs で、数 µs の decode には粗すぎる。そこで 1 サンプルを約 1 ms 分(最大 100 回)の連続実行の平均とし、30 サンプルの中央値をとった。
    - 計測中はスピンカーネルでストリームを止めておき、全回分を投入してから解放する。これでホスト側の起動の隙間を含まない。
    - L2 は温まった状態。
  - **gpu_cold**: 毎回 128 MiB を書き込んで L2 を追い出してから測る(参考)。
  - **e2e**: `prefill()` / `decode()` 呼び出し全体(ホストの plan と転送を含む)+ 同期の実時間。15 回の中央値(参考)。
- **ウォームアップ**: 各条件 5 回。加えて、測定開始前に約 1 秒 GPU を回してクロックを上げる。
  これを入れないと、最初の条件がクロック上昇中に測られて 8% 遅く出ることがあった。
- **判定規則**: 条件ごとに最速値との比をとり、全条件での最悪値が 1.05 以下になる最小の page_size を採用する。
  該当がなければ幾何平均が最小のもの。
- 3 回実行し、判定がすべて同じであることを確認した。下の表は 3 回目の結果。

### 結果 (RTX 4060 Ti, ms, 中央値。括弧内は最速値との比)

**gpu: 追記 + attention、L2 温まった状態(判定に使用)**

| condition | P=1 | P=2 | P=4 | P=8 | P=16 | P=32 |
|---|---|---|---|---|---|---|
| prefill b=1 kv=512 | 0.0671 (1.00) | 0.0671 (1.00) | 0.0670 (1.00) | 0.0671 (1.00) | 0.0671 (1.00) | 0.0671 (1.00) |
| prefill b=1 kv=4096 | 0.5566 (1.02) | 0.5499 (1.01) | 0.5489 (1.01) | 0.5432 (1.00) | 0.5479 (1.01) | 0.5532 (1.02) |
| prefill b=8 kv=512 | 0.0953 (1.00) | 0.0949 (1.00) | 0.0951 (1.00) | 0.0950 (1.00) | 0.0953 (1.00) | 0.0948 (1.00) |
| prefill b=8 kv=4096 | 4.0000 (1.00) | 4.0078 (1.00) | 4.0172 (1.00) | 4.0120 (1.00) | 4.0087 (1.00) | 4.0128 (1.00) |
| prefill b=32 kv=512 | 0.5300 (1.00) | 0.5301 (1.00) | 0.5279 (1.00) | 0.5289 (1.00) | 0.5284 (1.00) | 0.5277 (1.00) |
| prefill b=32 kv=4096 | 15.8890 (1.00) | 15.8984 (1.00) | 15.9046 (1.00) | 15.8623 (1.00) | 15.8854 (1.00) | 15.8828 (1.00) |
| decode b=1 kv=512 | 0.0074 (1.01) | 0.0074 (1.01) | 0.0074 (1.00) | 0.0074 (1.00) | 0.0074 (1.01) | 0.0074 (1.01) |
| decode b=1 kv=4096 | 0.0126 (1.03) | 0.0126 (1.03) | 0.0126 (1.03) | 0.0125 (1.02) | 0.0124 (1.01) | 0.0122 (1.00) |
| decode b=8 kv=512 | 0.0105 (1.01) | 0.0104 (1.01) | 0.0104 (1.00) | 0.0104 (1.00) | 0.0104 (1.00) | 0.0103 (1.00) |
| decode b=8 kv=4096 | 0.2522 (1.00) | 0.2519 (1.00) | 0.2519 (1.00) | 0.2520 (1.00) | 0.2519 (1.00) | 0.2519 (1.00) |
| decode b=32 kv=512 | 0.0307 (1.01) | 0.0306 (1.00) | 0.0305 (1.00) | 0.0306 (1.00) | 0.0306 (1.00) | 0.0307 (1.01) |
| decode b=32 kv=4096 | 0.9953 (1.01) | 0.9913 (1.00) | 0.9902 (1.00) | 0.9892 (1.00) | 0.9892 (1.00) | 0.9892 (1.00) |

**e2e: plan + 転送 + 追記 + attention + 同期(参考)**

| condition | P=1 | P=2 | P=4 | P=8 | P=16 | P=32 |
|---|---|---|---|---|---|---|
| prefill b=1 kv=512 | 0.0797 (1.03) | 0.0784 (1.01) | 0.0779 (1.01) | 0.0777 (1.00) | 0.0777 (1.00) | 0.0774 (1.00) |
| prefill b=1 kv=4096 | 0.5908 (1.05) | 0.5811 (1.03) | 0.5787 (1.03) | 0.5799 (1.03) | 0.5729 (1.02) | 0.5620 (1.00) |
| prefill b=8 kv=512 | 0.1334 (1.07) | 0.1297 (1.04) | 0.1260 (1.01) | 0.1243 (1.00) | 0.1253 (1.01) | 0.1242 (1.00) |
| prefill b=8 kv=4096 | 4.2397 (1.02) | 4.2101 (1.01) | 4.1860 (1.00) | 4.1752 (1.00) | 4.1705 (1.00) | 4.1659 (1.00) |
| prefill b=32 kv=512 | 0.6585 (1.06) | 0.6364 (1.03) | 0.6267 (1.01) | 0.6230 (1.01) | 0.6223 (1.01) | 0.6192 (1.00) |
| prefill b=32 kv=4096 | 16.7814 (1.02) | 16.6266 (1.01) | 16.5618 (1.01) | 16.4616 (1.00) | 16.5186 (1.00) | 16.4923 (1.00) |
| decode b=1 kv=512 | 0.0165 (1.03) | 0.0163 (1.01) | 0.0162 (1.01) | 0.0161 (1.00) | 0.0162 (1.01) | 0.0163 (1.01) |
| decode b=1 kv=4096 | 0.0226 (1.06) | 0.0223 (1.04) | 0.0219 (1.02) | 0.0215 (1.01) | 0.0217 (1.02) | 0.0214 (1.00) |
| decode b=8 kv=512 | 0.0223 (1.06) | 0.0217 (1.03) | 0.0211 (1.01) | 0.0211 (1.01) | 0.0210 (1.00) | 0.0211 (1.00) |
| decode b=8 kv=4096 | 0.2955 (1.13) | 0.2827 (1.08) | 0.2762 (1.05) | 0.2634 (1.01) | 0.2637 (1.01) | 0.2620 (1.00) |
| decode b=32 kv=512 | 0.0693 (1.42) | 0.0557 (1.14) | 0.0501 (1.03) | 0.0488 (1.00) | 0.0495 (1.01) | 0.0489 (1.00) |
| decode b=32 kv=4096 | 1.1332 (1.12) | 1.0692 (1.06) | 1.0421 (1.03) | 1.0266 (1.01) | 1.0121 (1.00) | 1.0174 (1.01) |

**最速値との比の最悪値(3 回の実行を通じた値)**

| page size | gpu(判定) | gpu_cold | e2e |
|---|---|---|---|
| 1 | 1.03 | 1.02 | **1.41–1.43** |
| 2 | 1.03 | 1.02–1.04 | 1.14–1.29 |
| 4 | 1.02–1.03 | 1.02–1.03 | 1.05–1.20 |
| 8 | 1.02 | 1.02–1.04 | 1.02–1.04 |
| 16 | 1.01–1.02 | 1.01–1.02 | 1.01–1.02 |
| 32 | 1.01–1.02 | 1.01 | 1.01–1.02 |

gpu_cold の全表と各回の出力は `./build/bench_page_size` を実行すると表示されます。
参考として `window_left=255` でも計測しました(`./build/bench_page_size 255`)。判定は同じく 1 でした。
window を有効にすると prefill はウィンドウ外を飛ばすので速くなり、b=32 kv=4096 では 15.9 → 4.75 ms でした。
decode の時間は window なしと変わりません(未対応事項を参照)。

### 採用値と理由

**既定値は page_size = 16** です(`attn::kDefaultPageSize`)。計測結果を踏まえて、最終的に指定いただいた値です。

- 16 は、GPU 時間でも e2e(plan を含む呼び出し全体)でも、全条件で最速値から約 2% 以内に収まります。最悪比は、3 回の実行を通じて GPU 1.015–1.022、e2e 1.01–1.02 でした。
- 当初合意した規則(GPU 時間で最速値から 5% 以内に入る最小値)だけを適用すると page_size = 1 になります(3 回とも同じ)。
  e2e も 5% 以内という条件を加えると、両方を満たす最小値は 8 です。
  経緯は以下のとおりです。

- GPU 時間は page_size にほとんど依存せず、すべての候補が 3.3% 以内に収まります。
  - 理由: NHD レイアウトでは、ページ内のトークンの間隔がもともと `heads × head_dim × 2B = 1 KiB` です。そのため page_size を大きくしても、1 ヘッドぶんの読み出し(128 B 単位)の連続性は変わりません。
  - page_size が効くのは、ページ表の参照回数と KV 分割(最小分割サイズ `128 / page_size` ページ)くらいです。
- **ただし e2e では page_size が小さいほど遅く**なります(P=1 は decode b=32 kv=512 で 1.42 倍)。
  - 原因はホスト側の処理で、毎ステップ全ページ表(`kv_indices`、page_size=1 ならトークン数と同じ要素数)を組み立てて転送しています。b=32 kv=4096 なら 13 万要素 = 512 KiB です。
  - ベンチマークの計画時に「plan のホスト時間は page_size にほぼ依存しない」と書きましたが、この前提は誤りでした。
  - e2e も 5% 以内という条件を加えると、両方を満たす最小値は page_size = 8 です。GPU 時間の最悪比は 1.02、e2e の最悪比は 1.02–1.04 で、3 回とも満たしました。
  - 16 は 8 よりさらに e2e の余裕が大きく(最悪比 1.02 以下)、ページ表もさらに小さくなります。
- **page_size を実行時引数にした理由**
  - FlashInfer の API で page_size は実行時の値です(`paged_kv_t::page_size` は `uint_fastdiv`、plan の引数も実行時)。page_size をテンプレート引数に取るカーネルはありません。
  - コンパイル時定数にしても速度の利点はなく、実体化の数が増えるだけです。
  - そのため `PagedKVCache(page_size, max_num_pages)` のコンストラクタ引数にし、既定値を定数 `kDefaultPageSize`(16)として用意しました。
  - `bench_page_size` の最後の行に出る "Chosen page_size" は、当初の規則(GPU 時間のみ)を機械的に適用した結果で、既定値とは別です。

## ビームサーチ(fork / コピーオンライト / reorder / release)

`PagedKVCache` にページの参照カウントを持たせ、ビーム間でページを共有できるようにしました。
attention の計算は既存の `prefill()` / `decode()` をそのまま使っています(関数本体は無変更)。

| API | 内容 |
|---|---|
| `fork(src, n)` | src のページ表を共有する n 本の新シーケンスを作る。K/V はコピーせず、各ページの参照カウントを +n する |
| `copy_on_write(seq_ids, stream)` | 追記の前に呼ぶ。最終ページが「途中まで埋まっていて共有されている」シーケンスにだけ新ページを割り当て、使用済みスロットを GPU でコピーし、ページ表を差し替え、元ページの参照を外す。1 回の呼び出しのコピーは 1 回のカーネル起動にまとめる。戻り値はコピーしたページ数 |
| `reorder(beam_ids, parent_idx)` | 新ビーム i が、旧ビーム `parent_idx[i]` のページ表(と長さ)を引き継ぐ |
| `release(seq)` | シーケンスの全ページの参照を外す。参照カウントが 0 になったページはフリーリストに戻る(`free_sequence` と同じ) |
| `check_invariants()` | 下記の不変条件を検査し、違反があれば `std::logic_error` を投げる |
| `num_used_pages()` / `ref_count(page)` | 使用中のページ数 / ページの参照カウント |

### 図解(page_size = 16、プロンプト長 37 の 1 リクエストを W = 4 で探索する場合)

**fork**: ページ表を共有するだけで、中身はコピーしません。
```
prefill 後     prompt: [P0 P1 P2*]                    * = 途中まで埋まっている (37 = 16+16+5)
fork(prompt,4) beam0..3: [P0 P1 P2*]                  ref(P0)=ref(P1)=ref(P2)=5
release(prompt)                                        ref = 4  (どのページも解放されない)
```

**コピーオンライト**: decode の前に、共有されている途中ページだけを複製します。
```
copy_on_write(beam0..3):   P2 は 5/16 埋まり、ref 4
  beam0: 新ページ Q0 を確保し、P2 のスロット 0..4 を Q0 へコピー   ref(P2)=3
  beam1: 同様に Q1                                                  ref(P2)=2
  beam2: 同様に Q2                                                  ref(P2)=1
  beam3: ref(P2)=1 なので P2 をそのまま専有する(コピーなし)
  → コピー P2→Q0, P2→Q1, P2→Q2 を 1 回のカーネル起動で実行
decode:  beam0:[P0 P1 Q0]  beam1:[P0 P1 Q1]  beam2:[P0 P1 Q2]  beam3:[P0 P1 P2]
         各ビームが自分の最終ページのスロット 5 に新トークンを書く。P0, P1 は 4 本で共有のまま
```
最終ページが満杯(`length % page_size == 0`)の場合はコピーしません。新トークンは新しいページから始まるため、共有ページに書き込むことがないからです。

**reorder**: 子は親のページ表を引き継ぎます。
```
before   beam0:[P0 P1 Q0]  beam1:[P0 P1 Q1]  beam2:[P0 P1 Q2]  beam3:[P0 P1 P2]
parent_idx = [1, 1, 0, 3]          (beam1 が 2 回選ばれ、beam2 は選ばれない)
 1) 加算: 各子 i について old[parent_idx[i]] の全ページを +1
 2) 減算: 各ビーム i について old[i] の全ページを -1
after    beam0:[P0 P1 Q1]  beam1:[P0 P1 Q1]  beam2:[P0 P1 Q0]  beam3:[P0 P1 P2]
         ref(Q1)=2, ref(Q0)=1, ref(P2)=1, ref(Q2)=0 → Q2 はフリーリストへ
```
次のステップでは Q1 が共有されているので、beam0 か beam1 のどちらか一方だけがコピーオンライトでコピーします。

### 処理順序の注意点
- **reorder では、参照カウントを先に全部加算し、その後で減算する。**
  上の例でビームごとに「自分の旧ページを -1 してから親のページを +1」と処理すると、i=0 の時点で Q0 が 0 になってフリーリストに戻ります。
  その後 i=2 が Q0 を +1 して使い始めるので、「フリーリストにあるのに使用中」のページができ、別のシーケンスに二重に割り当てられてしまいます。
  実装では、旧ページ表のスナップショットを取ってから、加算 → 減算 → 差し替えの順で処理しています。
  わざと順序を逆にした版では、不変条件チェックが `page 5 is in the free list but in use` を検出しました(下記「検証の検証」)。
- **追記の前に必ず `copy_on_write()` を呼ぶ。**
  - 呼ばずに `decode()` / `prefill()` が共有中の途中ページへ追記しようとすると、`PagedKVCache::extend` が `std::logic_error` を投げます。この場合キャッシュは変更されません(サンプルで確認しています)。
  - `copy_on_write()` は、ページが足りなければ何も変えずに例外を投げます(全部やるか、何もしないかのどちらか)。
- **コピーオンライトのコピーと、それに続く decode の追記は同じストリームで行う。**
  - 同じストリームにすることで、コピーの完了後に追記が行われます。
  - reorder や release で解放したページはすぐに再利用されることがありますが、同じストリーム上なら、前のステップのカーネルが読み終えてから書き換えられます。
- 参照カウントの変更はすべてホスト上で行います。GPU の仕事はコピーオンライトのページコピーだけです。
  `kv_indptr` / `kv_indices` / `kv_last_page_len` は、`decode()` / `prefill()` の plan のたびに CPU で再構築して GPU に転送しています(既存の仕組みのまま)。

### 不変条件(毎ステップ `check_invariants()`、デバッグビルドでは操作のたびに assert)
- 各ページの参照カウントは、全ページ表に現れる回数と一致する。したがって参照カウントの合計 = ページ表の要素数の合計。
- フリーリストに重複がなく、フリーリストのページは参照カウント 0。
- 使用中ページ数 + フリーリスト長 = 全ページ数(リークなし)。
- 各シーケンスのページ数 = ceil(length / page_size)。

### サンプル (`beam_search_sample.cu`)
**処理の流れ**
1. R 本のプロンプト(長さはバラバラ)を `prefill()` する
2. 各プロンプトを `fork(prompt, W)` し、プロンプト自体は `release()` する
3. 次を T ステップ繰り返す
   - `copy_on_write()` → R×W 本まとめて `decode()`
   - 疑似スコアでリクエストごとに top-W を選び、`parent_idx` を得る
   - リクエストごとに `reorder()`
4. 全ビームを `release()` し、全ページがフリーリストに戻ったことを確認する

**疑似スコア**: 各ビームから W 個の候補を作り、「累積スコア − 指数分布の乱数」で top-W を選びます。
5 ステップに 1 回は 1 本のビームを優遇して、全子を同じ親から選ばせます。親の重複、選ばれない親、全員同じ親の 3 パターンが必ず起き、終了時に発生回数を確認しています。

**毎ステップの検証**
- 全ビームの decode 出力を、リファレンスと既存テストと同じ許容誤差で比較する。
  リファレンスはビームごとの連続した K/V 履歴で、reorder のたびに親の履歴を丸ごとコピーする。
- コピーオンライトのコピー数が期待値と一致すること。期待値は「途中まで埋まった最終ページごとに、保持ビーム数 − 1」。
- 不変条件チェック。
- 使用中ページ数を、共有なし(ビームごとに全コピー)の場合のページ数 Σ ceil(len/page_size) と並べて表示する。

**構成**(それぞれ window なしと `window_left=12` の 2 通りで実行)

| 構成 | R | W | T | page_size | プロンプト長 | 確認すること |
|---|---|---|---|---|---|---|
| basic | 3 | 4 | 20 | 16 | 37, 16, 5 | 基本動作 |
| W=1 | 2 | 1 | 20 | 16 | 21, 8 | 共有もコピーも起きない |
| all beams from one parent | 2 | 4 | 12 | 4 | 10, 3 | 毎ステップ全ビームが同じ親から選ばれる |
| last page full at fork | 2 | 4 | 10 | 16 | 16, 32 | fork 直後の step 0 でコピーが 0 回 |
| page_size=1 | 2 | 3 | 20 | 1 | 7, 12 | ページが常に満杯なので、コピーオンライトが一度も起きない |

**結果**: 10 実行すべて PASS(最大絶対誤差 2.8e-3)。リーク 0。compute-sanitizer の memcheck / initcheck もエラー 0 です。

出力例(basic、window なし。`len` は decode 後の長さ、`CoW` はそのステップでコピーしたページ数、`unshared` は共有しない場合に必要なページ数):
```
=== basic | R=3 W=4 T=20 page_size=16 window_left=off | prompts: 37 16 5
after fork: used pages 5, unshared would need 20
      (step 0: decode() without copy_on_write() into a shared page was refused, cache unchanged)
 step   len    CoW     used   unshared   max_abs_err  parents of request 0
    0    38      6       11         24     1.950e-03  [3 0 1 1]
    1    39      4       10         24     1.949e-03  [0 1 1 1]
    2    40      5       12         24     1.947e-03  [2 2 2 0]
    3    41      3       11         24     1.951e-03  [0 0 1 1]
    4    42      4        6         24     1.929e-03  [0 0 0 0]
  ...
   10    48      9       12         24     1.944e-03  [3 1 1 2]
   11    49      1       17         32     1.917e-03  [0 1 3 3]
  ...
   19    57      2        9         36     1.944e-03  [3 3 3 3]
after release: used pages 0, free 60 / 60 -> no leak
summary: rows=298 bad=0 max_abs_err=2.386e-03 CoW pages=83 | steps with a parent chosen >1x: 20, with an unchosen parent: 20, all from one parent: 4 -> PASS
```
共有によって、使用ページ数は共有なしの場合の 1/4〜1/2 程度に収まっています(例: 24 → 6〜12、36 → 9〜17)。
全ビームが同じ親から選ばれたステップ(step 4, 9, 14, 19)の直後は、全ビームが 1 本の履歴を共有するので最小になります。

**検証の検証**: `paged_kv_cache.cu` にわざと誤りを入れた版で、サンプルが FAIL することを確認しました。

| 誤り | 検出のされ方 |
|---|---|
| コピーするトークン数を 1 減らす | attention 出力がリファレンスと不一致 |
| コピーカーネルを起動しない | attention 出力がリファレンスと不一致 |
| reorder で減算を先にする | 不変条件違反(`page 5 is in the free list but in use`、`page 2 is twice in the free list`)。デバッグビルドでは reorder 直後に assert で停止 |

### FlashInfer の共有プレフィックス向け機能(調査のみ)
- **FlashInfer にある機能**
  - Python API の `MultiLevelCascadeAttentionWrapper`(cascade attention)。旧 API の `BatchDecodeWithSharedPrefixPagedKVCacheWrapper` は deprecated です。
    仕組みは、KV をレベル(共有プレフィックス / 各リクエスト固有の部分)に分けて、レベルごとに paged prefill を LSE 付きで実行し、`merge_state_in_place` で統合するものです。
  - C++ 側に専用のカーネルはありません。部品は C++ ヘッダで揃っています:
    - `BatchPrefillWithPagedKVCacheDispatched`(`params.lse` に LSE を出力)
    - `cascade.cuh` の `MergeStateInPlace` / `MergeStates` / `VariableLengthMergeStates`
- **ビームサーチへの適用**: 可能です。全ページを 1 つのプールに置く前提なので、今回のページ共有と同じプールでそのまま使えます。
  - 上位レベルでは、リクエストごとに、W 本のビームのクエリ(qo_len = W)と、全ビームに共通するページ列(ページ表の最長共通接頭辞)の attention を計算します。
    マスクなしの `MaskMode::kNone` で、共有部分を 1 回だけ読みます。
  - 下位レベルでは、各ビーム固有のページだけで decode し、2 つの結果を LSE でマージします。
  - 長いプロンプトに対して W 本のビームが同じ KV を W 回読む部分が 1 回になるので、メモリ帯域の節約になります。
  - 一方で、カーネル起動とマージが増えるので、プレフィックスが短いうちは得になりません。
- **今回の実装に組み込む場合に必要なこと**
  1. prefill / decode から LSE を出力できるようにする(今は `lse = nullptr` 固定)。
  2. ビーム間の共通接頭辞ページを、ステップごとにホストで計算する。
     コピーオンライトの後なら共通部分は満杯ページだけになるので、上位レベルの `last_page_len = page_size` で表せます。
  3. **sliding window との組み合わせには注意が必要です。**
     - 上位レベルの W 本のクエリは同じ位置にあります(ビームは同じ長さ)。しかし FlashInfer の window マスクは、クエリを末尾揃えで別々の位置とみなします。そのため `window_left` をそのまま渡すと誤ります。
     - 対策は 2 つ考えられます:
       - ウィンドウ外の共通ページを上位レベルから外し(ページ単位)、境界ページ内だけをマスクするカスタム Variant を使う(`custom/` に置く想定)
       - window ありのときは cascade を使わない
  4. reorder で共有の形が変わるため、上位レベルの範囲もステップごとに変わります(plan は毎ステップ実行)。

## 仮定と未対応事項

**仮定 / 仕様の解釈**
- `window_left` は FlashInfer の定義どおり「左側の文脈トークン数」です(0 なら自分だけ、負値なら無効)。ウィンドウ幅は `window_left + 1` になります。
- head 数、head_dim、dtype はコンパイル時定数(`attention.h`)です。GQA は `kNumKvHeads` を変えれば FlashInfer 側は対応していますが、テストしていません。
- 位置エンコーディング(RoPE など)はカーネルに融合していません(`PosEncodingMode::kNone`)。呼び出し側で適用済みの Q/K を渡す前提です。
- 通常の(右ウィンドウなしの)`prefill` では、各シーケンスに 1 トークン以上必要です。既にトークンを持つシーケンスへの prefill は、その prefix の続きとして扱います。
- ストリーミングでは、クエリは右コンテキストが揃ったもの(位置 `len - 1 - window_right` まで)だけが渡される前提です。
- 1 つの `AttentionWorkspace` は 1 つのホストスレッドと 1 つのストリームで使う前提です。
  - 毎ステップの冒頭で、前のステップのホスト→デバイス転送の完了を待ちます(ピン留めバッファの再利用のため)。カーネルの完了は待ちません。
- ページ不足以外の理由で plan の途中に例外が出た場合、そのバッチのシーケンス長は伸びたまま(K/V は未書き込み)になります。`truncate` で戻せます。

**未対応**
- ビームサーチのコピーオンライトは、1 回の呼び出しで 512 ページを超えると、カーネル起動を 512 ページごとに分割します。コピー情報はカーネル引数で渡しています。
- 参照カウントを使った共有は、cascade attention のような計算量の削減は行いません(上記の調査を参照)。
- **ストリーム終端の処理**: 入力が終わった後に、右コンテキストが途中で切れたまま最後の `window_right` 個のクエリを計算する機能はありません。
  必要なら追加できます。参照範囲は `[p - L, len - 1]` になり、FlashInfer の `MaskMode::kNone` + `window_left = L` で表せるので、カーネルの改変は不要です。- CUDA Graph(仕様どおり不要)。plan は `enable_cuda_graph = false` です。
- prefill と decode が混在するバッチ(仕様どおり不要)。
- **decode の sliding window はマスクだけで、計算量は削減されません。**
  FlashInfer の decode カーネル(CUDA コア版)はウィンドウ外の KV も読んでから要素ごとにマスクするため、時間はキャッシュの全長に比例します(上の計測でも window の有無で decode 時間は同じ)。
  改善案は 2 つあります:
  - RoPE を融合しない今回の構成なら、ウィンドウ外のページをページ表から外して渡す(マスクは末尾からの相対位置なので結果は変わらない)
  - decode にも prefill(テンソルコア)カーネルを使う
- ウィンドウ外になったページを解放する仕組み(メモリの回収)はありません。
- workspace のサイズは固定です(既定は float 128 MiB / int 8 MiB)。足りない場合は FlashInfer が `flashinfer::Error` を投げます。今回の形状では余裕があります。
- ページ表は毎ステップ全体を組み立てて転送しています。差分だけ更新すれば、小さい page_size での e2e のオーバーヘッドを減らせます。
- LSE の出力、logits soft cap、ALiBi、カスタムマスクには対応していません。FlashInfer 側には機能があります。
- `-static-global-template-stub=false`(FlashInfer の JIT が付けるフラグ)は使っていません。
  付けると、複数の翻訳単位で同じカーネルテンプレート(`PersistentVariableLengthMergeStatesKernel`)が重複登録され、compute-sanitizer が "Duplicate entry kernels" を報告したためです。
