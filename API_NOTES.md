# FlashInfer C++ API 調査メモ

調査対象: `flashinfer/` (コミット `3637083a3a3d08538cce60937cd876b17779c6f0`, v0.7.2, 2026-10-15)
記載のパスはすべて `flashinfer/` からの相対パス。

## 0. 結論: C++ から直接使えるか

- **使える(ヘッダオンリー)**。Python 版は JIT 時に Jinja (`csrc/*_customize_config.jinja`) で `Params` 構造体と
  `DISPATCH_context` マクロを生成し、`include/flashinfer/attention/*.cuh` のテンプレートを明示的実体化しているだけ。
  ヘッダ側に同等の構造体 (`default_prefill_params.cuh` / `default_decode_params.cuh`) があるので、それを使えば
  torch / TVM-FFI / Python は不要。
- ビルドに必要なもの:
  - include パス: `include/` と同梱 CCCL の `3rdparty/cccl/{libcudacxx/include,cub,thrust}`
    (JIT も同梱 CCCL を `-I` で CTK 同梱版より優先させている: `flashinfer/jit/cpp_ext.py` `build_common_cflags`)
  - nvcc フラグ: `-std=c++17 --expt-relaxed-constexpr`。JIT は `-DFLASHINFER_ENABLE_BF16 -DFLASHINFER_ENABLE_F16 ...` も付ける
  - cutlass / spdlog は FA2 系 (prefill.cuh / decode.cuh) では不要 (`logging.h` を include しない限り)
- 実際に `BatchPrefillWithPagedKVCacheDispatched` / `BatchDecodeWithPagedKVCacheDispatched` / `AppendPagedKVCache`
  を bf16・head_dim=64 で実体化し、nvcc 13.2 + GCC 15.2 / `-arch=sm_89` でコンパイルできることを確認した(1 実体化あたり約 11 秒)。
- 実体化はテンプレートが定義されている `namespace flashinfer` の中で書く必要がある。

## 1. ページ KV キャッシュ

### `paged_kv_t<DType, IdType>` (`include/flashinfer/page.cuh`)
| フィールド | 意味 |
|---|---|
| `page_size` | `uint_fastdiv`(**実行時値**。テンプレート引数ではない) |
| `num_heads`, `head_dim`, `batch_size` | |
| `k_data`, `v_data` | K/V プールの先頭。NHD なら `[max_num_pages, page_size, num_heads, head_dim]`、HND なら `[max_num_pages, num_heads, page_size, head_dim]` |
| `indices` | 全シーケンスのページ番号を連結したもの `[nnz_pages]` |
| `indptr` | `[batch_size+1]`、**ページ単位**の CSR オフセット |
| `last_page_len` | `[batch_size]`、最後のページに入っているトークン数 (1..page_size) |
| `rope_pos_offset` | RoPE 用(今回は nullptr) |

- コンストラクタ: `paged_kv_t(num_heads, page_size, head_dim, batch_size, QKVLayout::kNHD, k_data, v_data, indices, indptr, last_page_len)`
  → K/V のストライドは同一になる。ストライドを個別に渡す版もある。
- 長さ: `get_length(b) = (indptr[b+1]-indptr[b]-1)*page_size + last_page_len[b]`(ページ 0 枚なら 0)。

### 追記関数
- `AppendPagedKVCache(paged_kv, append_key, append_value, batch_indices, positions, nnz, k_stride_n, k_stride_h, v_stride_n, v_stride_h, stream)`
  - ragged な K/V `[nnz, H, D]` を書き込む。`batch_indices[i]` は**バッチ内の添字**、`positions[i]` は**シーケンス内の絶対位置**。
    どちらも int32 のデバイス配列。
  - Python では `get_batch_indices_positions()` が Triton カーネルで作っている。C++ ではホストで計算して転送する。
  - 書き込み先は `indptr[b]*page_size + pos` をページ単位に分解して求める。そのため `paged_kv` は**追記後の長さ**を表している必要がある。
  - `nnz == 0` だと grid が 0 になり起動エラーになるので、呼び出し側で除外する。
- `AppendPagedKVCacheDecode(paged_kv, key, value, stream)`
  - 各シーケンスに 1 トークン `[batch, H, D]` を、位置 `get_length(b)-1` に書く。メタデータは +1 済みであることが前提。

## 2. スケジューリング (plan) — `include/flashinfer/attention/scheduler.cuh`

### Prefill
```
PrefillPlan<IdType>(float_buf, float_bytes, int_buf, pinned_int_buf, int_bytes, PrefillPlanInfo& info,
                    qo_indptr_h, kv_indptr_h, total_num_rows, batch_size, num_qo_heads, num_kv_heads,
                    head_dim_qk, head_dim_vo, page_size, enable_cuda_graph, sizeof_dtype_o,
                    window_left, fixed_split_size, disable_split_kv, num_colocated_ctas,
                    uniform_q_len, stream, kv_dtype_bytes = 2)
```
- `qo_indptr_h` と `kv_indptr_h` は**ホスト**配列。paged の場合 `kv_indptr_h` は**ページ単位**(Python も `paged_kv_indptr` の CPU コピーを渡す)。
- Python の既定値: `fixed_split_size = -1`, `disable_split_kv = false`, `num_colocated_ctas = 0`, `uniform_q_len = 0`, `enable_cuda_graph = false`。
- 決まるもの: `cta_tile_q` ∈ {16, 64, 128}(head_dim 64 の場合、`FA2DetermineCtaTileQ`)、KV 分割の有無と分割サイズ、
  タイルごとの (request, qo_tile, kv_tile) の割り当て。
- `window_left >= 0` のとき、ウィンドウ外のページは分割サイズの計算対象から外れる。

### Decode
```
DecodePlan<HEAD_DIM, POS_ENCODING_MODE, AttentionVariant, Params>(float_buf, float_bytes, int_buf, pinned_int_buf,
                    int_bytes, DecodePlanInfo& info, indptr_h, batch_size, num_qo_heads, page_size,
                    enable_cuda_graph, stream, work_estimation_func)
```
- `work_estimation_func = BatchDecodeWithPagedKVCacheWorkEstimationDispatched<GROUP_SIZE, HEAD_DIM, POS_ENCODING_MODE, AttentionVariant, Params>`
  - 実際のカーネルの occupancy を使うので、実行時と**同じ Variant / Params** で実体化する必要がある。
- `window_left` は受け取らない(分割はウィンドウを考慮しない)。

### plan の副作用と注意点
- 結果のインデックス配列を `pinned_int_buf`(ページロックされたホストメモリ)に書き、`cudaMemcpyAsync` で `int_buf` に転送する。
  - 次の plan が同じピン留めバッファを書き換える前に、この転送が終わっている必要がある。
    → 実装側で、ステップの終わりにイベントを記録し、次ステップの冒頭でそのイベントを同期する。
- 実行時に `GetPtrFromBaseOffset(int_buf, info.xxx_offset)` で `request_indices` / `qo_tile_indices` / `kv_tile_indices` /
  `o_indptr` / `kv_chunk_size_ptr` / (分割時) `merge_indptr` を取り出して Params に入れる。
  - `info.split_kv` のときは `tmp_v = float_buf + v_offset`, `tmp_s = float_buf + s_offset`、そうでなければ nullptr。
  - 参考実装: `csrc/batch_prefill_paged.cuh`, `csrc/batch_decode.cu`

## 3. 実行関数
```
template <bool SAME_KV_STRIDES, uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          PosEncodingMode, bool USE_FP16_QK_REDUCTION, MaskMode, typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(Params, DTypeO* tmp_v, float* tmp_s, bool enable_pdl, cudaStream_t);

template <uint32_t HEAD_DIM, PosEncodingMode, typename AttentionVariant, typename Params>
cudaError_t BatchDecodeWithPagedKVCacheDispatched(Params, DTypeO* tmp_v, float* tmp_s, bool enable_pdl, cudaStream_t);
```
- `CTA_TILE_Q` は `plan_info.cta_tile_q` に従って `DISPATCH_CTA_TILE_Q` で切り替える(head_dim 64 では 16 / 64 / 128 を実体化)。
- `tmp_v != nullptr` なら分割カーネル + `VariableLengthMergeStates` / `MergeStates` が走る(`partition_kv` は関数内で設定される)。
- GQA のグループサイズは decode 側が `DISPATCH_GQA_GROUP_SIZE` で内部的に切り替える。MHA ではグループサイズ 1。

## 4. パラメータ構造体
- `BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeO, IdType>` (`attention/default_prefill_params.cuh`)
  - コンストラクタ: `(q, paged_kv, maybe_custom_mask, q_indptr(device), maybe_mask_indptr, maybe_q_rope_offset, o, lse, maybe_alibi_slopes, num_qo_heads, q_stride_n, q_stride_h, window_left, logits_soft_cap, sm_scale, rope_scale, rope_theta)`
  - plan の結果は後から代入する: `request_indices`, `qo_tile_indices`, `kv_tile_indices`, `merge_indptr`, `o_indptr`,
    `kv_chunk_size_ptr`, `padded_batch_size`, `max_total_num_rows (= total_num_rows)`, `block_valid_mask` / `total_num_rows`(CUDA Graph 時のみ)
- `BatchDecodeParams<DTypeQ, DTypeKV, DTypeO, IdType>` (`attention/default_decode_params.cuh`)
  - コンストラクタ: `(q, q_rope_offset, paged_kv, o, lse, maybe_alibi_slopes, num_qo_heads, q_stride_n, q_stride_h, window_left, logits_soft_cap, sm_scale, rope_scale, rope_theta)`
  - plan の結果: `request_indices`, `kv_tile_indices`, `o_indptr`, `kv_chunk_size_ptr`, `padded_batch_size`, `block_valid_mask`
- softmax のスケールは `sm_scale` で渡す(Variant 内で `log2e` を掛けて exp2 で計算する)。

## 5. テンプレート引数
| 引数 | 今回の値 |
|---|---|
| dtype | `nv_bfloat16`(Q/KV/O 共通)、`IdType = int32_t` |
| head_dim | 64(QK・VO とも。テンプレート引数) |
| `PosEncodingMode` | `kNone`(RoPE はカーネルに融合しない) |
| `MaskMode` (prefill) | `kCausal`。causal マスクは「クエリは KV の末尾 qo_len 個に対応する」前提(既存 prefix がある prefill でも正しい) |
| `USE_FP16_QK_REDUCTION` | false |
| `SAME_KV_STRIDES` | true(K/V を同一形状の別プールにする) |
| Variant | `DefaultAttention<use_custom_mask=false, use_sliding_window, use_logits_soft_cap=false, use_alibi=false>` (`attention/variants.cuh`) |

## 6. Sliding Window (`window_left`)
- マスク式 (`DefaultAttention::LogitsMask`): `kv_idx + qo_len + window_left >= kv_len + qo_idx`
  - クエリの絶対位置を `p = kv_len - qo_len + qo_idx` とすると、参照できるのは `kv_idx ∈ [p - window_left, p]`。
  - **`window_left` は左側の文脈トークン数**(ウィンドウ幅は `window_left + 1`)。`window_left = 0` なら自分自身だけを見る。
  - `tests/attention/test_sliding_window.py` も `k[-(window_left + 1):]` をリファレンスにしている。
- **無効化は `window_left = -1`**(負値)。Variant 内で `window_left = kv_len` に置き換わる。
- 注意: `use_sliding_window` はテンプレート引数だが、prefill カーネルは `use_sliding_window=false` でも
  `variant.window_left` を使って KV 走査の開始位置を飛ばす (`prefill.cuh` の `kv_start_idx`)。
  そのため `window_left >= 0` のときは必ず `use_sliding_window=true` で実体化した版を使う(Python も `window_left >= 0` で切り替えている)。
- decode カーネル (`decode.cuh`) はウィンドウ外の KV を**読み飛ばさず**、要素ごとにマスクするだけ
  → 結果は正しいが、計算量はキャッシュ全長に比例する。

## 7. Workspace
- 3 つ: float 用デバイスバッファ、int 用デバイスバッファ、int 用**ピン留めホスト**バッファ(int と同じサイズ)。
- Python の既定: float 128 MiB(ユーザが確保)、int 8 MiB、pinned 8 MiB (`flashinfer/decode.py`, `flashinfer/prefill.py`)。
- 必要量は `PrefillPlanWorkspaceSize(...)` / `DecodePlanWorkspaceSize(...)` で計算できる(plan と同じ引数で、カウントのみ行う)。
  - float 側は KV 分割時の部分出力 `tmp_v` / `tmp_s` だけで、`num_qo_heads × 行数 × head_dim × 4B` 程度。
  - int 側はタイル数に比例する。
- 足りない場合は plan 内の `AlignedAllocator` が `flashinfer::Error`("Increase the workspace buffer size") を投げる。

## 8. page_size
- `paged_kv_t::page_size` は実行時の `uint_fastdiv`。カーネルもスケジューラも任意の正整数を受け付ける
  (FlashInfer のテストでは 1, 5, 16 なども使っている)。
  → page_size は**実行時引数**として扱う(コンパイル時定数にする理由がない)。
- スケジューラは最小分割サイズを `max(128 / page_size, 1)` ページとしている。

## 9. 右ウィンドウ (`window_right`) — 追加調査
- 今回使っている FA2 系の C++ カーネル(`prefill.cuh` / `decode.cuh`)と `DefaultAttention` には `window_right` が**ない**。
  - `window_right` があるのは CuTe-DSL の Python バックエンド (`flashinfer/attention/cute_dsl/`, `prims_ts/`) と
    SM120 の NVFP4 カーネル (`attention/sm120/nvfp4_attention_sm120/common/params.h` の `window_size_right`) だけ。
- カスタム Variant で一般の右ウィンドウを実現する場合:
  - `MaskMode::kCustom` で実体化すると、全 KV タイルで `Variant::LogitsMask` が呼ばれる (`prefill.cuh:4176`)。
    カーネル自身はカスタムマスクのビット列を読まない(読むのは `DefaultAttention` だけ)。
  - `MaskMode::kNone` では、最後のタイルとウィンドウ左端のタイルにしかマスクが適用されない
    (`iter >= mask_iteration || iter < window_iteration`)。そのため途中のタイルの右側マスクが効かず、使えない。
- **今回の前提(ストリーミング: 右コンテキストが揃ったクエリだけを渡す)では、カスタム Variant は不要**:
  - クエリはキャッシュ末尾から R = window_right だけ遅れる。クエリ i の位置は `p = kv_len - R - qo_len + i`。
  - 参照範囲は `[p - L, p + R]`。右端 `p + R = kv_len - qo_len + i` は FlashInfer の causal マスク(末尾揃え)の右端と一致する。
  - 左端は `window_left = L + R` を渡すと `kv_len - qo_len + i - (L + R) = p - L` になり一致する。
  - decode カーネルのマスク `kv_idx >= kv_len - 1 - window_left`(上限は kv_len - 1)も、`p = kv_len - 1 - R`、`window_left = L + R` で同じ範囲になる。
  - つまり既存の `MaskMode::kCausal` + `DefaultAttention<…, use_sliding_window, …>` で厳密に同じ計算になる。
    ウィンドウ左端の KV 読み飛ばしや、plan の KV 分割計算もそのまま正しく効く。
- この前提から外れるケースでは、カスタム Variant(`kCustom`)か `MaskMode::kNone` が必要になる。
  - 例: ストリーム終端で、右コンテキストが途中で切れた最後の R 個のクエリを計算するとき。
  - この場合、各クエリの参照範囲は `[p - L, kv_len - 1]` で末尾揃えの非 causal になる。`MaskMode::kNone` + `window_left = L` で表せる。

## 10. 共有プレフィックス(cascade attention)— ビームサーチ向け調査
- Python: `flashinfer/cascade.py` の `MultiLevelCascadeAttentionWrapper`。
  `BatchDecodeWithSharedPrefixPagedKVCacheWrapper` と `BatchPrefillWithSharedPrefixPagedKVCacheWrapper` は deprecated 予定。
  - 全レベルの KV を 1 つのページプールに置く前提で、レベルごとに `BatchPrefillWithPagedKVCacheWrapper` を `return_lse=True` で実行し、`merge_state_in_place` で統合する(`run()` の実装を参照)。
  - plan の既定は `causal=False`(上位レベルのクエリは共有部分をすべて見る)。
- C++: 専用のカーネルはない。部品として使えるのは次のとおり。
  - `BatchPrefillWithPagedKVCacheDispatched`(`params.lse` に LSE を出力)
  - `attention/cascade.cuh` の `MergeState` / `MergeStateInPlace` / `MergeStates` / `VariableLengthMergeStates`
- ビームサーチへの適用可否と、組み込む場合の課題は README の「FlashInfer の共有プレフィックス向け機能」を参照。
  課題は次の 3 つ: LSE の出力、共通接頭辞ページの計算、sliding window の扱い(上位レベルの W クエリは同じ位置なのに、FlashInfer の window マスクは末尾揃えで別位置とみなす)。
