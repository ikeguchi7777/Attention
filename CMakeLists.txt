cmake_minimum_required(VERSION 3.24)

# RTX 4060 Ti = compute capability 8.9. Override with -DCMAKE_CUDA_ARCHITECTURES=... for other GPUs.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

project(flashinfer_paged_attention LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(FLASHINFER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/flashinfer" CACHE PATH "FlashInfer source checkout")
if(NOT EXISTS "${FLASHINFER_DIR}/include/flashinfer/attention/prefill.cuh")
  message(FATAL_ERROR "FlashInfer headers not found under FLASHINFER_DIR=${FLASHINFER_DIR}")
endif()

# FlashInfer is header-only for these kernels. Like its own JIT build, put the vendored CCCL ahead
# of the copy bundled with the CUDA toolkit.
add_library(flashinfer_headers INTERFACE)
target_include_directories(flashinfer_headers INTERFACE
  "${FLASHINFER_DIR}/3rdparty/cccl/libcudacxx/include"
  "${FLASHINFER_DIR}/3rdparty/cccl/cub"
  "${FLASHINFER_DIR}/3rdparty/cccl/thrust"
  "${FLASHINFER_DIR}/include")
target_compile_definitions(flashinfer_headers INTERFACE FLASHINFER_ENABLE_BF16 FLASHINFER_ENABLE_F16)
target_compile_options(flashinfer_headers INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>)

add_library(attention STATIC
  attention.cu
  paged_kv_cache.cu
  fi_decode.cu
  fi_prefill_nowindow.cu
  fi_prefill_window.cu)
target_include_directories(attention PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(attention PUBLIC flashinfer_headers)

add_executable(test_attention test_attention.cu)
target_link_libraries(test_attention PRIVATE attention)

add_executable(bench_page_size bench_page_size.cu)
target_link_libraries(bench_page_size PRIVATE attention)

add_executable(beam_search_sample beam_search_sample.cu)
target_link_libraries(beam_search_sample PRIVATE attention)
