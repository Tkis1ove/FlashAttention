#include <stdio.h>
#include <cuda.h>
#include <cuda_bf16.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

template <typename T, int block_m, int block_N, class q_gmem_tensor, class k_gmem_tensor, class v_gmem_tensor, class o_gmem_tensor, class smem_layout,
          class q_tma_load, class k_tma_load, class v_tma_load>
__global__ void flash_attention_v1_kernel(
    T* q, T* k, T* v, T* o,
    int B, int N, int H, int D)
{
    int bid = blockIdx.z; // batch id
    int hid = blockIdx.y; // head id
    int nid = blockIdx.x; // block id
    
    extern __shared__ T smem[];
    __shared__ uint64_t tma_load_mbar;

    size_t qkv_gmem_offset = bid * N * D * H + hid * N * D + nid * block_m * block_n; // qkv offset for mha, gqa to do.
    size_t qkv_smem_offset = block_m * block_n;

    q_smem = smem;
    k_smem = q_smem + qkv_smem_offset;
    v_smem = k_smem + qkv_smem_offset;
    o_smem = v_smem + qkv_smem_offset;

    auto q_smem_tensor = make_tensor(make_smem_ptr(q_smem), smem_layout);
    auto k_smem_tensor = make_tensor(make_smem_ptr(k_smem), smem_layout);
    auto v_smem_tensor = make_tensor(make_smem_ptr(v_smem), smem_layout);
    auto o_smem_tensor = make_tensor(make_smem_ptr(o_smem), smem_layout);

    if (threadIdx.x == 0) {
        auto
    }
}

template<typename T, int block_m, int block_n>
void flash_attention_v1(T* q, T* k, T* v, T* o, int B, int N, int H, int D) {

    Dim3 grid((N + block_m - 1) / block_m, H, B);
    Dim3 block(block_m, block_n);

    size_t shared_memory_size = block_m * block_n * sizeof(T) * 4;

    auto q_gmem_layout = make_layout(make_shape(B, H, N, D), make_stride(N * D * H, N * D, D, 1));
    auto kv_gmem_layout = make_layout(make_shape(B, H, N, D), make_stride(N * D * H, N * D, D, 1));
    auto smem_layout = make_layout(make_shape(block_m, block_n), make_stride(block_n, 1));

    auto q_gmem_tensor = make_tensor(make_gmem_ptr(q), q_gmem_layout);
    auto k_gmem_tensor = make_tensor(make_gmem_ptr(k), kv_gmem_layout);
    auto v_gmem_tensor = make_tensor(make_gmem_ptr(v), kv_gmem_layout);
    auto o_gmem_tensor = make_tensor(make_gmem_ptr(o), q_gmem_layout);

    auto q_tma_load = make_tma_load(SM90_TMA_LOAD(), q_gmem_tensor, smem_layout);
    auto k_tma_load = make_tma_load(SM90_TMA_LOAD(), k_gmem_tensor, smem_layout);
    auto v_tma_load = make_tma_load(SM90_TMA_LOAD(), v_gmem_tensor, smem_layout);

    flash_attention_v1_kernel<T, block_m, block_n, q_gmem_tensor, k_gmem_tensor, v_gmem_tensor, o_gmem_tensor, smem_layout>
        <<<grid, block, shared_memory_size>>>(q, k, v, o, B, N, H, D);
}