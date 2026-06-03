#include <stdio.h>
#include <cuda.h>
#include <cuda_bf16.h>

#include <cute/layout.hpp>

template <typename T, int block_m, int block_N>
__global__ void flash_attention_v1_kernel(
    T* q, T* k, T* v, T* o,
    int B, int N, int H, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N * H * D;
    if (idx < total) {
        o[idx] = q[idx];  // 占位：编译通过后再替换为真正的 FlashAttention
    }
}

template<typename T, int block_m, int block_n>
void flash_attention_v1(T* q, T* k, T* v, T* o, int B, int N, int H, int D) {
    int total = B * N * H * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    flash_attention_v1_kernel<T, block_m, block_n>
        <<<blocks, threads>>>(q, k, v, o, B, N, H, D);
}