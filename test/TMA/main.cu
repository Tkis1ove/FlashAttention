#include <stdio.h>
#include <cuda_bf16.h>

#include "TMA.cuh"

int main(int argc, char** argv) {
    int M = 256;
    int N = 512;

    constexpr int CTA_M = 128;
    constexpr int CTA_N = 64;

    using bf16 = cute::bfloat16_t;

    bf16* data_h = (bf16*)malloc(sizeof(bf16) * M * N);
    bf16* data_d;
    cudaMalloc(&data_d, sizeof(bf16) * M * N);

    for (int i = 0; i < M * N; ++i) {
        *(data_h + i) = (bf16)i;
    }

    cudaMemcpy((void*)data_d, (void*)data_h, sizeof(bf16) * M * N, cudaMemcpyHostToDevice);

    host_fn<bf16, CTA_M, CTA_N>(data_d, M, N);
}