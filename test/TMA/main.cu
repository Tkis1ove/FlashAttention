#include <stdio.h>

#include "tma_load.cuh"
#include "tma_store.cuh"

int main(int argc, char** argv) {
    int M = 256;
    int N = 512;

    constexpr int CTA_M = 128;
    constexpr int CTA_N = 64;

    using bf16 = cute::bfloat16_t;

    bf16* h_data = (bf16*)malloc(sizeof(bf16) * M * N);
    bf16* d_data;
    bf16* d_out;
    cudaMalloc(&d_data, sizeof(bf16) * M * N);
    cudaMalloc(&d_out, sizeof(bf16) * M * N);

    for (int i = 0; i < M * N; ++i) {
        h_data[i] = (bf16)i;
    }
    cudaMemcpy(d_data, h_data, sizeof(bf16) * M * N, cudaMemcpyHostToDevice);

    // ==================== TMA Load verification ====================
    tma_load<bf16, CTA_M, CTA_N>(d_data, d_out, M, N);

    cudaMemcpy(h_data, d_out, sizeof(bf16) * M * N, cudaMemcpyDeviceToHost);
    int load_errors = 0;
    for (int i = 0; i < M * N; ++i) {
        if (h_data[i] != (bf16)i) {
            if (load_errors < 5) printf("LOAD mismatch at %d: expected %d, got %d\n", i, i, (int)h_data[i]);
            ++load_errors;
        }
    }
    printf("TMA Load: %s (%d errors)\n", load_errors == 0 ? "PASS" : "FAIL", load_errors);

    // ==================== TMA Store verification ====================
    tma_store<bf16, CTA_M, CTA_N>(d_data, M, N);

    cudaMemcpy(h_data, d_data, sizeof(bf16) * M * N, cudaMemcpyDeviceToHost);
    int store_errors = 0;
    for (int bx = 0; bx < M / CTA_M; ++bx) {
        for (int by = 0; by < N / CTA_N; ++by) {
            for (int m = 0; m < CTA_M; ++m) {
                for (int n = 0; n < CTA_N; ++n) {
                    bf16 expected = (bf16)(m * CTA_N + n);
                    bf16 got = h_data[(bx * CTA_M + m) * N + (by * CTA_N + n)];
                    if (got != expected) {
                        if (store_errors < 5)
                            printf("STORE mismatch at tile(%d,%d) smem(%d,%d): expected %d, got %d\n",
                                   bx, by, m, n, (int)expected, (int)got);
                        ++store_errors;
                    }
                }
            }
        }
    }
    printf("TMA Store: %s (%d errors)\n", store_errors == 0 ? "PASS" : "FAIL", store_errors);

    cudaFree(d_data);
    cudaFree(d_out);
    free(h_data);
}