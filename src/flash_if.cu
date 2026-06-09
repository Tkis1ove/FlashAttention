#include <torch/extension.h>
#include <cuda_bf16.h>

#include "ops/flash_attention_v1.cuh"

typedef __nv_bfloat16 T;

template<typename scalar_t, int block_m, int block_n>
void flash_attention_v1(scalar_t* q, scalar_t* k, scalar_t* v, scalar_t* o,
                        int B, int N, int H, int D);

torch::Tensor flash_attention_v1_wrapper(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(),
                "inputs must be on CUDA");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                "inputs must be contiguous");

    int B = Q.size(0), N = Q.size(1), H = Q.size(2), D = Q.size(3);

    TORCH_CHECK(D == block_n, "D must be equal to block_n")

    auto O = torch::empty_like(Q);

    constexpr int block_m = 128, block_n = 128;

    flash_attention_v1<T, block_m, block_n>(
        reinterpret_cast<T*>(Q.data_ptr<at::BFloat16>()),
        reinterpret_cast<T*>(K.data_ptr<at::BFloat16>()),
        reinterpret_cast<T*>(V.data_ptr<at::BFloat16>()),
        reinterpret_cast<T*>(O.data_ptr<at::BFloat16>()),
        B, N, H, D);

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_v1", &flash_attention_v1_wrapper,
          "FlashAttention v1 forward");
}