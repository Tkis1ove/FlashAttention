#include <cute/tensor.hpp>

template <typename T, int CTA_M, int CTA_N, class TmaLoad, class GmemTensor>
__global__ void tma_load_kernel(CUTE_GRID_CONSTANT const TmaLoad tma_load, GmemTensor gmem_tensor) {
    using namespace cute;
    constexpr int tma_transaction_bytes = CTA_M * CTA_N * sizeof(T);

    __shared__ T smem_data[CTA_M * CTA_N];
    __shared__ uint64_t tma_load_mbar;

    auto smem_layout = make_layout(make_shape(CTA_M, CTA_N), LayoutRight{});
    auto smem_tensor = make_tensor(make_smem_ptr(smem_data), smem_layout);

    if (threadIdx.x == 0) {
        auto gmem_tensor_coord = tma_load.get_tma_tensor(shape(gmem_tensor));

        auto gmem_tensor_coord_cta = local_tile(
            gmem_tensor_coord,
            Tile<Int<CTA_M>, Int<CTA_N>>{},
            make_coord(blockIdx.x, blockIdx.y));

        initialize_barrier(tma_load_mbar, 1);

        set_barrier_transaction_bytes(tma_load_mbar, tma_transaction_bytes);

        auto tma_load_per_cta = tma_load.get_slice(Int<0>{});
        copy(tma_load.with(tma_load_mbar),
             tma_load_per_cta.partition_S(gmem_tensor_coord_cta),
             tma_load_per_cta.partition_D(smem_tensor));
    }

    __syncthreads();
    wait_barrier(tma_load_mbar, 0);
}

template <typename T, int CTA_M, int CTA_N>
void host_fn(T* data, int M, int N) {
    using namespace cute;

    // create the GMEM tensor
    auto gmem_layout = make_layout(make_shape(M, N), LayoutRight{});
    auto gmem_tensor = make_tensor(make_gmem_ptr(data), gmem_layout);

    // craete the SMEM layout
    auto smem_layout = make_layout(make_shape(CTA_M, CTA_N), LayoutRight{});

    // create the TMA object
    auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, gmem_tensor, smem_layout);

    // launch the kernel
    tma_load_kernel<CTA_M, CTA_N>
                   <<<dim3{M / CTA_M, N / CTA_N, 1}, 1>>>
                   (tma_load, gmem_tensor, smem_layout);
}