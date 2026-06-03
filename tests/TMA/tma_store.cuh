#include <cute/tensor.hpp>
#include <cute/numeric/numeric_types.hpp>

template <typename T, int CTA_M, int CTA_N, class TmaStore, class GmemTensor, class SmemLayout>
__global__ void tma_store_kernel(__grid_constant__ const TmaStore tma_store, GmemTensor gmem_tensor, SmemLayout smem_layout) {
    using namespace cute;
    __shared__ T smem_data[CTA_M * CTA_N];

    auto smem_tensor = make_tensor(make_smem_ptr(smem_data), smem_layout);

    for (int i = 0; i < CTA_M * CTA_N; ++i) {
        *(smem_data + i) = (T)i;
    }

    __syncthreads();
    tma_store_fence();

    if (threadIdx.x == 0) {
        auto gmem_tensor_coord = tma_store.get_tma_tensor(shape(gmem_tensor));

        auto gmem_tensor_coord_cta = local_tile(
            gmem_tensor_coord,
            Tile<Int<CTA_M>, Int<CTA_N>>{},
            make_coord(blockIdx.x, blockIdx.y));

        auto tma_tensor_per_cta = tma_store.get_slice(Int<0>{});
        copy(tma_store,
             tma_tensor_per_cta.partition_S(smem_tensor),
             tma_tensor_per_cta.partition_D(gmem_tensor_coord_cta));
        tma_store_arrive();
    }
    tma_store_wait<0>();
}

template <typename T, int CTA_M, int CTA_N>
void tma_store(T* data, int M, int N) {
    using namespace cute;

    // create the GMEM tensor
    auto gmem_layout = make_layout(make_shape(M, N), LayoutRight{});
    auto gmem_tensor = make_tensor(make_gmem_ptr(data), gmem_layout);

    // create the SMEM tensor
    auto smem_layout = make_layout(make_shape(Int<CTA_M>{}, Int<CTA_N>{}), LayoutRight{});

    // create the TMA object
    auto tma_store = make_tma_copy(SM90_TMA_STORE{}, gmem_tensor, smem_layout);

    // launch the kernel
    tma_store_kernel<T, CTA_M, CTA_N, decltype(tma_store), decltype(gmem_tensor), decltype(smem_layout)>
                   <<<dim3{M / CTA_M, N / CTA_N, 1}, 1>>>
                   (tma_store, gmem_tensor, smem_layout);
    cudaDeviceSynchronize();
}