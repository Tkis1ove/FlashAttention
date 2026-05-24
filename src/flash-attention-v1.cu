#include <stdio.h>
#include <cuda.h>
#include <cuda_bf16.h>

#include "../3rd/cutlass/include/cute/layout.hpp"

typedef __nv_bfloat16 bf16;

void __global__ flash_attention_v1_kernel(bf16* q, bf16* k, bf16* v, bf16* o) {
    
}
