TMA是Hopper架构引入的新特性，用来在global memory和shared memory之间异步搬运数据。对比之前的cp.saync，TMA有两大优势：
1. 能更好地使warp-specialized，提高GPU利用率。
2. 单线程launch TMA，节省寄存器，同时自动处理断言。
本文章主要介绍如何用cute来使用TMA。
## TMA Load
### Example task

为了演示 TMA 加载的用法，我们考虑一个简单的任务：对一个二维行优先矩阵进行分块（tiling）。我们给定一个形状为 $[m, n]$ 的矩阵 A，以及两个正整数 CTA_M 和 CTA_N。注意，CTA_M 和 CTA_N 在编译时是已知的，而 m 和 n 则在运行时通过矩阵 A 给出。为简单起见，我们暂且假设 $m \% CTA_M == n \% CTA_N == 0$ ，不过稍后我们会看到这个要求是可以放宽的。

我们令gridSize为 $[m/CTA_M,n/CTA_N,1]$ ，其中第 $(i,j)$ 个 CTA 的smem存放来自 矩阵A 的第 $(i,j)$ 个形状为 $[CTA\_M,CTA\_N]$ 的分块。

我们使用TMA来完成这个任务。在 CuTe 中，TMA 加载操作通过两步来实现。第一步是在**host**中构造 TMA copy descriptor，第二步是在**kernel**中使用这个descriptor执行实际的 TMA 操作。
### Host code


## TMA Store

TMA Store 将数据从 SMEM 异步搬运到 GMEM。与 TMA Load 使用 mbar（memory barrier）不同，TMA Store 使用更简单的 **commit_group / wait_group** 机制。

### Kernel 端的完整流程

```cpp
// 1. 线程写 SMEM
for (int i = 0; i < CTA_M * CTA_N; ++i) {
    smem_data[i] = (T)i;
}

// 2. 线程间同步 — 确保所有线程写完了
__syncthreads();

// 3. fence — 让 TMA 的 async proxy 能看到 SMEM 写入
tma_store_fence();

// 4. 发起 TMA Store（单线程）
if (threadIdx.x == 0) {
    auto tma_store_per_cta = tma_store.get_slice(Int<0>{});
    copy(tma_store,
         tma_store_per_cta.partition_S(smem_tensor),
         tma_store_per_cta.partition_D(gmem_tensor_coord_cta));
    tma_store_arrive();
}

// 5. 等待完成
tma_store_wait<0>();
```

### `tma_store_fence()` — `fence.proxy.async.shared::cta`

线程通过 LSU（Load-Store Unit）写 SMEM，TMA 硬件通过 async proxy 读 SMEM——这两条路径是独立的。`__syncthreads()` 只保证 LSU 侧所有线程能看到写入，但 async proxy 可能还看不到（数据卡在 store buffer 里）。`fence.proxy.async.shared::cta` 把 CTA 范围内的 SMEM store 刷到 async proxy 可见域。

```
线程 store → [LSU → store buffer → SMEM]
TMA  load ← [async proxy] ───→ SMEM
                              ↑
              fence 打通这条通路
```

### `tma_store_arrive()` — `cp.async.bulk.commit_group`

把之前发出的所有 TMA store 操作打成一个"完成组"。每调用一次 `commit_group`，内部组号递增（0 → 1 → 2 → ...）。

### `tma_store_wait<Count>()` — `cp.async.bulk.wait_group.read Count`

阻塞直到**还剩 ≤ Count 个组未完成**。例如：
- `tma_store_wait<0>()` — 等所有组完成
- `tma_store_wait<1>()` — 最多允许 1 个组 pending，其余完成

多轮流水时可以不等最后一轮就复用 SMEM：
```cpp
copy(TMA_STORE, ...); commit_group;  // 组 0
copy(TMA_STORE, ...); commit_group;  // 组 1
copy(TMA_STORE, ...); commit_group;  // 组 2

wait_group<1>();  // 组 0、1 完成，组 2 可以 pending
// 此时可以安全覆盖组 0、1 用过的 SMEM 区域
wait_group<0>();  // 全部完成
```

### 与 TMA Load 的对比

| | TMA Load | TMA Store |
|---|---|---|
| 同步机制 | mbar（异步 barrier） | commit_group / wait_group |
| 单向性 | 需 `with(mbar)` 绑定 arrive 通知 | 不需绑定，用 group 机制追踪完成 |
| 完成通知 | TMA 硬件搬完后自动 arrive | `commit_group` + `wait_group` |
| fence | 不需要（Load 不写 SMEM） | 需要 `tma_store_fence()` |
