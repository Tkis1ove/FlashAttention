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



