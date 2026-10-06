"""对比同一矩阵乘法在 CPU 与 GPU 上的耗时，验证 GPU 加速是否真正生效。

为保证对比公平：每个后端先空跑 2 次做预热（warmup），再取 5 次运行的
平均值；GPU 侧在计时前后各调用一次 torch.cuda.synchronize()，否则
测到的只是内核入队时间而非真实计算时间。
"""
import time

import torch

N = 4096
REP = 5


def bench(fn):
    for _ in range(2):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(REP):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - t0) / REP


a = torch.randn(N, N)
b = torch.randn(N, N)
t_cpu = bench(lambda: a @ b)

ag, bg = a.cuda(), b.cuda()
t_gpu = bench(lambda: ag @ bg)

print("matrix size      : %dx%d  (repeat %d, 2x warmup)" % (N, N, REP))
print("CPU matmul       : %.4f s" % t_cpu)
print("GPU matmul       : %.4f s" % t_gpu)
print("speedup          : %.1fx" % (t_cpu / t_gpu))
print("GPU memory alloc : %.1f MB" % (torch.cuda.memory_allocated() / 1024 / 1024))