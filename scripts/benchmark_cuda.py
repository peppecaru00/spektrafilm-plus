"""Benchmark comparing CPU vs CUDA simulation performance across resolutions."""
from __future__ import annotations

import time
import numpy as np

from spektrafilm.cuda.device import is_cuda_available, get_device_info, synchronize
from spektrafilm.runtime import init_params, simulate
from spektrafilm.utils.numba_warmup import warmup


def run_benchmark():
    print("=" * 70)
    print("Spektrafilm CPU vs CUDA Simulation Benchmark")
    print("=" * 70)

    if not is_cuda_available():
        print("CUDA is NOT available. Exiting benchmark.")
        return

    dev_info = get_device_info()
    print(f"GPU Device: {dev_info['device_name']}")
    print(f"VRAM: {dev_info['total_memory_gb']} GB")
    print(f"Compute Capability: {dev_info['compute_capability']}")
    print("-" * 70)

    print("Warming up Numba JIT and CUDA kernels...")
    warmup()
    dummy = np.random.uniform(0.2, 0.8, (64, 64, 3)).astype(np.float64)
    p_warm = init_params()
    p_warm.settings.device = "cuda"
    _ = simulate(dummy, p_warm)
    p_warm.settings.device = "cpu"
    _ = simulate(dummy, p_warm)
    print("Warmup complete.\n")

    resolutions = [
        ("Preview (640x480, 0.3 MP)", (480, 640, 3)),
        ("Standard (1000x667, 0.67 MP)", (667, 1000, 3)),
        ("Large (2048x1536, 3.1 MP)", (1536, 2048, 3)),
    ]

    results = []

    for name, shape in resolutions:
        print(f"Benchmarking {name} ...")
        img = np.random.uniform(0.1, 0.9, shape).astype(np.float64)

        # Common params
        p_cpu = init_params()
        p_cpu.settings.device = "cpu"
        p_cpu.film_render.grain.amount = 0.0
        p_cpu.print_render.glare.active = False

        p_cuda = init_params()
        p_cuda.settings.device = "cuda"
        p_cuda.film_render.grain.amount = 0.0
        p_cuda.print_render.glare.active = False

        # --- CPU Benchmark ---
        # 1 run warmup
        _ = simulate(img, p_cpu)
        n_iters = 3 if shape[0] <= 1000 else 2
        cpu_times = []
        cpu_out = None
        for i in range(n_iters):
            t0 = time.perf_counter()
            cpu_out = simulate(img, p_cpu)
            t1 = time.perf_counter()
            cpu_times.append(t1 - t0)

        cpu_mean = np.mean(cpu_times)
        print(f"  CPU Mean:  {cpu_mean:.3f} s (best: {min(cpu_times):.3f} s)")

        # --- CUDA Benchmark ---
        # 1 run warmup
        _ = simulate(img, p_cuda)
        synchronize()
        cuda_times = []
        cuda_out = None
        for i in range(n_iters):
            synchronize()
            t0 = time.perf_counter()
            cuda_out = simulate(img, p_cuda)
            synchronize()
            t1 = time.perf_counter()
            cuda_times.append(t1 - t0)

        cuda_mean = np.mean(cuda_times)
        speedup = cpu_mean / cuda_mean if cuda_mean > 0 else 0
        print(f"  CUDA Mean: {cuda_mean:.3f} s (best: {min(cuda_times):.3f} s) -> Speedup: {speedup:.2f}x")

        # Fidelity check
        diff = np.abs(cpu_out - cuda_out)
        print(f"  Max Diff:  {diff.max():.2e}, Mean Diff: {diff.mean():.2e}\n")

        results.append({
            "name": name,
            "shape": f"{shape[1]}x{shape[0]}",
            "mp": f"{shape[0] * shape[1] / 1e6:.2f} MP",
            "cpu_time": cpu_mean,
            "cuda_time": cuda_mean,
            "speedup": speedup,
            "max_diff": diff.max(),
            "mean_diff": diff.mean(),
        })

    print("=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    header = f"| {'Resolution':<28} | {'Megapixels':<10} | {'CPU (s)':<10} | {'CUDA (s)':<10} | {'Speedup':<8} |"
    sep = f"|{'-' * 30}|{'-' * 12}|{'-' * 12}|{'-' * 12}|{'-' * 10}|"
    print(header)
    print(sep)
    for r in results:
        row = f"| {r['name']:<28} | {r['mp']:<10} | {r['cpu_time']:<10.3f} | {r['cuda_time']:<10.3f} | {r['speedup']:<7.2f}x |"
        print(row)
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
