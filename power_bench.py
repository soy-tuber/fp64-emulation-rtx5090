#!/usr/bin/env python3
"""Sustained-load power measurement for native vs emulated FP64 DGEMM.

Runs DGEMM back-to-back for a fixed wall-clock window while sampling
nvidia-smi, then reports achieved throughput, mean board power in the
steady-state part of the window, and GFLOP per joule.

Mode is chosen by the caller through the cuBLAS environment variables,
same as dgemm_bench.py.
"""
import argparse
import ctypes as C
import json
import os
import subprocess
import sys
import time

import numpy as np

import dgemm_bench as D

SAMPLE_MS = 200
QUERY = ("power.draw,clocks.sm,clocks.mem,temperature.gpu,"
         "utilization.gpu,utilization.memory")


def start_sampler(path):
    f = open(path, "w", encoding="utf-8")
    p = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=" + QUERY, "--format=csv,noheader,nounits",
         "-lms", str(SAMPLE_MS)],
        stdout=f, stderr=subprocess.DEVNULL)
    return p, f


def parse_samples(path, skip_head, skip_tail):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                rows.append([float(x) for x in parts[:6]])
            except ValueError:
                continue
    if len(rows) <= skip_head + skip_tail:
        return rows, rows
    return rows, rows[skip_head:len(rows) - skip_tail]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=16384)
    p.add_argument("--dist", default="easy")
    p.add_argument("--seconds", type=float, default=25.0)
    p.add_argument("--phi", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=20261004)
    p.add_argument("--label", default="power")
    p.add_argument("--out", default="")
    a = p.parse_args()

    n = a.n
    rng = np.random.default_rng(a.seed)
    A = D.gen_matrix(n, a.dist, rng, a.phi)
    B = D.gen_matrix(n, a.dist, rng, a.phi)

    handle = C.c_void_p()
    D.ckb(D.cublas.cublasCreate_v2(C.byref(handle)), "cublasCreate")

    nbytes = n * n * 8
    dA, dB, dC = C.c_void_p(), C.c_void_p(), C.c_void_p()
    D.ck(D.cudart.cudaMalloc(C.byref(dA), nbytes), "malloc A")
    D.ck(D.cudart.cudaMalloc(C.byref(dB), nbytes), "malloc B")
    D.ck(D.cudart.cudaMalloc(C.byref(dC), nbytes), "malloc C")
    D.ck(D.cudart.cudaMemcpy(dA, A.ctypes.data_as(C.c_void_p), nbytes, D.H2D), "H2D A")
    D.ck(D.cudart.cudaMemcpy(dB, B.ctypes.data_as(C.c_void_p), nbytes, D.H2D), "H2D B")
    del A, B

    alpha, beta = C.c_double(1.0), C.c_double(0.0)

    def gemm():
        return D.cublas.cublasDgemm_v2(handle, D.OP_N, D.OP_N, n, n, n,
                                       C.byref(alpha), dA, n, dB, n,
                                       C.byref(beta), dC, n)

    for _ in range(2):
        D.ckb(gemm(), "warmup")
    D.ck(D.cudart.cudaDeviceSynchronize(), "sync warmup")

    smp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "_power_%s.csv" % a.label)
    proc, fh = start_sampler(smp_path)
    time.sleep(1.0)

    t0 = time.perf_counter()
    iters = 0
    while time.perf_counter() - t0 < a.seconds:
        D.ckb(gemm(), "dgemm")
        iters += 1
        if iters % 4 == 0:
            D.ck(D.cudart.cudaDeviceSynchronize(), "sync")
    D.ck(D.cudart.cudaDeviceSynchronize(), "sync end")
    elapsed = time.perf_counter() - t0

    time.sleep(0.4)
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
    fh.close()

    D.cudart.cudaFree(dA)
    D.cudart.cudaFree(dB)
    D.cudart.cudaFree(dC)
    D.cublas.cublasDestroy_v2(handle)

    skip = max(1, int(3000 / SAMPLE_MS))
    allr, steady = parse_samples(smp_path, skip, 1)
    if not steady:
        print("no power samples captured", flush=True)
        return 1

    def col(i):
        return [r[i] for r in steady]

    mean_w = sum(col(0)) / len(steady)
    flops = 2.0 * n * n * n * iters
    tflops = flops / elapsed / 1e12
    rec = {
        "label": a.label, "n": n, "dist": a.dist,
        "env": {k: os.environ[k] for k in
                ("CUBLAS_EMULATE_DOUBLE_PRECISION", "CUBLAS_EMULATION_STRATEGY",
                 "CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT")
                if k in os.environ},
        "iters": iters, "elapsed_s": round(elapsed, 3),
        "tflops_sustained": tflops,
        "power_w_mean": mean_w,
        "power_w_max": max(col(0)),
        "clock_sm_mean": sum(col(1)) / len(steady),
        "temp_c_max": max(col(3)),
        "util_gpu_mean": sum(col(4)) / len(steady),
        "gflops_per_watt": (tflops * 1000.0) / mean_w,
        "samples_total": len(allr), "samples_steady": len(steady),
    }
    print("[%-14s] n=%d %-5s %7.3f TFLOPS  %6.1f W  %6.1f GFLOPS/W  "
          "sm=%.0fMHz tmax=%.0fC util=%.0f%%"
          % (a.label, n, a.dist, rec["tflops_sustained"], rec["power_w_mean"],
             rec["gflops_per_watt"], rec["clock_sm_mean"], rec["temp_c_max"],
             rec["util_gpu_mean"]), flush=True)

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
