#!/usr/bin/env python3
"""
Native vs emulated FP64 DGEMM on one GPU, driven straight through ctypes
against the cuBLAS 13 / CUDA runtime DLLs that ship with
YaneuraOu-Deep-TensorRT. No CUDA Toolkit install, no compiler.

The emulation mode is selected by the caller through cuBLAS environment
variables, which cuBLAS reads at handle creation. So one process = one
mode; run_all.py does the spawning.
"""
import argparse
import ctypes as C
import json
import math
import os
import sys
import time

import numpy as np

CUDA_DIR = os.environ.get("FP64EMU_CUDA_DIR", r"C:\shogi\fukauraou_v940")
os.add_dll_directory(CUDA_DIR)

cudart = C.CDLL("cudart64_13.dll")
cublas = C.CDLL("cublas64_13.dll")

cudart.cudaMalloc.restype = C.c_int
cudart.cudaMalloc.argtypes = [C.POINTER(C.c_void_p), C.c_size_t]
cudart.cudaFree.restype = C.c_int
cudart.cudaFree.argtypes = [C.c_void_p]
cudart.cudaMemcpy.restype = C.c_int
cudart.cudaMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
cudart.cudaDeviceSynchronize.restype = C.c_int
cudart.cudaGetErrorString.restype = C.c_char_p
cudart.cudaGetErrorString.argtypes = [C.c_int]
cudart.cudaEventCreate.restype = C.c_int
cudart.cudaEventCreate.argtypes = [C.POINTER(C.c_void_p)]
cudart.cudaEventRecord.restype = C.c_int
cudart.cudaEventRecord.argtypes = [C.c_void_p, C.c_void_p]
cudart.cudaEventSynchronize.restype = C.c_int
cudart.cudaEventSynchronize.argtypes = [C.c_void_p]
cudart.cudaEventElapsedTime.restype = C.c_int
cudart.cudaEventElapsedTime.argtypes = [C.POINTER(C.c_float), C.c_void_p, C.c_void_p]
cudart.cudaMemGetInfo.restype = C.c_int
cudart.cudaMemGetInfo.argtypes = [C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)]

H2D, D2H = 1, 2
OP_N = 0

cublas.cublasCreate_v2.restype = C.c_int
cublas.cublasCreate_v2.argtypes = [C.POINTER(C.c_void_p)]
cublas.cublasDestroy_v2.restype = C.c_int
cublas.cublasDestroy_v2.argtypes = [C.c_void_p]
cublas.cublasGetVersion_v2.restype = C.c_int
cublas.cublasGetVersion_v2.argtypes = [C.c_void_p, C.POINTER(C.c_int)]
cublas.cublasDgemm_v2.restype = C.c_int
cublas.cublasDgemm_v2.argtypes = [
    C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
    C.POINTER(C.c_double), C.c_void_p, C.c_int,
    C.c_void_p, C.c_int,
    C.POINTER(C.c_double), C.c_void_p, C.c_int,
]


def ck(rc, what):
    if rc != 0:
        msg = cudart.cudaGetErrorString(rc)
        raise RuntimeError("%s -> rc=%d (%s)" % (what, rc, msg.decode() if msg else "?"))


def ckb(rc, what):
    if rc != 0:
        raise RuntimeError("%s -> cublas status %d" % (what, rc))


def gen_matrix(n, dist, rng, phi):
    """Column-major n*n matrix, built in column blocks so temporaries stay
    bounded even at n=16384 (2.1 GiB per matrix)."""
    out = np.empty((n, n), order="F", dtype=np.float64)
    cols = max(1, (1 << 23) // n)
    for j0 in range(0, n, cols):
        j1 = min(n, j0 + cols)
        w = j1 - j0
        if dist == "easy":
            out[:, j0:j1] = rng.uniform(-1.0, 1.0, size=(n, w))
        else:
            # Wide dynamic range. This is what drives the mantissa-bit
            # requirement of the fixed-point (Ozaki-style) split.
            out[:, j0:j1] = rng.standard_normal((n, w)) * np.exp2(
                rng.uniform(-phi, phi, size=(n, w)))
    return out


_HAS_FMA = hasattr(math, "fma")


def exact_dot(a, b):
    """Correctly-rounded exact dot product.

    two-product via fma recovers the rounding error of each a*b exactly, so
    fsum over the (product, error) pairs is the exact sum of the real
    products. Falls back to fsum of rounded products without fma.
    """
    al = a.tolist()
    bl = b.tolist()
    if not _HAS_FMA:
        return math.fsum([x * y for x, y in zip(al, bl)])
    terms = []
    ap = terms.append
    fma = math.fma
    for x, y in zip(al, bl):
        p = x * y
        ap(p)
        e = fma(x, y, -p)
        if e:
            ap(e)
    return math.fsum(terms)


def accuracy(A, B, Cout, n, nsample, rng):
    """Relative error of Cout vs the exact product, over sampled entries."""
    ii = rng.integers(0, n, size=nsample)
    jj = rng.integers(0, n, size=nsample)
    rels = []
    worst = 0.0
    for i, j in zip(ii.tolist(), jj.tolist()):
        ref = exact_dot(A[i, :], B[:, j])
        got = float(Cout[i, j])
        denom = abs(ref)
        if denom == 0.0:
            continue
        r = abs(got - ref) / denom
        rels.append(r)
        worst = max(worst, r)
    if not rels:
        return None
    rels.sort()
    return {
        "samples": len(rels),
        "rel_err_max": worst,
        "rel_err_median": rels[len(rels) // 2],
        "rel_err_mean": sum(rels) / len(rels),
    }


def run_size(handle, n, dist, reps, warmup, nsample, phi, seed, want_acc):
    rng = np.random.default_rng(seed)
    A = gen_matrix(n, dist, rng, phi)
    B = gen_matrix(n, dist, rng, phi)

    nbytes = n * n * 8
    dA, dB, dC = C.c_void_p(), C.c_void_p(), C.c_void_p()
    ck(cudart.cudaMalloc(C.byref(dA), nbytes), "cudaMalloc A")
    ck(cudart.cudaMalloc(C.byref(dB), nbytes), "cudaMalloc B")
    ck(cudart.cudaMalloc(C.byref(dC), nbytes), "cudaMalloc C")
    ck(cudart.cudaMemcpy(dA, A.ctypes.data_as(C.c_void_p), nbytes, H2D), "H2D A")
    ck(cudart.cudaMemcpy(dB, B.ctypes.data_as(C.c_void_p), nbytes, H2D), "H2D B")

    alpha, beta = C.c_double(1.0), C.c_double(0.0)

    def gemm():
        return cublas.cublasDgemm_v2(handle, OP_N, OP_N, n, n, n,
                                     C.byref(alpha), dA, n, dB, n,
                                     C.byref(beta), dC, n)

    for _ in range(warmup):
        ckb(gemm(), "dgemm warmup")
    ck(cudart.cudaDeviceSynchronize(), "sync warmup")

    ev0, ev1 = C.c_void_p(), C.c_void_p()
    ck(cudart.cudaEventCreate(C.byref(ev0)), "eventCreate0")
    ck(cudart.cudaEventCreate(C.byref(ev1)), "eventCreate1")

    ms = []
    for _ in range(reps):
        ck(cudart.cudaEventRecord(ev0, None), "record0")
        ckb(gemm(), "dgemm")
        ck(cudart.cudaEventRecord(ev1, None), "record1")
        ck(cudart.cudaEventSynchronize(ev1), "eventSync")
        el = C.c_float()
        ck(cudart.cudaEventElapsedTime(C.byref(el), ev0, ev1), "elapsed")
        ms.append(float(el.value))

    free_b, total_b = C.c_size_t(), C.c_size_t()
    cudart.cudaMemGetInfo(C.byref(free_b), C.byref(total_b))
    mem_used = (total_b.value - free_b.value) / 1048576.0

    acc = None
    if want_acc:
        Cout = np.empty((n, n), order="F", dtype=np.float64)
        ck(cudart.cudaMemcpy(Cout.ctypes.data_as(C.c_void_p), dC, nbytes, D2H), "D2H C")
        acc = accuracy(A, B, Cout, n, nsample, np.random.default_rng(seed + 991))
        del Cout

    cudart.cudaFree(dA)
    cudart.cudaFree(dB)
    cudart.cudaFree(dC)

    ms.sort()
    flops = 2.0 * n * n * n
    median_ms = ms[len(ms) // 2]
    best_ms = ms[0]
    return {
        "n": n,
        "dist": dist,
        "reps": reps,
        "ms_median": median_ms,
        "ms_best": best_ms,
        "ms_all": ms,
        "tflops_median": flops / (median_ms * 1e-3) / 1e12,
        "tflops_best": flops / (best_ms * 1e-3) / 1e12,
        "gpu_mem_used_MiB": mem_used,
        "accuracy": acc,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="2048,4096,8192,16384")
    p.add_argument("--dists", default="easy,hard")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--samples", type=int, default=48)
    p.add_argument("--phi", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=20261004)
    p.add_argument("--no-accuracy", action="store_true")
    p.add_argument("--label", default="unlabeled")
    p.add_argument("--out", default="")
    a = p.parse_args()

    handle = C.c_void_p()
    ckb(cublas.cublasCreate_v2(C.byref(handle)), "cublasCreate")
    ver = C.c_int()
    cublas.cublasGetVersion_v2(handle, C.byref(ver))

    env_keys = ["CUBLAS_EMULATE_DOUBLE_PRECISION", "CUBLAS_EMULATION_STRATEGY",
                "CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT",
                "CUBLAS_EMULATE_SINGLE_PRECISION"]
    rec = {
        "label": a.label,
        "cublas_version": ver.value,
        "cuda_dir": CUDA_DIR,
        "has_fma": _HAS_FMA,
        "phi": a.phi,
        "env": {k: os.environ[k] for k in env_keys if k in os.environ},
        "runs": [],
    }

    for dist in [d for d in a.dists.split(",") if d]:
        for n in [int(s) for s in a.sizes.split(",") if s]:
            t0 = time.time()
            try:
                r = run_size(handle, n, dist, a.reps, a.warmup, a.samples,
                             a.phi, a.seed, not a.no_accuracy)
                r["wall_s"] = round(time.time() - t0, 2)
                rec["runs"].append(r)
                acc = r["accuracy"]
                print("[%-20s] %-4s n=%-6d %9.2f ms %8.3f TFLOPS relerr_med=%s"
                      % (a.label, dist, n, r["ms_median"], r["tflops_median"],
                         ("%.3e" % acc["rel_err_median"]) if acc else "-"),
                      flush=True)
            except Exception as e:
                rec["runs"].append({"n": n, "dist": dist, "error": str(e)})
                print("[%-20s] %-4s n=%-6d ERROR %s" % (a.label, dist, n, e),
                      flush=True)

    cublas.cublasDestroy_v2(handle)

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
