# FP64 emulation on a GeForce RTX 5090

Native vs Ozaki-style fixed-point **FP64 emulation** (`cuBLAS` 13, CUDA 13.0 Update 2+) measured on *consumer* Blackwell, where hardware FP64 is rate-limited to 1/64 of FP32.

**Report: https://soy-tuber.github.io/fp64-emulation-rtx5090/**

Measured on `NVIDIA GeForce RTX 5090`, cuBLAS `130202` (`cublasGetVersion`), 2026-10-04.

## Headline

| | native FP64 | emulated (ADP) |
|---|---|---|
| peak throughput | 1.719 TFLOPS | **14.124 TFLOPS** |
| speedup, narrow exponent range | 1.00x | **8.22x** |
| speedup, wide exponent range | 1.00x | **5.81x** |
| relative error vs exact product | ~4e-15 | **0 to 1 ULP** |

### Energy (n=16384, 25 s sustained)

| mode | sustained TFLOPS | mean board power | GFLOP/s per W | vs native |
|---|---|---|---|---|
| native | 1.716 | 221 W | **7.8** | 1.00x |
| adp | 14.446 | 567 W | **25.5** | 3.29x |
| mant48 | 18.516 | 570 W | **32.5** | 4.19x |
| mant08 | 147.476 | 575 W | **256.4** | 33.08x |

`mant48` keeps full FP64-grade accuracy. `mant08` does not (relative error ~1e-4) and is listed to show where the hardware ceiling actually is. Native FP64 draws only 221 W of the 575 W budget: too little of the die is active, so it is clock-limited rather than power-limited, and the emulation converts that unused power budget into throughput.

The emulation is not a precision/speed trade here: it is both faster **and** more accurate than native FP64, because the fixed-point split accumulates exactly in integer arithmetic and rounds once at the end, whereas native FP64 accumulates rounding error over n terms.

## Why measure this

Published emulation numbers come mostly from data-center GPUs, where native FP64 is already fast and the gain looks modest. On a GeForce card the premise is inverted: FP64 is deliberately crippled while INT8 tensor throughput is abundant, so the ratio should be much larger. Measurements for consumer Blackwell are hard to find, so this repo provides some.

## No CUDA Toolkit required

The harness calls `cublasDgemm` through Python `ctypes` against `cublas64_13.dll` / `cudart64_13.dll`. Any CUDA 13.0u2+ redistributable set works; this run used the DLLs bundled with a TensorRT application, read-only. No compiler, no toolkit install.

## Layout

| path | what |
|---|---|
| `dgemm_bench.py` | one mode per process: timing + accuracy |
| `power_bench.py` | sustained load with `nvidia-smi` sampling |
| `run_all.py` | spawns every mode, writes `results/*.json` |
| `make_report.py` | generates `docs/index.html` and this README |
| `results/` | raw JSON, every repetition included |

## Reproduce

```
set FP64EMU_CUDA_DIR=C:\path\to\cuda13\dlls
python run_all.py
python make_report.py
```

## Accuracy methodology

Results are compared against the **exact** product, not another floating-point result. `math.fma` two-product recovers the rounding error of every `a*b`, and `math.fsum` sums products and errors exactly, giving the correctly-rounded dot product. Relative error is reported over randomly sampled output entries.

## What this does not show

- DGEMM only. Real applications gain in proportion to how much FP64 matrix multiply they actually do.
- One machine, one driver, one session.
- The gain depends on the exponent range of the inputs.
- **Nothing for deep-learning inference.** Inference runs in FP16/INT8/FP8 and uses no FP64. This technique spends INT8 throughput to reconstruct FP64 accuracy; it does not make low-precision inference faster.

## References

- [cuBLAS documentation - Fixed-Point Emulation](https://docs.nvidia.com/cuda/cublas/)
- [NVIDIA: Unlocking Tensor Core Performance with Floating Point Emulation in cuBLAS](https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/)
- [Ozaki Scheme II (arXiv:2504.08009)](https://arxiv.org/abs/2504.08009)
- [RIKEN-RCCS/GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8)

## License

MIT
