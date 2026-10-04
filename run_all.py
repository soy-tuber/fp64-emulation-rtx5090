#!/usr/bin/env python3
"""Spawn dgemm_bench.py once per emulation mode and collect the JSON.

Each mode needs its own process because cuBLAS reads the emulation
environment variables when the handle is created.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
BENCH = os.path.join(HERE, "dgemm_bench.py")
os.makedirs(RESULTS, exist_ok=True)

EMU_KEYS = ["CUBLAS_EMULATE_DOUBLE_PRECISION", "CUBLAS_EMULATION_STRATEGY",
            "CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT"]


def run(label, env_extra, args):
    env = dict(os.environ)
    for k in EMU_KEYS:
        env.pop(k, None)
    env.update(env_extra)
    out = os.path.join(RESULTS, label + ".json")
    cmd = [sys.executable, BENCH, "--label", label, "--out", out] + args
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stdout.write("  !! rc=%d\n%s\n" % (r.returncode, r.stderr[-1500:]))
    sys.stdout.write("  -> %s (%.0fs)\n" % (os.path.basename(out), time.time() - t0))
    sys.stdout.flush()


def main():
    sizes_a = "2048,4096,8192,16384"

    print("=== Phase A: size sweep, native vs emulated (ADP automatic) ===",
          flush=True)
    common_a = ["--sizes", sizes_a, "--dists", "easy,hard",
                "--reps", "5", "--warmup", "2", "--samples", "64"]
    run("A_native", {}, common_a)
    run("A_emu_adp", {"CUBLAS_EMULATE_DOUBLE_PRECISION": "1"}, common_a)

    print("=== Phase B: strategy check at n=8192 ===", flush=True)
    common_b = ["--sizes", "8192", "--dists", "easy,hard",
                "--reps", "5", "--warmup", "2", "--samples", "48"]
    for strat in ("performant", "eager"):
        run("B_emu_" + strat,
            {"CUBLAS_EMULATE_DOUBLE_PRECISION": "1",
             "CUBLAS_EMULATION_STRATEGY": strat}, common_b)

    print("=== Phase C: mantissa-bit sweep at n=8192 (accuracy/speed curve) ===",
          flush=True)
    common_c = ["--sizes", "8192", "--dists", "easy,hard",
                "--reps", "3", "--warmup", "2", "--samples", "32"]
    for bits in (8, 12, 16, 20, 24, 28, 32, 40, 48, 53):
        run("C_mant%02d" % bits,
            {"CUBLAS_EMULATE_DOUBLE_PRECISION": "1",
             "CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT": str(bits)},
            common_c)

    print("=== done ===", flush=True)
    files = sorted(f for f in os.listdir(RESULTS) if f.endswith(".json"))
    print("%d result files" % len(files), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
