"""gotranx_gpu (numba) against gotranx's native backends, one GRL step over N cells.

    python bench/native.py [model ...] [--cells 1000,10000,100000,1000000]

Default models: beeler_reuter_1977 (8 states), tentusscher_panfilov_2006_M_cell (19),
ToRORd_dyn_chloride (52). Needs gotranx with ode2cuda (GitHub main), gcc and nvcc.

Contestants, each with the layout its code expects:
  numpy            gotranx ode2py, vectorised over cells, (num_states, N)
  c_1t / c_omp     gotranx ode2c, gcc -O3 -march=native; the per-cell function in a C loop
                   over cells (cell-major (N, num_states)), serial / OpenMP on all cores
  numba_1t / numba gotranx_gpu numba CPU, prange, 1 thread / all cores, (num_states, N)
  cuda_c           gotranx ode2cuda, nvcc -O3 -arch=native, one thread per cell, (num_states, N)
  numba_cuda       gotranx_gpu numba.cuda, one thread per cell, (num_states, N)
Time per step: median of up to REPS steps (at least 3, stopping after ~3 s), GPU synchronised.
Compile: gcc / nvcc wall time against numba's first call (CPU, CUDA) of the same scheme.
Error: max relative deviation from numpy's step. Writes bench/native_<model>.json.
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import gotranx
from gotranx.cli import gotran2c, gotran2cuda, gotran2py
from gotranx.codegen import CFormat
from gotranx.codegen.python import Format
from gotranx.schemes import Scheme

from gotranx_gpu import get_code

HERE = Path(__file__).resolve().parent
ODEDIR = HERE.parent / "odefiles"
SCHEME, DT, T, REPS, THREADS = "generalized_rush_larsen", 0.01, 0.0, 10, 128
NTHREADS = os.cpu_count()

C_DRIVER = r"""
#include "model.h"
void step_serial(const double *s, double t, double dt, const double *p, double *out, long n) {
    for (long c = 0; c < n; ++c)
        generalized_rush_larsen(s + c * NUM_STATES, t, dt, p, out + c * NUM_STATES);
}
void step_omp(const double *s, double t, double dt, const double *p, double *out, long n) {
    #pragma omp parallel for schedule(static)
    for (long c = 0; c < n; ++c)
        generalized_rush_larsen(s + c * NUM_STATES, t, dt, p, out + c * NUM_STATES);
}
"""
CU_DRIVER = r"""
#include "model.cuh"
extern "C" int step(const double *s, double t, double dt, const double *p, double *out,
                    long long n, int threads) {
    long long blocks = (n + threads - 1) / threads;
    model::generalized_rush_larsen<<<blocks, threads>>>(s, t, dt, p, out, n);
    return (int)cudaDeviceSynchronize();
}
"""


def timeit(step, sync=lambda: None):
    step(); sync()  # warm-up
    ts = []
    while len(ts) < REPS and (len(ts) < 3 or sum(ts) < 3.0):
        t = time.perf_counter(); step(); sync(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def relerr(a, b):
    return float(np.max(np.abs(a - b) / (np.abs(b) + 1e-12)))


def build(model: str, work: Path) -> dict:
    ode = gotranx.load_ode(ODEDIR / f"{model}.ode")
    sch = [Scheme(SCHEME)]
    info = {"num_states": ode.num_states}
    ref: dict = {}
    exec(gotran2py.get_code(ode, scheme=sch, format=Format.none), ref)
    nb: dict = {}
    exec(get_code(ode, backend="numba", scheme=[SCHEME], format=Format.none), nb)

    (work / "model.h").write_text(gotran2c.get_code(ode, scheme=sch, format=CFormat.none))
    (work / "drv.c").write_text(C_DRIVER)
    t = time.perf_counter()
    subprocess.run(["gcc", "-O3", "-march=native", "-fopenmp", "-fPIC", "-shared", "-I", str(work),
                    "-o", str(work / "libc.so"), str(work / "drv.c"), "-lm"], check=True)
    info["compile_gcc_s"] = time.perf_counter() - t
    clib = ctypes.CDLL(str(work / "libc.so"))
    for f in ("step_serial", "step_omp"):
        getattr(clib, f).argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double,
                                     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]

    culib = None
    try:
        (work / "model.cuh").write_text(gotran2cuda.get_code(ode, scheme=sch, format=CFormat.none,
                                                             namespace="model"))
        (work / "drv.cu").write_text(CU_DRIVER)
        t = time.perf_counter()
        subprocess.run(["nvcc", "-O3", "-arch=native", "-shared", "-Xcompiler", "-fPIC", "-I", str(work),
                        "-o", str(work / "libcu.so"), str(work / "drv.cu")], check=True)
        info["compile_nvcc_s"] = time.perf_counter() - t
        culib = ctypes.CDLL(str(work / "libcu.so"))
        culib.step.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_void_p,
                               ctypes.c_void_p, ctypes.c_longlong, ctypes.c_int]
        culib.step.restype = ctypes.c_int
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        print("native CUDA unavailable:", e, flush=True)
    return {"ode": ode, "ref": ref, "nb": nb, "clib": clib, "culib": culib, "info": info}


def run(model: str, cells: list[int]) -> dict:
    import numba
    from numba import cuda
    work = Path(tempfile.mkdtemp(prefix=f"nat_{model}_"))
    b = build(model, work)
    ref, nb, clib, culib, info = b["ref"], b["nb"], b["clib"], b["culib"], b["info"]
    s0, p = ref["init_state_values"](), ref["init_parameter_values"]()
    rows = []
    for N in cells:
        rng = np.random.default_rng(0)
        s = np.ascontiguousarray(np.outer(s0, 1 + 0.01 * rng.standard_normal(N)))  # (ns, N)
        with np.errstate(all="ignore"):
            expect = ref[SCHEME](s, T, DT, p)
        r: dict = {"N": N}

        r["numpy"] = timeit(lambda: ref[SCHEME](s, T, DT, p)) if N <= 10 ** 6 else None

        sc, oc = np.ascontiguousarray(s.T), np.empty((N, s.shape[0]))  # (N, ns) for C
        for key, f in (("c_1t", clib.step_serial), ("c_omp", clib.step_omp)):
            call = lambda: f(sc.ctypes.data, T, DT, p.ctypes.data, oc.ctypes.data, N)
            r[key] = timeit(call)
            r[key + "_err"] = relerr(oc.T, expect)

        out = np.empty_like(s)
        f = nb[SCHEME]
        if N == cells[0]:
            t = time.perf_counter(); f(s, T, DT, p, out=out); info["compile_numba_cpu_s"] = time.perf_counter() - t
        for key, nt in (("numba_1t", 1), ("numba", NTHREADS)):
            numba.set_num_threads(nt)
            r[key] = timeit(lambda: f(s, T, DT, p, out=out))
            r[key + "_err"] = relerr(out, expect)

        if cuda.is_available():
            sd, pd = cuda.to_device(s), cuda.to_device(p)
            od = cuda.device_array_like(sd)
            g = nb[SCHEME + "_cuda"]
            if N == cells[0]:
                t = time.perf_counter(); g(sd, T, DT, pd, out=od); cuda.synchronize()
                info["compile_numba_cuda_s"] = time.perf_counter() - t
            r["numba_cuda"] = timeit(lambda: g(sd, T, DT, pd, out=od, threads=THREADS), cuda.synchronize)
            r["numba_cuda_err"] = relerr(od.copy_to_host(), expect)
            if culib is not None:
                ptr = lambda a: a.__cuda_array_interface__["data"][0]
                od2 = cuda.device_array_like(sd)
                rc = []
                r["cuda_c"] = timeit(lambda: rc.append(culib.step(ptr(sd), T, DT, ptr(pd), ptr(od2), N, THREADS)))
                assert not any(rc), f"CUDA error {set(rc)}"
                r["cuda_c_err"] = relerr(od2.copy_to_host(), expect)
            del sd, pd, od
        line = "  ".join(f"{k} {v * 1e3:9.3f}" for k, v in r.items()
                         if k != "N" and not k.endswith("_err") and v is not None)
        print(f"{model} N {N:>8d} (ms/step): {line}", flush=True)
        rows.append(r)
    out = {"model": model, "scheme": SCHEME, "cpu_threads": NTHREADS, **info, "rows": rows}
    (HERE / f"native_{model}.json").write_text(json.dumps(out, indent=1))
    print(model, {k: round(v, 2) for k, v in info.items() if k.startswith("compile")}, flush=True)
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    cells = [1000, 10000, 100000, 1000000]
    if "--cells" in args:
        i = args.index("--cells"); cells = [int(x) for x in args[i + 1].split(",")]; del args[i:i + 2]
    models = args or ["beeler_reuter_1977", "tentusscher_panfilov_2006_M_cell", "ToRORd_dyn_chloride"]
    for m in models:
        run(m, cells)
