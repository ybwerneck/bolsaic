"""Time one generalized Rush-Larsen step over N cells, per backend.

    python bench/bench.py [model] [N1,N2,...]

Backends: numpy (gotranx's own vectorised code), numba CPU (prange), numba CUDA;
torch CUDA and torch.compile only with BENCH_TORCH=1. Time per step = median over REPS
steps after WARM warm-up steps (GPU timings synchronised). Each backend also
reports its max relative deviation from numpy after the timed steps.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import gotranx
from gotranx.cli.gotran2py import get_code as numpy_code
from gotranx.codegen.python import Format

from gotranx_gpu import get_code

MODEL = sys.argv[1] if len(sys.argv) > 1 else "ToRORd_dyn_chloride"
NS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1000,10000,100000,1000000").split(",")]
SCHEME, DT, WARM, REPS = "generalized_rush_larsen", 0.01, 3, 20
SKIP = set(os.environ.get("SKIP", "").split(",")) - {""}
ode = gotranx.load_ode(Path(__file__).resolve().parent.parent / "odefiles" / f"{MODEL}.ode")


def _exec(code):
    ns = {}
    exec(compile(code, "<generated>", "exec"), ns)
    return ns


t0 = time.perf_counter()
ref = _exec(numpy_code(ode, scheme=[gotranx.schemes.Scheme(SCHEME)], format=Format.none))
tch = _exec(get_code(ode, backend="torch", scheme=[SCHEME], format=Format.none)) if os.environ.get("BENCH_TORCH") else {}
nb = _exec(get_code(ode, backend="numba", scheme=[SCHEME], format=Format.none))
print(f"{MODEL}: {ode.num_states} states, {ode.num_parameters} parameters; codegen {time.perf_counter() - t0:.1f} s",
      flush=True)
s0, p = ref["init_state_values"](), ref["init_parameter_values"]()


def timeit(step, sync=lambda: None):
    for _ in range(WARM):
        step()
    sync()
    ts = []
    for _ in range(REPS):
        t = time.perf_counter()
        step()
        sync()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


rows = []
for N in NS:
    s = np.ascontiguousarray(np.repeat(s0[:, None], N, axis=1))
    res = {}
    state = {"numpy": s.copy()}

    def run_numpy():
        state["numpy"] = ref[SCHEME](state["numpy"], 0.0, DT, p)
    if "numpy" not in SKIP:
        res["numpy"] = timeit(run_numpy)
    ref_s = ref[SCHEME](s, 0.0, DT, p)

    if "numba_cpu" not in SKIP:
        f = nb[SCHEME]
        x, y = s.copy(), np.empty_like(s)
        t = time.perf_counter(); f(x, 0.0, DT, p, out=y); jit = time.perf_counter() - t

        def run_nb():
            f(x, 0.0, DT, p, out=y)
        res["numba_cpu"] = timeit(run_nb)
        res["numba_cpu_compile_s"] = jit
        res["numba_cpu_err"] = float(np.max(np.abs(y - ref_s) / (np.abs(ref_s) + 1e-12)))

    try:
        from numba import cuda
        have_cuda = cuda.is_available()
    except Exception:
        have_cuda = False
    if have_cuda and "numba_cuda" not in SKIP:
        g = nb[SCHEME + "_cuda"]
        xd, pd = cuda.to_device(s), cuda.to_device(p)
        yd = cuda.device_array_like(xd)
        t = time.perf_counter(); g(xd, 0.0, DT, pd, out=yd); cuda.synchronize(); jit = time.perf_counter() - t

        def run_nc():
            g(xd, 0.0, DT, pd, out=yd)
        res["numba_cuda"] = timeit(run_nc, cuda.synchronize)
        res["numba_cuda_compile_s"] = jit
        res["numba_cuda_err"] = float(np.max(np.abs(yd.copy_to_host() - ref_s) / (np.abs(ref_s) + 1e-12)))

    have_torch = False
    if os.environ.get("BENCH_TORCH"):  # torch backend on hold; opt in with BENCH_TORCH=1
        try:
            import torch
            have_torch = torch.cuda.is_available()
        except Exception:
            pass
    if have_torch:
        st, pt = torch.as_tensor(s, device="cuda"), torch.as_tensor(p, device="cuda")
        variants = [("torch_cuda", tch[SCHEME])]
        if "torch_compile" not in SKIP:
            variants.append(("torch_compile", torch.compile(tch[SCHEME], dynamic=False)))
        for key, fn in variants:
            if key in SKIP:
                continue
            t = time.perf_counter(); out = fn(st, 0.0, DT, pt); torch.cuda.synchronize(); jit = time.perf_counter() - t
            res[key] = timeit(lambda: fn(st, 0.0, DT, pt), torch.cuda.synchronize)
            res[key + "_first_call_s"] = jit
            res[key + "_err"] = float(np.max(np.abs(out.cpu().numpy() - ref_s) / (np.abs(ref_s) + 1e-12)))
        del st, pt
        torch.cuda.empty_cache()

    line = "  ".join(f"{k} {v * 1e3:9.3f} ms" for k, v in res.items()
                     if not k.endswith(("_s", "_err")))
    print(f"N {N:>8d}: {line}", flush=True)
    rows.append({"N": N, **res})

out = Path(os.environ.get("OUT", Path(__file__).resolve().parent / f"bench_{MODEL}.json"))
out.write_text(json.dumps({"model": MODEL, "scheme": SCHEME, "rows": rows}, indent=1))
print("wrote", out)
