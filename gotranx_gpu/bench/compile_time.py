"""Time each stage from .ode file to running machine code, numba backend.

    python bench/compile_time.py [model ...]          # default: all of odefiles/

Stages, for rhs, monitor_values and generalized_rush_larsen:
  load      gotranx: parse the .ode file, build the sympy model, guard singularities
  codegen   gotranx: schemes (linearisation) and printing -> Python source text
  import    Python: compile the generated source to bytecode and run the module
  cpu       numba, first call of each CPU function, split by numba's own timers into
              frontend  type inference + lowering to LLVM IR
              llvm      LLVM optimisation + x86 machine code
  cuda      numba-cuda, first launch of each kernel (NVVM -> PTX -> cubin via nvJitLink)

Each model runs in fresh processes, three ways:
  nocache     cache=False (as the tests and bench.py use it)
  cache_cold  cache=True, empty cache dir: compiles and writes the cache
  cache_warm  cache=True, a new process: machine code loaded from the cache (no codegen,
              the generated module already exists on disk)
Writes bench/compile_time.json.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ODEDIR = HERE.parent / "odefiles"
FUNCS = ["rhs", "monitor_values", "generalized_rush_larsen"]
N = 1024


def child(model: str, mode: str, workdir: str) -> dict:
    import importlib.util
    import numpy as np

    res: dict = {"model": model, "mode": mode}
    work = Path(workdir)
    modfile = work / f"{model}_numba.py"
    if mode != "cache_warm":
        import gotranx
        from gotranx.codegen.python import Format
        from gotranx_gpu import get_code

        t = time.perf_counter()
        ode = gotranx.load_ode(ODEDIR / f"{model}.ode")
        res["load_s"] = time.perf_counter() - t
        res["num_states"], res["num_parameters"] = ode.num_states, ode.num_parameters
        t = time.perf_counter()
        code = get_code(ode, backend="numba", scheme=["generalized_rush_larsen"],
                        format=Format.none, cache=(mode != "nocache"))
        res["codegen_s"] = time.perf_counter() - t
        res["source_lines"] = code.count("\n")
        modfile.write_text(code)

    t = time.perf_counter()
    spec = importlib.util.spec_from_file_location(f"{model}_numba", modfile)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    res["import_s"] = time.perf_counter() - t

    s = np.ascontiguousarray(np.repeat(m.init_state_values()[:, None], N, axis=1))
    p = m.init_parameter_values()
    args = {"rhs": (0.0, s, p), "monitor_values": (0.0, s, p),
            "generalized_rush_larsen": (s, 0.0, 0.01, p)}

    for f in FUNCS:
        t = time.perf_counter(); getattr(m, f)(*args[f]); first = time.perf_counter() - t
        t = time.perf_counter(); getattr(m, f)(*args[f]); again = time.perf_counter() - t
        disp = getattr(m, f"_{f}_cpu")
        timers = {}
        for cres in disp.overloads.values():
            timers = dict((cres.metadata or {}).get("timers", {}))
        r = {"first_call_s": first, "run_s": again}
        if "compiler_lock" in timers:  # absent when loaded from the cache
            r["compile_s"] = timers["compiler_lock"]
            r["llvm_s"] = timers.get("llvm_lock", 0.0)
            r["frontend_s"] = r["compile_s"] - r["llvm_s"]
        res[f"cpu_{f}"] = r

    try:
        from numba import cuda
        if cuda.is_available():
            sd, pd = cuda.to_device(s), cuda.to_device(p)
            dargs = {"rhs": (0.0, sd, pd), "monitor_values": (0.0, sd, pd),
                     "generalized_rush_larsen": (sd, 0.0, 0.01, pd)}
            for f in FUNCS:
                g = getattr(m, f + "_cuda")
                t = time.perf_counter(); g(*dargs[f]); cuda.synchronize(); first = time.perf_counter() - t
                t = time.perf_counter(); g(*dargs[f]); cuda.synchronize(); again = time.perf_counter() - t
                res[f"cuda_{f}"] = {"first_call_s": first, "run_s": again}
    except ImportError:
        pass
    return res


def main():
    models = sys.argv[1:] or sorted(p.stem for p in ODEDIR.glob("*.ode"))
    out = []
    for model in models:
        work = tempfile.mkdtemp(prefix=f"ct_{model}_")
        env = {**os.environ, "NUMBA_CACHE_DIR": os.path.join(work, "numba_cache"),
               "CUDA_PYTHON_DISABLE_MAJOR_VERSION_WARNING": "1"}
        for mode in ("nocache", "cache_cold", "cache_warm"):
            r = subprocess.run([sys.executable, __file__, "--child", model, mode, work],
                               env=env, capture_output=True, text=True)
            line = [x for x in r.stdout.splitlines() if x.startswith("{")]
            if r.returncode or not line:
                print(f"{model} {mode} FAILED\n{r.stderr[-3000:]}", flush=True)
                continue
            res = json.loads(line[-1])
            out.append(res)
            cpu = sum(res[f"cpu_{f}"]["first_call_s"] for f in FUNCS)
            cud = sum(res.get(f"cuda_{f}", {}).get("first_call_s", 0.0) for f in FUNCS)
            print(f"{model:34s} {mode:10s} load {res.get('load_s', 0):6.2f}  codegen {res.get('codegen_s', 0):6.2f}  "
                  f"import {res['import_s']:5.2f}  cpu first calls {cpu:7.2f}  cuda first calls {cud:7.2f} s",
                  flush=True)
        shutil.rmtree(work, ignore_errors=True)
    path = Path(os.environ.get("OUT", HERE / "compile_time.json"))
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        print(json.dumps(child(*sys.argv[2:5])))
    else:
        main()
