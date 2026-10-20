# gotranx_gpu

PyTorch and numba (njit + CUDA) backends for [gotranx](https://finsberg.github.io/gotranx/):
generate code that advances many cells of a cardiac cell model at once, on CPU or GPU.

gotranx already emits numpy (vectorised), JAX, C and CUDA C. This adds:

| backend | functions generated | runs on | notes |
|---|---|---|---|
| `torch` | `rhs`, `monitor_values`, each scheme | CPU or GPU (device of `states`) | batched elementwise; differentiable (autograd through `parameters` and `states`); works with `torch.compile` |
| `numba` | `<f>(...)` and `<f>_cuda(...)` for each of the above | CPU (`njit(parallel=True)`, `prange` over cells) and GPU (`numba.cuda`, one thread per cell) | one scalar per-cell body, written once, compiled for both; the CUDA functions accept numba device arrays or CUDA torch tensors |

Layout: `states` is `(num_states, N)`, cell index last (coalesced on the GPU);
`parameters` is `(num_parameters,)` shared by all cells (torch also accepts
`(num_parameters, N)`; numba with `--per-cell-parameters`).

## Use

```bash
python -m gotranx_gpu odefiles/ToRORd_dyn_chloride.ode --backend torch --scheme generalized_rush_larsen -o torord_torch.py
python -m gotranx_gpu odefiles/ToRORd_dyn_chloride.ode --backend numba --scheme generalized_rush_larsen -o torord_numba.py
```

```python
import torch, torord_torch as m
p = m.init_parameter_values(device="cuda")
s = m.init_state_values(device="cuda")[:, None].repeat(1, 1_000_000)   # 1e6 cells
for n in range(1000):
    s = m.generalized_rush_larsen(s, n * 0.01, 0.01, p)

import numpy as np, torord_numba as m
from numba import cuda
s = cuda.to_device(np.repeat(m.init_state_values()[:, None], 1_000_000, axis=1))
p = cuda.to_device(m.init_parameter_values())
out = cuda.device_array_like(s)
for n in range(1000):
    m.generalized_rush_larsen_cuda(s, n * 0.01, 0.01, p, out=out)
    s, out = out, s
```

From Python: `gotranx_gpu.get_code(ode, backend="torch"|"numba", scheme=[...])`.

## Against the native gotranx backends

One generalized Rush–Larsen step over 10⁶ cells, ms (node .119: 12 CPU cores, RTX 4070;
`bench/native.py`, results in `bench/native_*.json`). Native C is gotranx's `ode2c`
per-cell function in a C loop (gcc -O3 -march=native, OpenMP for 12 threads); native
CUDA is `ode2cuda` (nvcc -O3, host compiler clang). All agree with numpy to ≤ 1.2e-13.

| model (states) | numpy | C 1 thread | numba 1 thread | C 12 threads | numba 12 threads | CUDA C | numba CUDA |
|---|---|---|---|---|---|---|---|
| Beeler–Reuter (8) | 505 | 184 | 160 | 30.6 | 27.3 | 7.1 | 6.0 |
| ten Tusscher (19) | 1216 | 512 | 438 | 84.1 | 71.1 | 31.9 | 28.7 |
| ToRORd (52) | 5242 | 1723 | 1216 | 278 | 205 | 82.1 | 79.0 |

Compile, ToRORd: gcc 1.0 s, nvcc 7.0 s, numba CPU 32 s, numba CUDA 15 s (under 1 s from the cache).

## Compile time

From `.ode` file to running machine code there are three stages; the first two are
paid on every fresh process unless the numba cache is used (`--cache`, or
`get_code(..., cache=True)` and import the generated module from a file).

| model (states) | load + codegen (gotranx) | numba front end | LLVM | CPU total | CUDA first launches | with cache (new process) |
|---|---|---|---|---|---|---|
| ToRORd (52) | 6.6 + 11.0 s | 45.7 s | 5.9 s | 51.6 s | 30.4 s | 0.4 + 0.3 + 0.1 s |
| ORd-Land (48) | 4.0 + 7.9 s | 24.1 s | 4.1 s | 28.2 s | 17.5 s | 0.3 + 0.3 + 0.1 s |
| ten Tusscher (19) | 0.7 + 1.8 s | 5.0 s | 1.2 s | 6.2 s | 5.0 s | 0.4 + 0.3 + 0.1 s |
| Beeler–Reuter (8) | 1.3 + 2.0 s | 2.0 s | 0.7 s | 2.8 s | 1.8 s | 0.3 + 0.2 + 0.1 s |

rhs + monitor_values + generalized_rush_larsen; node .119 (RTX 4070); `bench/compile_time.py`,
results in `bench/compile_time.json`. Cache column: import + CPU first calls + CUDA first calls.

## Layout

- `src/gotranx_gpu/torch_backend.py`, `torch_template.py`: printer (`numpy.*` → `torch.*`, constant sub-expressions → `math.*`, `Piecewise` → `torch.where`) and templates.
- `src/gotranx_gpu/numba_backend.py`, `numba_template.py`: scalar printer (`math.*`, ternaries, `states[k, _i]`) and the CPU/CUDA wrappers.
- `src/gotranx_gpu/generate.py`: `get_code` and the CLI (`python -m gotranx_gpu`).
- `tests/test_models.py`: numba CPU and CUDA against gotranx's numpy code, every model in `odefiles/` (rhs, monitors, three schemes, single cell, in-place step, trailing CUDA threads).
- `tests/test_edge_cases.py`: gotranx's own edge cases (Min/Max kinks, conditionals, removable poles, missing variables, missing_values, stiff states, remove_unused) through numba CPU and CUDA. 114 tests, all passing on node .119.
- `bench/bench.py`: time per generalized Rush–Larsen step vs number of cells, per backend (torch only with `BENCH_TORCH=1`).
- `bench/native.py`: numba against gotranx's native numpy, C (serial / OpenMP) and CUDA C.
- `bench/compile_time.py`: time of each stage from `.ode` to machine code, with and without the numba cache.
- `odefiles/`: models from the gotranx repository (MIT, H. Finsberg).

## Environment (MMC cluster)

`~/gotranx_gpu/venv` on the shared home: gotranx, numba, numba-cuda; torch is reused
from `~/inverse_ecg/venv` through a `.pth` file (not installed twice).
Run on a GPU node, e.g. `ssh 10.22.10.119`, then `cd ~/gotranx_gpu && venv/bin/python -m pytest tests`.
