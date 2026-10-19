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

## Layout

- `src/gotranx_gpu/torch_backend.py`, `torch_template.py`: printer (`numpy.*` → `torch.*`, constant sub-expressions → `math.*`, `Piecewise` → `torch.where`) and templates.
- `src/gotranx_gpu/numba_backend.py`, `numba_template.py`: scalar printer (`math.*`, ternaries, `states[k, _i]`) and the CPU/CUDA wrappers.
- `src/gotranx_gpu/generate.py`: `get_code` and the CLI (`python -m gotranx_gpu`).
- `tests/test_equivalence.py`: each backend against gotranx's numpy code on 257 perturbed cells (rhs, monitors, explicit Euler and generalized Rush–Larsen steps; rtol 1e-10), five models.
- `bench/bench.py`: time per generalized Rush–Larsen step vs number of cells, per backend.
- `odefiles/`: models from the gotranx repository (MIT, H. Finsberg).

## Environment (MMC cluster)

`~/gotranx_gpu/venv` on the shared home: gotranx, numba, numba-cuda; torch is reused
from `~/inverse_ecg/venv` through a `.pth` file (not installed twice).
Run on a GPU node, e.g. `ssh 10.22.10.119`, then `cd ~/gotranx_gpu && venv/bin/python -m pytest tests`.
