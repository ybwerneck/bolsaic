"""Templates for the numba backend: the same scalar body under a CPU prange loop
and inside a CUDA kernel."""
from __future__ import annotations

from textwrap import dedent, indent

from gotranx.templates.python import (init_parameter_values as _init_p, init_state_values as _init_s,
                                      missing_index, monitor_index, parameter_index, state_index)


def imports() -> str:
    return dedent("""\
        import math
        import numpy
        import numba
        try:
            from numba import cuda
        except Exception:  # numba-cuda not installed
            cuda = None
        """)


def init_state_values(*args, **kwargs):
    return _init_s(*args, **kwargs)


def init_parameter_values(*args, **kwargs):
    return _init_p(*args, **kwargs)


def method(name, args, states, parameters, values, return_name, num_return_values,
           missing_variables="", **kwargs):
    body = "\n".join([states, parameters, missing_variables, values])
    body8 = indent(body, " " * 8)
    out = return_name
    return dedent(f'''
def _{name}_cpu_body({args}, {out}):
    for _i in numba.prange(states.shape[1]):
{body8}


_{name}_cpu = numba.njit(parallel=True, error_model="numpy")(_{name}_cpu_body)


def {name}({args}, out=None):
    """CPU, all cells in parallel. states: (num_states, N) or (num_states,)."""
    single = states.ndim == 1
    if single:
        states = states[:, None]
    if out is None:
        out = numpy.empty(({num_return_values}, states.shape[1]), dtype=states.dtype)
    _{name}_cpu({args}, out)
    return out[:, 0] if single else out


def _{name}_cuda_body({args}, {out}):
    _i = cuda.grid(1)
    if _i < states.shape[1]:
{body8}


_{name}_cuda_kernel = cuda.jit(_{name}_cuda_body) if cuda is not None else None


def {name}_cuda({args}, out=None, threads=128):
    """GPU, one thread per cell. states: (num_states, N) device array (numba
    device array or a CUDA torch tensor); returns a device array."""
    n = states.shape[1]
    if out is None:
        out = cuda.device_array(({num_return_values}, n), dtype=numpy.float64)
    _{name}_cuda_kernel[(n + threads - 1) // threads, threads]({args}, out)
    return out
''')


__all__ = ["imports", "init_state_values", "init_parameter_values", "method", "parameter_index",
           "state_index", "monitor_index", "missing_index"]
