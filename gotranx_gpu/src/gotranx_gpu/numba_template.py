"""Templates for the numba backend: the same scalar body under a CPU prange loop
and inside a CUDA kernel.

``NumbaTemplate(per_cell_parameters, cache)`` is what ``NumbaCodeGenerator.template``
returns; gotranx's ``CodeGenerator`` calls its ``method``, ``init_*`` and ``*_index``.
"""
from __future__ import annotations

from textwrap import dedent, indent

from gotranx.templates import python as _py


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


class NumbaTemplate:
    state_index = staticmethod(_py.state_index)
    parameter_index = staticmethod(_py.parameter_index)
    monitor_index = staticmethod(_py.monitor_index)
    missing_index = staticmethod(_py.missing_index)
    init_state_values = staticmethod(_py.init_state_values)
    init_parameter_values = staticmethod(_py.init_parameter_values)

    def __init__(self, per_cell_parameters: bool = False, cache: bool = False):
        self.per_cell_parameters = per_cell_parameters
        self.cache = cache

    def method(self, name, args, states, parameters, values, return_name, num_return_values,
               missing_variables="", **kwargs):
        body8 = indent("\n".join([states, parameters, missing_variables, values]), " " * 8)
        out = return_name
        names = [a.strip() for a in args.split(",")]
        # a single cell, (num_states,), runs as one column; so must its per-cell arguments
        per_cell = ["states"] + [a for a in ("missing_variables",) if a in names]
        if self.per_cell_parameters and "parameters" in names:
            per_cell.append("parameters")
        to_col = "\n".join(f"        {a} = {a}[:, None]" for a in per_cell)
        cache = "True" if self.cache else "False"
        return dedent(f'''
def _{name}_cpu_body({args}, {out}):
    for _i in numba.prange(states.shape[1]):
{body8}


_{name}_cpu = numba.njit(parallel=True, error_model="numpy", cache={cache})(_{name}_cpu_body)


def {name}({args}, out=None):
    """CPU, all cells in parallel. states: (num_states, N), or (num_states,) for one cell."""
    single = states.ndim == 1
    if single:
{to_col}
    if out is None:
        out = numpy.empty(({num_return_values}, states.shape[1]), dtype=numpy.float64)
    _{name}_cpu({args}, out)
    return out[:, 0] if single else out


def _{name}_cuda_body({args}, {out}):
    _i = cuda.grid(1)
    if _i < states.shape[1]:
{body8}


_{name}_cuda_kernel = cuda.jit(_{name}_cuda_body, cache={cache}) if cuda is not None else None


def {name}_cuda({args}, out=None, threads=128):
    """GPU, one thread per cell. states: (num_states, N) device array (numba device
    array or CUDA torch tensor); returns a device array. out may alias states."""
    n = states.shape[1]
    if out is None:
        out = cuda.device_array(({num_return_values}, n), dtype=numpy.float64)
    _{name}_cuda_kernel[(n + threads - 1) // threads, threads]({args}, out)
    return out
''')
