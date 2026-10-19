"""numba backend for gotranx: one scalar per-cell body, compiled twice.

* ``<name>(...)``       CPU, ``numba.njit(parallel=True)``, ``prange`` over cells
* ``<name>_cuda(...)``  GPU, ``numba.cuda.jit`` kernel, one thread per cell

States are ``(num_states, N)`` (cell index last, coalesced on the GPU);
parameters are ``(num_parameters,)`` shared by all cells, or
``(num_parameters, N)`` per cell when generated with ``per_cell_parameters``.
The body uses ``math`` only, so the same text is valid njit and CUDA code.
"""
from __future__ import annotations

import functools

import sympy
from sympy.codegen.ast import Assignment
from sympy.printing.pycode import PythonCodePrinter

from gotranx.codegen.base import _print_Piecewise
from gotranx.codegen.python import Format, PythonCodeGenerator

from . import numba_template

CELL = "_i"  # loop / thread index in the generated code


class NumbaScalarPrinter(PythonCodePrinter):
    """Scalar code for one cell: math.*, ternaries, [k, _i] indexing."""

    def __init__(self, per_cell_parameters: bool = False, **kwargs):
        super().__init__({"fully_qualified_modules": True, **kwargs})
        self.per_cell = {"states", "values", "missing_variables"}
        if per_cell_parameters:
            self.per_cell.add("parameters")

    def _print_Indexed(self, expr):
        base = self._print(expr.base.label)
        idx = ", ".join(str(int(i)) for i in expr.indices)
        if base in self.per_cell:
            return f"{base}[{idx}, {CELL}]"
        return f"{base}[{idx}]"

    def _print_Float(self, flt):
        return repr(float(flt))

    def _print_Integer(self, expr):
        return f"{int(expr)}.0"  # keep float arithmetic in the kernels

    def _print_Rational(self, expr):
        return f"({expr.p}.0/{expr.q}.0)"

    def _print_Min(self, expr):
        return functools.reduce(lambda x, y: f"min({x}, {y})", [self._print(a) for a in expr.args])

    def _print_Max(self, expr):
        return functools.reduce(lambda x, y: f"max({x}, {y})", [self._print(a) for a in expr.args])

    def _print_sign(self, expr):
        e = self._print(expr.args[0])
        return f"(0.0 if ({e}) == 0.0 else math.copysign(1.0, {e}))"

    def _print_DiracDelta(self, expr):
        return "0.0"

    def _print_Equality(self, expr):
        lhs, rhs = expr.args
        return f"({self._print(lhs)} == {self._print(rhs)})"

    def _print_Piecewise(self, expr):
        if isinstance(expr.args[0][0], Assignment):
            lhs = self._print(expr.args[0][0].lhs)
            inner = sympy.Piecewise(*[(a.rhs, c) for a, c in expr.args])
            return f"{lhs} = {self._print_Piecewise(inner)}"
        conds, exprs = _print_Piecewise(self, expr)
        if conds[-1] != "True":
            raise ValueError("Last condition in Piecewise must be True")
        out = exprs[-1]
        for c, e in zip(reversed(conds[:-1]), reversed(exprs[:-1])):
            out = f"({e} if {c} else {out})"
        return out


class NumbaCodeGenerator(PythonCodeGenerator):
    def __init__(self, *args, per_cell_parameters: bool = False, **kwargs) -> None:
        kwargs.setdefault("format", Format.none)
        super().__init__(*args, **kwargs)
        self._printer = NumbaScalarPrinter(per_cell_parameters=per_cell_parameters)
        self.per_cell_parameters = per_cell_parameters

    def imports(self) -> str:
        return numba_template.imports()

    @property
    def template(self):
        return numba_template

    def _shape_info(self, shape) -> str:
        return ""
