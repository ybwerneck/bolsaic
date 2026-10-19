"""PyTorch backend for gotranx: batched, device-agnostic (CPU/GPU), differentiable.

States are laid out as gotranx's vectorised numpy code expects, shape
``(num_states, N)`` (or ``(num_states,)`` for a single cell); parameters are
``(num_parameters,)`` shared by all cells or ``(num_parameters, N)`` per cell.
Every generated function works on ``torch.Tensor`` and keeps the device and
dtype of ``states``.
"""
from __future__ import annotations

import functools

import sympy
from sympy.codegen.ast import Assignment
from sympy.printing.pycode import PythonCodePrinter

from gotranx.codegen.base import _print_Piecewise
from gotranx.codegen.python import Format, GotranPythonCodePrinter, PythonCodeGenerator

from . import torch_template

# math name -> torch name, for every function sympy's Python printer knows
_FUNCS = {k: v.replace("math.", "") for k, v in PythonCodePrinter._kf.items()}
_TORCH_NAME = {"Abs": None, "Min": None, "Max": None}  # handled separately


class TorchPrinter(GotranPythonCodePrinter):
    """Elementwise torch code; constant sub-expressions go through ``math``."""

    _kc = {k: v for k, v in PythonCodePrinter._kc.items()}  # math.pi, math.e

    def _func(self, expr, name):
        args = ", ".join(self._print(a) for a in expr.args)
        if not expr.free_symbols:
            return f"math.{name}({args})"
        return f"torch.{name}({args})"

    def _print_Abs(self, expr):
        return f"abs({self._print(expr.args[0])})"

    def _hprint_Pow(self, expr, rational=False, sqrt="torch.sqrt"):
        if not expr.free_symbols:
            sqrt = "math.sqrt"
        return PythonCodePrinter._hprint_Pow(self, expr, rational=rational, sqrt=sqrt)

    def _print_Min(self, expr):
        args = [self._print(a) for a in expr.args]
        return functools.reduce(lambda x, y: f"torch.where({x} < {y}, {x}, {y})", args)

    def _print_Max(self, expr):
        args = [self._print(a) for a in expr.args]
        return functools.reduce(lambda x, y: f"torch.where({x} > {y}, {x}, {y})", args)

    def _print_And(self, expr):
        args = [self._print(a) for a in expr.args]
        return functools.reduce(lambda x, y: f"torch.logical_and({x}, {y})", args)

    def _print_Or(self, expr):
        args = [self._print(a) for a in expr.args]
        return functools.reduce(lambda x, y: f"torch.logical_or({x}, {y})", args)

    def _print_Not(self, expr):
        return f"torch.logical_not({self._print(expr.args[0])})"

    def _print_sign(self, expr):
        return f"torch.sign({self._print(expr.args[0])})"

    def _print_DiracDelta(self, expr):
        return f"torch.zeros_like({self._print(expr.args[0])})"

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
            out = f"torch.where({c}, {e}, {out})"
        return out


for _k, _name in _FUNCS.items():
    if _k in _TORCH_NAME:
        continue
    setattr(TorchPrinter, f"_print_{_k}",
            functools.partialmethod(lambda self, expr, name: self._func(expr, name), name=_name))


class TorchCodeGenerator(PythonCodeGenerator):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("format", Format.none)
        super().__init__(*args, **kwargs)
        self._printer = TorchPrinter()

    def imports(self) -> str:
        return "import math\nimport torch"

    @property
    def template(self):
        return torch_template

    def _rhs_arguments(self, order="tsp"):
        f = super()._rhs_arguments(order)
        return f._replace(values_type="torch.empty_like(states)")

    def _scheme_arguments(self, order="stdp"):
        f = super()._scheme_arguments(order)
        return f._replace(values_type="torch.empty_like(states)")

    def _shape_info(self, shape) -> str:
        return f"shape = ({shape},) + tuple(states.shape[1:])"
