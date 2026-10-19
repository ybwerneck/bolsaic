"""Shared helpers: gotranx's numpy code is the reference for the numba backend.

CUDA tests run only where ``numba.cuda`` sees a GPU (set NO_CUDA=1 to skip them).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from gotranx.cli import gotran2py
from gotranx.codegen.python import Format

from gotranx_gpu import get_code

ODEDIR = Path(__file__).resolve().parent.parent / "odefiles"
ODEFILES = sorted(ODEDIR.glob("*.ode"))


def _exec(code: str) -> dict:
    ns: dict = {}
    exec(compile(code, "<generated>", "exec"), ns)
    return ns


def numpy_ns(ode, **kwargs) -> dict:
    """gotranx's own vectorised numpy code (the reference)."""
    return _exec(gotran2py.get_code(ode, format=Format.none, **kwargs))


def numba_ns(ode, **kwargs) -> dict:
    return _exec(get_code(ode, backend="numba", format=Format.none, **kwargs))


def perturbed(states: np.ndarray, n_cells: int, seed: int = 0) -> np.ndarray:
    """(num_states, n_cells): each cell's states differ by about 2 %."""
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(np.outer(states, 1 + 0.02 * rng.standard_normal(n_cells)))


def assert_close(actual, desired, rtol=1e-12, atol=1e-12, msg=""):
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, err_msg=msg)


def _have_cuda() -> bool:
    if os.environ.get("NO_CUDA"):
        return False
    try:
        from numba import cuda
        return cuda.is_available()
    except Exception:
        return False


HAVE_CUDA = _have_cuda()
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not HAVE_CUDA, reason="no CUDA GPU"))]


def call(ns, name, device, *args, out=None, threads=128):
    """Run generated function ``name`` on ``device``; numpy in, numpy out."""
    if device == "cpu":
        return ns[name](*args, out=out)
    from numba import cuda
    dargs = [cuda.to_device(np.ascontiguousarray(a)) if isinstance(a, np.ndarray) else a for a in args]
    return ns[name + "_cuda"](*dargs, out=out, threads=threads).copy_to_host()
