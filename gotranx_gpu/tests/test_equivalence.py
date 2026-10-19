"""Generated torch / numba code against gotranx's own numpy code, on N cells.

For each model: rhs, monitor_values and one step of each scheme, from the
initial state perturbed per cell, must match the numpy reference to rtol 1e-10.
GPU paths (torch on cuda, numba.cuda) run only when a GPU is visible.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import gotranx
from gotranx.cli.gotran2py import get_code as numpy_code
from gotranx.codegen.python import Format

from gotranx_gpu import get_code

ODEDIR = Path(__file__).resolve().parent.parent / "odefiles"
MODELS = ["fitzhughnagumo", "beeler_reuter_1977", "tentusscher_panfilov_2006_M_cell",
          "ORdmm_Land", "ToRORd_dyn_chloride"]
SCHEMES = ["explicit_euler", "generalized_rush_larsen"]
N, DT, T = 257, 0.01, 1.3
# numpy and torch round exp/pow differently in the last bits; on derivatives that are
# differences of large fluxes (ToRORd) this shows up as ~1e-11 absolute
RTOL, ATOL = 1e-10, 1e-10


def _exec(code):
    ns = {}
    exec(compile(code, "<generated>", "exec"), ns)
    return ns


@pytest.fixture(scope="module", params=MODELS)
def model(request):
    ode = gotranx.load_ode(ODEDIR / f"{request.param}.ode")
    ref = _exec(numpy_code(ode, scheme=[gotranx.schemes.Scheme(s) for s in SCHEMES], format=Format.none))
    s0 = ref["init_state_values"]()
    p = ref["init_parameter_values"]()
    rng = np.random.default_rng(0)
    s = s0[:, None] * (1 + 0.02 * rng.standard_normal((len(s0), N)))
    out = {"rhs": ref["rhs"](T, s, p), "monitor_values": ref["monitor_values"](T, s, p)}
    for sc in SCHEMES:
        out[sc] = ref[sc](s, T, DT, p)
    return request.param, ode, s, p, out


def _calls(ns, s, p, suffix="", conv=lambda x: x):
    S, P = conv(s), conv(p)
    res = {"rhs": ns["rhs" + suffix](T, S, P), "monitor_values": ns["monitor_values" + suffix](T, S, P)}
    for sc in SCHEMES:
        res[sc] = ns[sc + suffix](S, T, DT, P)
    return res


def _check(name, got, ref):
    for k, r in ref.items():
        np.testing.assert_allclose(np.asarray(got[k]), r, rtol=RTOL, atol=ATOL, err_msg=f"{name}: {k}")


def test_torch(model):
    torch = pytest.importorskip("torch")
    name, ode, s, p, ref = model
    ns = _exec(get_code(ode, backend="torch", scheme=SCHEMES, format=Format.none))
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devs:
        got = _calls(ns, s, p, conv=lambda x: torch.as_tensor(x, device=dev))
        _check(f"{name} torch/{dev}", {k: v.cpu().numpy() for k, v in got.items()}, ref)


def test_torch_single_cell_and_grad(model):
    torch = pytest.importorskip("torch")
    name, ode, s, p, ref = model
    ns = _exec(get_code(ode, backend="torch", scheme=SCHEMES, format=Format.none))
    S = torch.as_tensor(s[:, 0])
    P = torch.as_tensor(p).requires_grad_(True)
    v = ns["rhs"](T, S, P)
    np.testing.assert_allclose(v.detach().numpy(), ref["rhs"][:, 0], rtol=RTOL, atol=ATOL)
    # gradients away from states that start at 0: e.g. the Land model's
    # (CaTrpn/cat50)**ntm has no finite d/dntm (or d/dCaTrpn) at CaTrpn = 0
    S = torch.where(S == 0, torch.full_like(S, 1e-3), S).requires_grad_(True)
    ns["rhs"](T, S, P).sum().backward()
    assert torch.isfinite(P.grad).all() and torch.isfinite(S.grad).all()


def test_numba_cpu(model):
    pytest.importorskip("numba")
    name, ode, s, p, ref = model
    ns = _exec(get_code(ode, backend="numba", scheme=SCHEMES, format=Format.none))
    _check(f"{name} numba/cpu", _calls(ns, s, p), ref)


def test_numba_cuda(model):
    numba = pytest.importorskip("numba")
    from numba import cuda
    if not cuda.is_available() or os.environ.get("NO_CUDA"):
        pytest.skip("no CUDA GPU")
    name, ode, s, p, ref = model
    ns = _exec(get_code(ode, backend="numba", scheme=SCHEMES, format=Format.none))
    got = _calls(ns, s, p, suffix="_cuda", conv=lambda x: cuda.to_device(np.ascontiguousarray(x)))
    _check(f"{name} numba/cuda", {k: v.copy_to_host() for k, v in got.items()}, ref)
