"""Every model in odefiles/, numba CPU and CUDA against gotranx's numpy code.

As gotranx's own tests/test_cuda_kernels.py: rhs, monitor_values and each scheme
on many cells with different states, a single cell, a step in place, and (CUDA)
threads past the last cell not writing.
"""
from __future__ import annotations

import numpy as np
import pytest

import gotranx
from gotranx.schemes import Scheme

from conftest import DEVICES, HAVE_CUDA, ODEFILES, assert_close, call, numba_ns, numpy_ns, perturbed

SCHEMES = [Scheme.explicit_euler, Scheme.generalized_rush_larsen, Scheme.hybrid_rush_larsen]
N_CELLS, T, DT = 257, 1.3, 0.01  # 257: not a multiple of the 128 threads per block
# tolerance: same arithmetic, but libm (numpy, numba CPU) and CUDA's exp/pow may
# differ in the last bit, amplified on derivatives that are differences of large fluxes
RTOL, ATOL = 1e-12, 1e-11


@pytest.fixture(scope="module", params=ODEFILES, ids=lambda p: p.stem)
def model(request):
    ode = gotranx.load_ode(request.param)
    ref = numpy_ns(ode, scheme=SCHEMES)
    nb = numba_ns(ode, scheme=[s.value for s in SCHEMES])
    s = perturbed(ref["init_state_values"](), N_CELLS)
    p = ref["init_parameter_values"]()
    return ode, ref, nb, s, p


@pytest.mark.parametrize("device", DEVICES)
def test_rhs_and_monitors(model, device):
    _, ref, nb, s, p = model
    assert_close(call(nb, "rhs", device, T, s, p), ref["rhs"](T, s, p), RTOL, ATOL)
    assert_close(call(nb, "monitor_values", device, T, s, p), ref["monitor_values"](T, s, p), RTOL, ATOL)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scheme", [s.value for s in SCHEMES])
def test_schemes(model, device, scheme):
    _, ref, nb, s, p = model
    with np.errstate(all="ignore"):  # numpy.where evaluates both branches
        expected = ref[scheme](s, T, DT, p)
    assert_close(call(nb, scheme, device, s, T, DT, p), expected, RTOL, ATOL)


def test_single_cell(model):
    _, ref, nb, s, p = model
    assert_close(nb["rhs"](T, s[:, 0].copy(), p), ref["rhs"](T, s[:, 0], p), RTOL, ATOL)
    assert_close(nb["generalized_rush_larsen"](s[:, 0].copy(), T, DT, p),
                 ref["generalized_rush_larsen"](s[:, 0], T, DT, p), RTOL, ATOL)


def test_init_values_match(model):
    _, ref, nb, _, _ = model
    np.testing.assert_array_equal(nb["init_state_values"](), ref["init_state_values"]())
    np.testing.assert_array_equal(nb["init_parameter_values"](), ref["init_parameter_values"]())
    assert nb["state"] == ref["state"] and nb["parameter"] == ref["parameter"]


@pytest.mark.parametrize("device", DEVICES)
def test_step_in_place_equals_out_of_place(model, device):
    _, ref, nb, s, p = model
    out = call(nb, "generalized_rush_larsen", device, s, T, DT, p)
    if device == "cpu":
        x = s.copy()
        nb["generalized_rush_larsen"](x, T, DT, p, out=x)
    else:
        from numba import cuda
        xd = cuda.to_device(s)
        nb["generalized_rush_larsen_cuda"](xd, T, DT, cuda.to_device(p), out=xd)
        x = xd.copy_to_host()
    np.testing.assert_array_equal(x, out)


@pytest.mark.skipif(not HAVE_CUDA, reason="no CUDA GPU")
def test_cuda_trailing_threads_do_not_write(model):
    # 5 cells on 2 blocks of 4 threads: threads 5-7 must return without writing.
    # out is a view of the first 5 columns of an 8-column buffer filled with nan.
    from numba import cuda
    _, ref, nb, s, p = model
    s5 = np.ascontiguousarray(s[:, :5])
    buf = cuda.to_device(np.full((s.shape[0], 8), np.nan))
    nb["rhs_cuda"](T, cuda.to_device(s5), cuda.to_device(p), out=buf[:, :5], threads=4)
    host = buf.copy_to_host()
    assert np.isnan(host[:, 5:]).all()
    assert_close(host[:, :5], ref["rhs"](T, s5, p), RTOL, ATOL)
