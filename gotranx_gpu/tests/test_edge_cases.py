"""Edge cases from gotranx's own suite, through the numba backend (CPU and CUDA).

Sources: tests/test_min_max.py, test_conditionals.py, test_missing_variables.py,
test_cuda_kernels.py ("other code paths") in the gotranx repository.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import gotranx
from gotranx.load import ode_from_string
from gotranx.schemes import Scheme

from conftest import DEVICES, ODEDIR, assert_close, call, numba_ns, numpy_ns, perturbed

T = 0.0

# --- Min / Max (test_min_max.py) ----------------------------------------------------------

MINMAX = """
parameters("m", k = 0.5)
states("m", y = 0.5)
expressions("m")
r = k*Max(y - 0.2, 0.0, 2.0*y - 3.0) + Min(y, 1.0, k)
dy_dt = -r
"""
# both sides of every kink and exactly on each: Max switches at 0.2 and 2.8, Min at 1.0 and k
Y = np.array([-1.0, 0.0, 0.1, 0.2, 0.3, 0.5, 0.9, 1.0, 1.5, 2.0, 2.8, 3.0, 3.5])


def _expected_minmax(y, k=0.5):
    return -(k * np.maximum(np.maximum(y - 0.2, 0.0), 2.0 * y - 3.0) + np.minimum(np.minimum(y, 1.0), k))


@pytest.mark.parametrize("device", DEVICES)
def test_min_max_at_and_around_kinks(device):
    nb = numba_ns(ode_from_string(MINMAX))
    out = call(nb, "rhs", device, T, Y[None, :].copy(), nb["init_parameter_values"]())
    assert_close(out[0], _expected_minmax(Y), rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("device", DEVICES)
def test_min_max_with_per_cell_parameters(device):
    ode = ode_from_string(MINMAX)
    nb = numba_ns(ode, per_cell_parameters=True)
    y, k = (a.ravel() for a in np.meshgrid(Y, [0.5, 1.5]))  # k on both sides of 1.0
    p = np.repeat(nb["init_parameter_values"]()[:, None], y.size, axis=1)
    p[nb["parameter_index"]("k")] = k
    out = call(nb, "rhs", device, T, y[None, :].copy(), p)
    assert_close(out[0], _expected_minmax(y, k), rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("device", DEVICES)
def test_grl_through_max_is_exact_on_each_side_and_finite_on_the_kink(device):
    ode = ode_from_string("parameters(k = 2.0, c = 0.5)\nstates(y = 0.0)\ndy_dt = c - k*Max(y - 0.2, 0.0)\n")
    nb = numba_ns(ode, scheme=["generalized_rush_larsen"])
    y0 = np.array([[0.0, 0.1, 0.2, 0.5, 1.0]])
    dt, k, c = 0.1, 2.0, 0.5
    y1 = call(nb, "generalized_rush_larsen", device, y0, 0.0, dt, nb["init_parameter_values"]())
    fixed = 0.2 + c / k
    expected = np.where(y0 < 0.2, y0 + c * dt, fixed + (y0 - fixed) * math.exp(-k * dt))
    off = y0 != 0.2
    assert_close(y1[off], expected[off])
    # numba evaluates only the branch taken, so unlike numpy.where there is no 0/0 even on the kink
    assert np.isfinite(y1).all()


@pytest.mark.parametrize("device", DEVICES)
def test_max_keeps_a_removable_pole_guarded(device):
    ode = ode_from_string("states(y = -50.0)\nr = (y + 40.0)/(exp((y + 40.0)/10.0) - 1.0)*Max(y, -60.0)\ndy_dt = -r\n")
    nb = numba_ns(ode)
    out = call(nb, "rhs", device, T, np.array([[-40.0, -30.0]]), nb["init_parameter_values"]())
    assert_close(out[0], [400.0, 30.0 * 10.0 / (math.e - 1.0)], rtol=1e-6, atol=0)


# --- Conditionals (test_min_max.py EXPORTED, test_conditionals.py) --------------------------

EXPRS = [
    "k*Max(y - 0.2, 0.0, 2.0*y - 3.0) + Min(y, 1.0, k)",
    "Min(Max(y, 0.0), 1.0)",
    "Max(Min(y, 1.0), k)",
    "Conditional(Ge(Max(y, 0.0), k), y, k)",
    "Max(Conditional(Gt(y, 0.0), y, -y), k)",
    "Max(y, 0.1, k, 2.0*y - 1.0)",
    "Min(y, 1.0, k, y*y, -y)",
    "Conditional(And(Gt(y, 0.0), Lt(y, 1.0), Gt(k, 0.0)), y, k)",
    "Conditional(Or(Lt(y, 0.0), Gt(y, 2.0)), y*y, Conditional(Lt(y, 1.0), 2*y, 3*y))",
    "Conditional(Eq(y, 1.0), 7.0, y)",
]
YS = np.array([-1.0, 0.0, 0.3, 0.5, 0.8, 1.0, 2.0, 2.5, 3.0])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("expr", EXPRS)
def test_conditionals_match_numpy(expr, device):
    ode = ode_from_string(f'parameters("m", k = 0.5)\nstates("m", y = 0.5)\nexpressions("m")\nr = {expr}\ndy_dt = -r\n')
    ref, nb = numpy_ns(ode), numba_ns(ode)
    p = ref["init_parameter_values"]()
    out = call(nb, "rhs", device, T, YS[None, :].copy(), p)
    # reference one cell at a time: gotranx's numpy code cannot do a 3-way And on many
    # cells with a scalar parameter (numpy.logical_and.reduce of mixed shapes)
    expected = np.array([ref["rhs"](T, np.array([y]), p)[0] for y in YS])
    assert_close(out[0], expected, rtol=1e-14, atol=0)


# --- Missing variables (test_missing_variables.py) ----------------------------------------

MISSING = """
parameters("inputs", u_in = 2.0, w_in = 3.0)
parameters("m", k = 0.5)
states("m", v = 1.0, y = 0.25)
expressions("m")
r = k*y - u_in
s = w_in*v
dy_dt = -r
dv_dt = -s + u_in
"""


@pytest.fixture(scope="module")
def split_ode():
    full = ode_from_string(MISSING)
    return full - full.get_component("inputs")


@pytest.mark.parametrize("device", DEVICES)
def test_missing_variables_per_cell(split_ode, device):
    ref = numpy_ns(split_ode, scheme=[Scheme.generalized_rush_larsen])
    nb = numba_ns(split_ode, scheme=["generalized_rush_larsen"])
    cells = np.array([[1.0, 1.1, 0.9], [0.25, 0.3, 0.2]])
    missing = np.array([[2.5, 2.0, 3.0], [-1.5, -1.0, -2.0]])  # u_in, w_in per cell
    p = ref["init_parameter_values"]()
    assert_close(call(nb, "rhs", device, T, cells, p, missing), ref["rhs"](T, cells, p, missing))
    assert_close(call(nb, "monitor_values", device, T, cells, p, missing), ref["monitor_values"](T, cells, p, missing))
    assert_close(call(nb, "generalized_rush_larsen", device, cells, T, 0.1, p, missing),
                 ref["generalized_rush_larsen"](cells, T, 0.1, p, missing))


def test_missing_variables_single_cell(split_ode):
    ref = numpy_ns(split_ode)
    nb = numba_ns(split_ode)
    s, m = np.array([1.0, 0.25]), np.array([2.5, -1.5])
    p = ref["init_parameter_values"]()
    assert_close(nb["rhs"](T, s.copy(), p, m.copy()), ref["rhs"](T, s, p, m))


# --- Other code paths on ToRORd (test_cuda_kernels.py) -------------------------------------

@pytest.fixture(scope="module")
def tor_ord():
    return gotranx.load_ode(ODEDIR / "ToRORd_dyn_chloride.ode")


@pytest.mark.parametrize("device", DEVICES)
def test_missing_values(tor_ord, device):
    values = {"INa": 0, "cai": 1}
    ref = numpy_ns(tor_ord, missing_values=values)
    nb = numba_ns(tor_ord, missing_values=values)
    s = perturbed(ref["init_state_values"](), 5)
    p = ref["init_parameter_values"]()
    assert_close(call(nb, "missing_values", device, T, s, p), ref["missing_values"](T, s, p), 1e-12, 1e-11)


@pytest.mark.parametrize("device", DEVICES)
def test_hybrid_rush_larsen_with_stiff_states(tor_ord, device):
    kw = dict(stiff_states=["v", "cai"])
    ref = numpy_ns(tor_ord, scheme=[Scheme.hybrid_rush_larsen], **kw)
    nb = numba_ns(tor_ord, scheme=["hybrid_rush_larsen"], **kw)
    s = perturbed(ref["init_state_values"](), 5)
    p = ref["init_parameter_values"]()
    with np.errstate(all="ignore"):
        expected = ref["hybrid_rush_larsen"](s, T, 0.01, p)
    assert_close(call(nb, "hybrid_rush_larsen", device, s, T, 0.01, p), expected, 1e-12, 1e-11)


def test_remove_unused_compiles_and_agrees(tor_ord):
    schemes = [s.value for s in Scheme if s.value in ("explicit_euler", "generalized_rush_larsen")]
    ref = numpy_ns(tor_ord, scheme=[Scheme(s) for s in schemes], remove_unused=True)
    nb = numba_ns(tor_ord, scheme=schemes, remove_unused=True)
    s = perturbed(ref["init_state_values"](), 5)
    p = ref["init_parameter_values"]()
    assert_close(nb["rhs"](T, s, p), ref["rhs"](T, s, p), 1e-12, 1e-11)
    for sc in schemes:
        with np.errstate(all="ignore"):
            expected = ref[sc](s, T, 0.01, p)
        assert_close(nb[sc](s, T, 0.01, p), expected, 1e-12, 1e-11)
