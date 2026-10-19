"""Templates for the PyTorch backend (method bodies as in gotranx's python template)."""
from __future__ import annotations

import functools
from textwrap import dedent, indent

from gotranx.templates.python import acc, missing_index, monitor_index, parameter_index, state_index


def _init(fname, name, names, values):
    comment = indent("#" + functools.reduce(acc, [f"{n}={v}" for n, v in zip(names, values)]), "    ")
    index = "state_index" if "state" in fname else "parameter_index"
    return dedent(
        f'''
def {fname}(*, device=None, dtype=torch.float64, **values):
    """Initial values as a 1D tensor; pass name=value to override."""
{comment}
    {name} = torch.tensor([{", ".join(map(str, values))}], dtype=dtype, device=device)
    for key, value in values.items():
        {name}[{index}(key)] = value
    return {name}
''')


def init_state_values(name, state_names, state_values, code=None, **kwargs):
    return _init("init_state_values", name, state_names, state_values)


def init_parameter_values(name, parameter_names, parameter_values, code=None, **kwargs):
    return _init("init_parameter_values", name, parameter_names, parameter_values)


def method(name, args, states, parameters, values, return_name, num_return_values,
           values_type="torch.empty_like(states)", shape_info="", missing_variables="", **kwargs):
    if values_type.startswith("numpy.zeros(shape)"):
        values_type = "torch.zeros(shape, dtype=states.dtype, device=states.device)"
    return dedent(
        f"""
def {name}({args}):

    # Assign states
{indent(states, "    ")}

    # Assign parameters
{indent(parameters, "    ")}
{indent(missing_variables, "    ")}
    # Assign expressions
    {shape_info}
    {return_name} = {values_type}
{indent(values, "    ")}

    return {return_name}
""")


__all__ = ["init_state_values", "init_parameter_values", "method", "parameter_index",
           "state_index", "monitor_index", "missing_index"]
