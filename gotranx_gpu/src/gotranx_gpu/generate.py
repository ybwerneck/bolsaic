"""Generate torch / numba code from a .ode file, mirroring ``gotranx ode2py``.

    python -m gotranx_gpu model.ode --backend torch --scheme generalized_rush_larsen -o model_torch.py
    python -m gotranx_gpu model.ode --backend numba --scheme forward_explicit_euler -o model_numba.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

from gotranx.cli.utils import add_schemes
from gotranx.codegen.python import Format, get_formatter
from gotranx.load import load_ode
from gotranx.ode import ODE
from gotranx.schemes import Scheme

from .numba_backend import NumbaCodeGenerator
from .torch_backend import TorchCodeGenerator

BACKENDS = {"torch": TorchCodeGenerator, "numba": NumbaCodeGenerator}


def get_code(ode: ODE, backend: str = "numba", scheme: list[Scheme | str] | None = None,
             delta: float = 1e-8, stiff_states: list[str] | None = None, cse: bool = True,
             remove_unused: bool = False, missing_values: dict[str, int] | None = None,
             format: Format = Format.black, **backend_kwargs) -> str:
    codegen = BACKENDS[backend](ode, remove_unused=remove_unused, **backend_kwargs)
    scheme = [Scheme(s) for s in (scheme or [])]
    comp = [
        codegen.imports(),
        codegen.parameter_index(),
        codegen.state_index(),
        codegen.monitor_index(),
        codegen.missing_index(),
        codegen.initial_parameter_values(),
        codegen.initial_state_values(),
        codegen.rhs(),
        codegen.monitor_values(),
        codegen.missing_values(missing_values) if missing_values else "",
    ] + add_schemes(codegen, scheme=scheme, delta=delta, stiff_states=stiff_states, cse=cse)
    return get_formatter(format)("\n".join(comp))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gotranx_gpu", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ode", type=Path)
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="numba")
    ap.add_argument("--scheme", action="append", default=[], choices=[s.value for s in Scheme])
    ap.add_argument("--per-cell-parameters", action="store_true",
                    help="numba: parameters are (num_parameters, N) instead of (num_parameters,)")
    ap.add_argument("--stiff-state", action="append", default=None, dest="stiff_states",
                    help="state treated as stiff by hybrid_rush_larsen (repeatable)")
    ap.add_argument("--cache", action="store_true",
                    help="numba: compile with cache=True (machine code kept on disk between runs)")
    ap.add_argument("--no-format", action="store_true")
    ap.add_argument("-o", "--output", type=Path)
    a = ap.parse_args(argv)
    kw = {"per_cell_parameters": a.per_cell_parameters, "cache": a.cache} if a.backend == "numba" else {}
    code = get_code(load_ode(a.ode), backend=a.backend, scheme=a.scheme, stiff_states=a.stiff_states,
                    format=Format.none if a.no_format else Format.black, **kw)
    out = a.output or a.ode.with_name(f"{a.ode.stem}_{a.backend}.py")
    out.write_text(code)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
