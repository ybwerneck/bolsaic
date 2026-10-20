"""Matplotlib figures for the native comparison: python bench/plot_report.py -> bench/fig_*.png"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MODELS = [("beeler_reuter_1977", "Beeler–Reuter (8)"), ("tentusscher_panfilov_2006_M_cell", "ten Tusscher (19)"),
          ("ToRORd_dyn_chloride", "ToRORd (52)")]
SERIES = [("numpy", "numpy (native)", "#999999", "-", 1.5), ("c_1t", "C 1 thread (native)", "#d9a066", "--", 1.8),
          ("c_omp", "C OpenMP 12 (native)", "#b06020", "-", 1.8), ("cuda_c", "CUDA C (native)", "#6b2d0a", "-", 3),
          ("numba_1t", "numba 1 thread", "#7fd0bf", "--", 1.8), ("numba", "numba 12 threads", "#14897a", "-", 1.8),
          ("numba_cuda", "numba CUDA", "#063f36", "-", 3)]
data = {m: json.loads((HERE / f"native_{m}.json").read_text()) for m, _ in MODELS}

fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=False)
for ax, (m, name) in zip(axs, MODELS):
    rows = data[m]["rows"]
    for k, lab, c, ls, lw in SERIES:
        N = [r["N"] for r in rows if k in r]
        ax.loglog(N, [r[k] * 1e3 for r in rows if k in r], ls, color=c, lw=lw, marker="o", ms=3.5, label=lab)
    ax.set_title(name); ax.set_xlabel("cells"); ax.grid(True, which="major", alpha=.3)
axs[0].set_ylabel("ms per GRL step")
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=7, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(HERE / "fig_step_time.png", dpi=130); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for ax, (m, name) in zip(axs, MODELS):
    rows = data[m]["rows"]; N = [r["N"] for r in rows]
    for (a, b, lab, c) in [("c_1t", "numba_1t", "CPU 1 thread", "#7fd0bf"), ("c_omp", "numba", "CPU 12 threads", "#14897a"),
                           ("cuda_c", "numba_cuda", "GPU", "#063f36")]:
        ax.semilogx(N, [r[a] / r[b] for r in rows], "-o", color=c, lw=2, ms=4, label=lab)
    ax.axhline(1, color="k", lw=.8); ax.set_title(name); ax.set_xlabel("cells"); ax.grid(True, alpha=.3)
axs[0].set_ylabel("native time / numba time  (>1: numba faster)")
axs[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(HERE / "fig_ratio.png", dpi=130); plt.close(fig)

ct = {r["model"]: r for r in json.loads((HERE / "compile_time.json").read_text()) if r["mode"] == "nocache"}
F = ["rhs", "monitor_values", "generalized_rush_larsen"]
order = [("beeler_reuter_1977", "Beeler–Reuter"), ("tentusscher_panfilov_2006_M_cell", "ten Tusscher"),
         ("ORdmm_Land", "ORd-Land"), ("ToRORd_dyn_chloride", "ToRORd")]
fig, ax = plt.subplots(figsize=(10, 4.2))
x = range(len(order)); w = 0.38
load = [ct[m]["load_s"] for m, _ in order]; cg = [ct[m]["codegen_s"] for m, _ in order]
fe = [sum(ct[m][f"cpu_{f}"]["frontend_s"] for f in F) for m, _ in order]
ll = [sum(ct[m][f"cpu_{f}"]["llvm_s"] for f in F) for m, _ in order]
cu = [sum(ct[m][f"cuda_{f}"]["first_call_s"] for f in F) for m, _ in order]
xl = [i - w / 2 for i in x]; xr = [i + w / 2 for i in x]
ax.bar(xl, load, w, color="#bbbbbb", label="gotranx load")
ax.bar(xl, cg, w, bottom=load, color="#888888", label="gotranx codegen")
b2 = [a + b for a, b in zip(load, cg)]
ax.bar(xl, fe, w, bottom=b2, color="#14897a", label="numba front end (CPU)")
ax.bar(xl, ll, w, bottom=[a + b for a, b in zip(b2, fe)], color="#7fd0bf", label="LLVM (CPU)")
ax.bar(xr, cu, w, color="#063f36", label="numba CUDA (all stages)")
nat = {m: data[m] for m, _ in MODELS}
for i, (m, _) in enumerate(order):
    if m in nat:
        ax.plot([xr[i]], [nat[m]["compile_nvcc_s"]], "v", color="#6b2d0a", ms=9, label="nvcc" if i == 0 else None)
        ax.plot([xl[i]], [nat[m]["compile_gcc_s"]], "v", color="#b06020", ms=9, label="gcc" if i == 0 else None)
ax.set_xticks(list(x), [n for _, n in order]); ax.set_ylabel("seconds, first run (no cache)")
ax.legend(frameon=False, fontsize=9); ax.grid(True, axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(HERE / "fig_compile.png", dpi=130); plt.close(fig)
print("ok")
