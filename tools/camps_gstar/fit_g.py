"""LAT fit of g on S: how low does the misfit go? (converged floor test)

Stage 3 of the pipeline alone, with several optimizers and long runs from g0 =
truth|S. Environment: RUN_TRUTH, NAME, EPS_SCREEN, BOX, N_ADAM.

OLD DOC:

Protocol (per truth):
    1. the FIM truth and its roots (as ``train_m1``)
    2. g* on S from the LAT fit (``train_gstar``; read from its ``gstar.npz``)
    3. ONE network fitted to g* (seeded, no residual), saved to ``start_net.pt``
    4. for each variant (optimizer, lr, iterations) the saved start is loaded
       again, checked to be bit-identical, and trained on the ECG mismatch

Reported per variant: ECG error, LAT corr / MAE, and how far training moved g
and the network from the start (g rms on the active candidates and on all of S).

Environment: RUN_TRUTH, NAME, GSTAR (path of gstar.npz), SEED, N_FIT, VARIANTS
(JSON list of {"name", "opt", "lr", "n_iter"[, "momentum"]}), EPS_SCREEN, BOX.
"""
import hashlib
import json
import os
import time

import numpy as np
import torch

from fisiocomPinn import Grid3D
from fisiocomPinn.Trainer import Trainer

from core import camps_mesh
from core.diagnostics import accuracy, admissibility
from core.forward import (ECGForward, einthoven_wilson, lead_field,
                          upstroke_for_spacing)
from core.loss import ECGMismatch
from core.mesh import surface_faces
from core.metric import Componentwise
from core.operator import DeltaPoisson
from core.output import open_run
from core.screened import ScreenedPoisson
from core.screened_front import ScreenedFront
from core.validate import MapValidator

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_TRUTH = os.path.join(HERE, os.environ.get('RUN_TRUTH', 'outputs/truth5ti_set1'))
NAME = os.environ.get('NAME', 'truth5ti_s1')
GSTAR = os.environ.get('GSTAR', os.path.join(
    HERE, f'outputs/gstar_{NAME}_s0/gstar.npz'))
SEED = int(os.environ.get('SEED', 0))
N_ADAM = int(os.environ.get('N_ADAM', 4000))
EPS_SCREEN = float(os.environ.get('EPS_SCREEN', 2.0))
BOX = float(os.environ.get('BOX', 0.01))
VARIANTS = json.loads(os.environ.get('VARIANTS', json.dumps([
    {'name': 'adam_1e-3', 'opt': 'adam', 'lr': 1e-3, 'n_iter': 1000},
    {'name': 'adam_5e-4', 'opt': 'adam', 'lr': 5e-4, 'n_iter': 1000},
    {'name': 'adam_1e-4', 'opt': 'adam', 'lr': 1e-4, 'n_iter': 1000},
    {'name': 'adam_5e-5', 'opt': 'adam', 'lr': 5e-5, 'n_iter': 1000},
    {'name': 'sgd_1e-2', 'opt': 'sgd', 'lr': 1e-2, 'n_iter': 1000},
    {'name': 'sgd_1e-1', 'opt': 'sgd', 'lr': 1e-1, 'n_iter': 1000},
    {'name': 'adam_5e-4_x5', 'opt': 'adam', 'lr': 5e-4, 'n_iter': 5000}])))
WIDTH, LAYERS, N_EIG, NT = 64, 4, 40, 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SPEED_NAMES = ('fibre', 'cross', 'endo_dense', 'endo_sparse')
RUN = open_run(__file__, f'fitg_{NAME}', header={'NAME': NAME, 'N_ADAM': N_ADAM})

# ---- truth, mesh, forward (as train_gstar) ---------------------------------
mesh = camps_mesh.load(os.path.join(RUN_TRUTH, 'mesh.npz'))
A = mesh.A
A4 = np.stack([A[:, 0], A[:, 1] + A[:, 2], A[:, 3], A[:, 4]], axis=1)
sol = np.load(os.path.join(RUN_TRUTH, f'{NAME}.npz'))
sp_ = dict(zip([str(k) for k in sol['speed_keys']], sol['speeds']))
v_true = [float(sp_['fibre']), float(sp_['sheet']), float(sp_['endo_dense']),
          float(sp_['endo_sparse'])]
keep = [k for k in range(A4.shape[1]) if np.abs(A4[:, k]).max() > 0]
A4 = A4[:, keep]
v_true = [v_true[k] for k in keep]
SPEED_NAMES = tuple(SPEED_NAMES[k] for k in keep)
metric = Componentwise(A4, names=SPEED_NAMES)
D_true = metric.tensor(*v_true)
roots = sol['root_nodes'].astype(np.int64)
tau = np.load(os.path.join(RUN_TRUTH, f'{NAME}_fim.npz'))['tau'].astype(float)
tau -= tau.min()
tau_true = torch.as_tensor(tau, dtype=torch.float32, device=DEVICE)
scale = float(tau.max())
grid = Grid3D(mesh.xyz, mesh.tet)
S = mesh.candidates
endo = np.union1d(mesh.lvendo, mesh.rvendo)
surf = surface_faces(mesh.tet)
f_elem = np.linalg.eigh(A[:, 0])[1][:, :, -1]
ELECS = mesh.electrodes
upstroke, _ = upstroke_for_spacing(grid, v_true[1])
grad_Z = torch.einsum('ab,bed->aed', einthoven_wilson(torch.eye(len(ELECS))),
                      lead_field(grid, ELECS))
t_grid = np.linspace(0.0, scale * 1.05, NT)
ecg = ECGForward(grid, grad_Z, t_grid, upstroke=upstroke, device=str(DEVICE))
with torch.no_grad():
    V_true = ecg.from_tau(tau_true)
op = DeltaPoisson(grid, metric=metric, sparse=str(DEVICE)).to(str(DEVICE))
metric.to(str(DEVICE))
a_s = ScreenedPoisson(grid, f_elem, v_true[0], v_true[1]).surface_weights(
    endo, surf)[S]
torch.manual_seed(SEED)
cvb = [(v, v * (1 - BOX), v * (1 + BOX)) for v in v_true]
model = ScreenedFront(grid, f_elem, S, a_s, EPS_SCREEN, g_scale=scale,
                      n_eig=N_EIG, n_layers=LAYERS, width=WIDTH, cv=cvb,
                      learn=True, solver='sparse', screen_D=D_true).to(DEVICE)
s_nodes = model.s_nodes


g0 = tau_true[s_nodes].clone()
active = (g0 - g0.min()) < 10.0
print(f'S={len(S)}; active candidates (g0-min<10 ms): {int(active.sum())}', flush=True)


def lat_loss(g):
    p = model.field_from_g(g, *model.cv)
    t = op.potential(p, *model.cv, n_project=0)
    t = t - t.min()
    return ((t - tau_true) ** 2).mean() / scale ** 2


def report(tag, g, L):
    with torch.no_grad():
        p = model.field_from_g(g, *model.cv)
        th = op.potential(p, *model.cv, n_project=0)
        V = ecg.from_tau(th)
        a = accuracy(th, tau_true)
        e = float(((V - V_true) ** 2).mean() / (V_true ** 2).mean())
    print(f"{tag:26s} loss {L:.3e}  LAT rms {1e3 * (L ** .5) * scale / 1e3:.3f} ms  corr {a['corr']:.5f}  MAE {a['mae']:.3f} ms  ECG {e:.2e}", flush=True)
    return {'tag': tag, 'loss': L, 'corr': a['corr'], 'mae': a['mae'], 'ecg': e}


model.eval()
res = []
with torch.no_grad():
    res.append(report('g0 = truth|S', g0, float(lat_loss(g0))))
for lr in (0.3, 0.1, 0.03):
    g = torch.nn.Parameter(g0.clone())
    opt = torch.optim.Adam([g], lr=lr)
    best = 1e9
    t0 = time.time()
    for it in range(N_ADAM):
        opt.zero_grad(set_to_none=True)
        L = lat_loss(g)
        L.backward()
        opt.step()
        best = min(best, float(L))
        if it in (N_ADAM // 8, N_ADAM // 2, N_ADAM - 1):
            print(f'  adam lr {lr}: it {it:5d} loss {float(L):.3e}', flush=True)
    res.append(report(f'adam constant lr {lr}', g.detach(), float(lat_loss(g.detach()))))
g = torch.nn.Parameter(g0.clone())
opt = torch.optim.LBFGS([g], lr=1.0, max_iter=20, history_size=50, line_search_fn='strong_wolfe')
def closure():
    opt.zero_grad(set_to_none=True)
    L = lat_loss(g)
    L.backward()
    return L
for k in range(30):
    opt.step(closure)
    if k in (0, 4, 14, 29):
        print(f'  lbfgs outer {k:3d} loss {float(lat_loss(g.detach())):.3e}', flush=True)
res.append(report('lbfgs strong-wolfe', g.detach(), float(lat_loss(g.detach()))))
json.dump(res, open(os.path.join(RUN, 'summary.json'), 'w'), indent=1, default=float)
print('wrote', RUN)
