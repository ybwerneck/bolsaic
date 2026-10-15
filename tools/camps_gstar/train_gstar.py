"""Method 1 on DTI004, ECG inverse started from the LAT-optimal g*.

Same truth, model and ECG loss as ``train_m1`` (frozen H, speeds boxed at the
truth). Four steps:

    1. roots   the run's root nodes (candidates of S), fixed by the truth file
    2. truth   FIM map ``<NAME>_fim.npz`` from those roots at t = 0
    3. g*      g on S minimising the LAT misfit of the Method 1 map to the
               truth, started from g0 = truth restricted to S (free vector,
               N_LAT Adam steps)
    4. inverse the network is regressed onto g* (a frozen residual makes the
               start exact), then trained on the ECG mismatch as in train_m1

Scores (ECG error, corr, MAE, eikonal residual, sources, root ranks) are
written for three points: g0 = truth|S, g*, and the final network.

Environment: RUN_TRUTH, NAME, SEED, N_LAT, LR_G, N_REG, N_ITER, VAL_FREQ,
EPS_SCREEN, BOX, MIN_DEPTH, W_CONSIST (as in train_m1), RUN_OUT.

    RUN_TRUTH=outputs/truth5ti_set1 NAME=truth5ti_s1 \\
        python -m problems.camps_dti004.train_gstar
"""

import json
import os
import time

import numpy as np
import torch

from fisiocomPinn import Grid3D
from fisiocomPinn.Trainer import Trainer

from core import camps_mesh
from core.diagnostics import accuracy, admissibility, detect_sources
from core.forward import (ECGForward, einthoven_wilson, lead_field,
                          upstroke_for_spacing)
from core.loss import ECGMismatch, activation_map, consistency_penalty
from core.mesh import Geodesic, one_hop, surface_faces
from core.metric import Componentwise
from core.operator import DeltaPoisson
from core.output import open_run
from core.screened import ScreenedPoisson
from core.screened_front import ScreenedFront
from core.validate import MapValidator

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_TRUTH = os.path.join(HERE, os.environ.get('RUN_TRUTH',
                                              'outputs/truth5ti_set1'))
NAME = os.environ.get('NAME', 'truth5ti_s1')
SEED = int(os.environ.get('SEED', 0))
N_LAT = int(os.environ.get('N_LAT', 600))
LR_G = float(os.environ.get('LR_G', 0.1))               # ms per Adam step
N_REG = int(os.environ.get('N_REG', 3000))
N_ITER = int(os.environ.get('N_ITER', 1000))
VAL_FREQ = int(os.environ.get('VAL_FREQ', 250))
EPS_SCREEN = float(os.environ.get('EPS_SCREEN', 2.0))   # ms
BOX = float(os.environ.get('BOX', 0.01))
MIN_DEPTH = float(os.environ.get('MIN_DEPTH', 1.0))     # ms
W_CONSIST = float(os.environ.get('W_CONSIST', 0.0))
WIDTH, LAYERS, N_EIG = 64, 4, 40
LR = 5e-4
NT = 100
DUMP_F = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SPEED_NAMES = ('fibre', 'cross', 'endo_dense', 'endo_sparse')

RUN = open_run(__file__, f'gstar_{NAME}_s{SEED}',
               header={'RUN_TRUTH': RUN_TRUTH, 'NAME': NAME, 'SEED': SEED,
                       'N_LAT': N_LAT, 'LR_G': LR_G, 'N_REG': N_REG,
                       'N_ITER': N_ITER, 'EPS_SCREEN': EPS_SCREEN,
                       'BOX': BOX, 'W_CONSIST': W_CONSIST, 'DEVICE': DEVICE})

# ---- 1-2. roots and FIM truth, mesh, tensors (as train_m1) -----------------
mesh = camps_mesh.load(os.path.join(RUN_TRUTH, 'mesh.npz'))
A = mesh.A
A4 = np.stack([A[:, 0], A[:, 1] + A[:, 2], A[:, 3], A[:, 4]], axis=1)
sol = np.load(os.path.join(RUN_TRUTH, f'{NAME}.npz'))
sp_ = dict(zip([str(k) for k in sol['speed_keys']], sol['speeds']))
if abs(sp_['sheet'] - sp_['normal']) > 1e-12:
    raise SystemExit('the truth is not transversely isotropic')
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
tau = tau - tau.min()
tau_true = torch.as_tensor(tau, dtype=torch.float32, device=DEVICE)
scale = float(tau.max())
grid = Grid3D(mesh.xyz, mesh.tet)
S = mesh.candidates
endo = np.union1d(mesh.lvendo, mesh.rvendo)
surf = surface_faces(mesh.tet)
f_elem = np.linalg.eigh(A[:, 0])[1][:, :, -1]
print(f'{grid.N} nodes; S = {len(S)} candidates; roots {roots.tolist()}; '
      f'truth max {scale:.1f} ms', flush=True)

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
                      learn=True, solver='sparse',
                      screen_D=D_true).to(DEVICE)
idx, msk = one_hop(grid)
geo = Geodesic(np.asarray(mesh.xyz), idx, msk)
d_cand = np.linalg.norm(mesh.xyz[S][:, None] - mesh.xyz[S][None], axis=-1)
near = {int(r): np.argsort(d_cand[list(S).index(r)]) for r in roots}
nearest_cand = lambda n: int(np.argmin(np.linalg.norm(mesh.xyz[S] - mesh.xyz[n],
                                                      axis=1)))


def score(tag, g=None):
    """Scores of the map for g on S (``g=None``: the model's own g)."""
    model.eval()
    with torch.no_grad():
        p = model() if g is None else model.field_from_g(g, *model.cv)
        th_t = op.potential(p, *model.cv, n_project=0)
        V = ecg.from_tau(th_t)
        row = admissibility(op, p, th_t, model.cv)
        row.pop('ug')
        row.update(accuracy(th_t, tau_true))
    th = th_t.cpu().numpy()
    src = detect_sources(th, idx, msk, MIN_DEPTH)
    det = np.array([s['node'] for s in src], dtype=np.int64)
    rank = [min(int(np.where(near[int(r)] == nearest_cand(n))[0][0])
                for n in det) if len(det) else -1 for r in roots]
    row.update(ecg=float(((V - V_true) ** 2).mean() / (V_true ** 2).mean()),
               n_sources=len(src), root_rank=rank)
    print(f"[{tag}] ECG err {row['ecg']:.4e}; corr {row['corr']:.4f}; MAE "
          f"{row['mae']:.2f} ms; eik {row['eik']:.3f}; {len(src)} sources; "
          f"root ranks {rank}", flush=True)
    return row, th, p


# ---- baseline: g0 = truth restricted to S ----------------------------------
s_nodes = model.s_nodes
g0 = tau_true[s_nodes].clone()
row0, _, _ = score('g0 = truth|S', g0)


# ---- 3. g*: minimise the LAT misfit over g on S -----------------------------
def lat_loss(g):
    p = model.field_from_g(g, *model.cv)
    t = op.potential(p, *model.cv, n_project=0)
    t = t - t.min()
    return ((t - tau_true) ** 2).mean() / scale ** 2


gfree = torch.nn.Parameter(g0.clone())
optg = torch.optim.Adam([gfree], lr=LR_G)
schg = torch.optim.lr_scheduler.CosineAnnealingLR(optg, T_max=N_LAT,
                                                  eta_min=LR_G * 0.02)
model.eval()
t0 = time.time()
lat_hist = []
with torch.enable_grad():
    for it in range(N_LAT + 1):
        optg.zero_grad(set_to_none=True)
        L = lat_loss(gfree)
        lat_hist.append(float(L))
        if it < N_LAT:
            L.backward()
            optg.step()
            schg.step()
        if it % max(N_LAT // 6, 1) == 0:
            print(f'[LAT fit] it {it:5d}  loss/scale^2 {float(L):.4e}',
                  flush=True)
model.zero_grad(set_to_none=True)
print(f'[LAT fit] {time.time() - t0:.1f} s', flush=True)
gstar = gfree.detach().clone()
row_star, _, _ = score('g* (LAT-optimal)', gstar)
print(f'g* - g0 on S: max |d| {float((gstar - g0).abs().max()):.2f} ms, mean '
      f'|d| {float((gstar - g0).abs().mean()):.2f} ms', flush=True)

# ---- 4. network at g*, then the ECG inverse --------------------------------
y = (gstar / model.g_scale).detach()
optr = torch.optim.Adam(model.net.parameters(), lr=1e-3)
for it in range(N_REG):
    optr.zero_grad(set_to_none=True)
    r = ((model.net(s_nodes).squeeze(-1) - y) ** 2).mean()
    r.backward()
    optr.step()
with torch.no_grad():
    g_net = model.g_scale * model.net(s_nodes).squeeze(-1)
    resid = (gstar - g_net).detach()
print(f'[regression] network vs g*: max |err| {float(resid.abs().max()):.3f}'
      f' ms, rms {float(resid.pow(2).mean().sqrt()):.3f} ms', flush=True)
model.register_buffer('g_off', resid, persistent=False)
_g_net = model.g
model.g = lambda: _g_net() + model.g_off     # start exactly at g*
row_init, _, _ = score('network start (= g*)')

loss = ECGMismatch(grid, op, ecg, V_true, n_project=0, keep_best=False)
if W_CONSIST > 0:
    loss.terms = (('consist', W_CONSIST,
                   lambda t: consistency_penalty(op, loss.p, t, model.cv)),)
opt = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_ITER,
                                                   eta_min=LR * 0.02)
val = MapValidator(loss, tau_true, name='map', root=RUN, freq=VAL_FREQ,
                   dump_f=DUMP_F)
trainer = Trainer(n_epochs=N_ITER, model=model, device=str(DEVICE),
                  adaptive=False, lr=LR, print_steps=max(N_ITER // 4, 1),
                  patience=None, optimizer=opt, scheduler=sched,
                  ckpt_path=os.path.join(RUN, 'model.pt'))
trainer.add_loss(loss, weigth=1)
trainer.add_validator(val, freq=VAL_FREQ)
t0 = time.time()
model, loss_dict = trainer.train()
t_train = time.time() - t0
row_end, th_end, p_end = score('after ECG inverse')
cv_hat = [float(c.detach()) for c in model.cv]
row_end.update(cv=cv_hat, cv_err=[c / t - 1 for c, t in zip(cv_hat, v_true)])
print('speeds ' + ', '.join(f'{n} {c:.4f} ({100 * (c / t - 1):+.2f}%)'
                            for n, c, t in zip(SPEED_NAMES, cv_hat, v_true)))
print(f'train {t_train:.0f} s', flush=True)

np.savez_compressed(
    os.path.join(RUN, 'gstar.npz'), s_nodes=S, roots=roots,
    g0=g0.cpu().numpy(), gstar=gstar.cpu().numpy(), g_off=resid.cpu().numpy(),
    tau_true=tau, tau=th_end, lat_hist=np.array(lat_hist))
with open(os.path.join(RUN, 'summary.json'), 'w') as fh:
    json.dump({'setup': {'run_truth': RUN_TRUTH, 'name': NAME, 'seed': SEED,
                         'n_lat': N_LAT, 'lr_g': LR_G, 'n_reg': N_REG,
                         'n_iter': N_ITER, 'eps': EPS_SCREEN, 'box': BOX,
                         'w_consist': W_CONSIST, 'roots': roots.tolist(),
                         'speeds_true': v_true, 'tau_max': scale},
               'g0_truth_on_S': row0, 'gstar': row_star,
               'start_from_gstar': row_init, 'final': row_end,
               'regression_max_err_ms': float(resid.abs().max()),
               'history': [{'it': it, **r} for it, r in val.history]},
              fh, indent=1, default=float)
val.save_history()
print(f'wrote {RUN}')
