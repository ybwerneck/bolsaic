"""Train the network on the LAT loss, then start the ECG inverse from those weights.

    1. net  trained on the LAT misfit of the Method 1 map to the FIM truth
    2. the weights with the best LAT are kept (``lat_net.pt``)
    3. the ECG inverse starts from them

Environment: RUN_TRUTH, NAME, SEED, N_LAT, LR_LAT, ECG_LR, N_ITER.
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
SEED = int(os.environ.get('SEED', 0))
N_FIT = int(os.environ.get('N_FIT', 20000))
N_LATFT = int(os.environ.get('N_LATFT', 3000))        # fine-tune on the LAT loss
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
RUN = open_run(__file__, f'latnet_{NAME}_s{SEED}', header={'NAME': NAME, 'SEED': SEED})

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

N_LAT = int(os.environ.get('N_LAT', 5000))
LR_LAT = float(os.environ.get('LR_LAT', 1e-3))
ECG_LR = float(os.environ.get('ECG_LR', 1e-4))
N_ITER = int(os.environ.get('N_ITER', 1000))


def lat_loss():
    p = model.field_from_g(model.g(), *model.cv)
    t = op.potential(p, *model.cv, n_project=0)
    t = t - t.min()
    return ((t - tau_true) ** 2).mean() / scale ** 2


def score(tag):
    model.eval()
    with torch.no_grad():
        p = model()
        th = op.potential(p, *model.cv, n_project=0)
        V = ecg.from_tau(th)
        a = accuracy(th, tau_true)
        L = float((((th - th.min()) - tau_true) ** 2).mean() / scale ** 2)
        row = {'lat_loss': L, 'lat_rms': L ** 0.5 * scale, 'mae': a['mae'], 'corr': a['corr'],
               'ecg': float(((V - V_true) ** 2).mean() / (V_true ** 2).mean())}
    print(f"[{tag}] LAT loss {L:.3e}  rms {row['lat_rms']:.2f} ms  MAE {row['mae']:.2f} ms  "
          f"corr {row['corr']:.4f}  ECG {row['ecg']:.3e}", flush=True)
    return row


# ---- 1-2. train the network on the LAT loss; keep the best weights -------------
torch.manual_seed(SEED)
opt = torch.optim.Adam(model.net.parameters(), lr=LR_LAT)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_LAT, eta_min=LR_LAT * 0.02)
best, best_sd = 1e9, None
model.eval()
t0 = time.time()
for it in range(N_LAT + 1):
    opt.zero_grad(set_to_none=True)
    L = lat_loss()
    if float(L) < best:
        best = float(L)
        best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if it < N_LAT:
        L.backward()
        opt.step()
        sch.step()
    if it % 500 == 0:
        print(f'[LAT training] it {it:5d}  loss {float(L):.3e}  (best {best:.3e})', flush=True)
model.zero_grad(set_to_none=True)
model.load_state_dict(best_sd, strict=False)
torch.save(model.state_dict(), os.path.join(RUN, 'lat_net.pt'))
print(f'[LAT training] {time.time() - t0:.0f} s', flush=True)
row_lat = score('network after LAT training')

# ---- 3. the ECG inverse from those weights ---------------------------------
loss = ECGMismatch(grid, op, ecg, V_true, n_project=0, keep_best=False)
opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=ECG_LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_ITER, eta_min=ECG_LR * 0.02)
val = MapValidator(loss, tau_true, name='map', root=RUN, freq=N_ITER, dump_f=100000)
tr = Trainer(n_epochs=N_ITER, model=model, device=str(DEVICE), adaptive=False, lr=ECG_LR,
             print_steps=N_ITER, patience=None, optimizer=opt, scheduler=sched,
             ckpt_path=os.path.join(RUN, 'model.pt'))
tr.add_loss(loss, weigth=1)
tr.add_validator(val, freq=N_ITER)
tr.train()
row_ecg = score('after ECG inverse')
json.dump({'name': NAME, 'seed': SEED, 'n_lat': N_LAT, 'lr_lat': LR_LAT, 'ecg_lr': ECG_LR,
           'n_iter': N_ITER, 'lat_net': row_lat, 'after_ecg': row_ecg},
          open(os.path.join(RUN, 'summary.json'), 'w'), indent=1, default=float)
print('wrote', RUN)
