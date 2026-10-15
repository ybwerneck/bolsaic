"""Noise sweep: how much do the end metrics change with noise on the starting point?

START (a saved network, ``lat_net.pt`` of train_lat_then_ecg) is perturbed by
noise on its weights, sigma x (per-tensor std) x N(0,1), the same draws for every
start; each noisy start is trained on the ECG mismatch with each optimizer.

Environment: START, TAG (A|B), SIGMAS, N_DRAWS, OPTS (JSON list of [opt, lr]), N_ITER, RUN_OUT.
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
RUN = open_run(__file__, f'noise_{NAME}', header={'NAME': NAME})

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

START = os.environ['START']
TAG = os.environ.get('TAG', 'X')
SIGMAS = [float(x) for x in os.environ.get('SIGMAS', '0,0.01,0.03,0.1,0.3').split(',')]
N_DRAWS = int(os.environ.get('N_DRAWS', 3))
OPTS = json.loads(os.environ.get('OPTS', '[["adam",5e-5],["adam",1e-4],["adam",5e-4],["adam",1e-3],["sgd",1e-2]]'))
N_ITER = int(os.environ.get('N_ITER', 1000))
OUTJ = os.path.join(RUN, f'noise_{TAG}.json')

start_sd = {k: v.to(DEVICE) for k, v in torch.load(START, map_location=DEVICE).items()}
model.load_state_dict(start_sd, strict=False)
with torch.no_grad():
    g_ref = model.g().detach().clone()
dom = (g_ref - g_ref.min()) < 10.0
print(f'[{TAG}] start {START}; dominant candidates {int(dom.sum())}', flush=True)


def score():
    model.eval()
    with torch.no_grad():
        p = model()
        th = op.potential(p, *model.cv, n_project=0)
        V = ecg.from_tau(th)
        a = accuracy(th, tau_true)
        L = float((((th - th.min()) - tau_true) ** 2).mean() / scale ** 2)
        g = model.g()
        return {'lat_rms': L ** 0.5 * scale, 'mae': a['mae'], 'corr': a['corr'],
                'ecg': float(((V - V_true) ** 2).mean() / (V_true ** 2).mean()),
                'dg_dom': float((g - g_ref).pow(2)[dom].mean().sqrt())}


def noisy_state(sigma, draw, si):
    sd = {k: v.clone() for k, v in start_sd.items()}
    gen = torch.Generator(device='cpu').manual_seed(1000 * si + draw)
    for k, v in sd.items():
        if k.startswith('net.') and v.dtype.is_floating_point and v.numel() > 1:
            n = torch.randn(v.shape, generator=gen).to(v.device)
            sd[k] = v + sigma * v.std() * n
    return sd


results = []
for si, sigma in enumerate(SIGMAS):
    for d in range(1 if sigma == 0 else N_DRAWS):
        sd = noisy_state(sigma, d, si)
        for opt_name, lr in OPTS:
            model.load_state_dict(sd, strict=False)
            r0 = score()
            for fn in ('m.pt',):                         # the Trainer resumes from an existing checkpoint
                if os.path.exists(os.path.join(RUN, fn)):
                    os.remove(os.path.join(RUN, fn))
            try:
                loss = ECGMismatch(grid, op, ecg, V_true, n_project=0, keep_best=False)
                params = [q for q in model.parameters() if q.requires_grad]
                opt = (torch.optim.SGD(params, lr=lr) if opt_name == 'sgd'
                       else torch.optim.Adam(params, lr=lr))
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_ITER, eta_min=lr * 0.02)
                val = MapValidator(loss, tau_true, name='m', root=RUN, freq=N_ITER, dump_f=100000)
                tr = Trainer(n_epochs=N_ITER, model=model, device=str(DEVICE), adaptive=False, lr=lr,
                             print_steps=N_ITER, patience=None, optimizer=opt, scheduler=sched,
                             ckpt_path=os.path.join(RUN, 'm.pt'))
                tr.add_loss(loss, weigth=1)
                tr.add_validator(val, freq=N_ITER)
                tr.train()
                r1 = score()
            except Exception as e:                      # diverged / failed run
                r1 = {k: float('nan') for k in r0}
                print('  failed:', repr(e)[:80], flush=True)
            results.append({'tag': TAG, 'sigma': sigma, 'draw': d, 'opt': opt_name, 'lr': lr,
                            'start': r0, 'end': r1})
            print(f"[{TAG}] sigma {sigma:g} draw {d} {opt_name} {lr:g}: start g-shift {r0['dg_dom']:.2f} ms "
                  f"ECG {r0['ecg']:.2e} | end ECG {r1['ecg']:.2e} MAE {r1['mae']:.2f} corr {r1['corr']:.4f}", flush=True)
            json.dump(results, open(OUTJ, 'w'), default=float)
print(f'[{TAG}] DONE', flush=True)
