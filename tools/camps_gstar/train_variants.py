"""One start network at g*, many optimizers: how much does training move it?

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
RUN = open_run(__file__, f'variants_{NAME}_s{SEED}',
               header={'NAME': NAME, 'GSTAR': GSTAR, 'SEED': SEED,
                       'N_FIT': N_FIT, 'VARIANTS': VARIANTS})

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

# ---- 2. g* from the LAT fit -------------------------------------------------
z = np.load(GSTAR)
assert np.array_equal(z['s_nodes'], np.asarray(S)), 'S differs from the g* run'
gstar = torch.as_tensor(z['gstar'], dtype=torch.float32, device=DEVICE)
gmin = gstar.min()
active = (gstar - gmin) < 5 * EPS_SCREEN
print(f'g* from {GSTAR}; S={len(S)}, active (g*-min < {5 * EPS_SCREEN:g} ms): '
      f'{int(active.sum())}', flush=True)

# ---- 3. one network fitted to g*, saved ------------------------------------
START = os.path.join(RUN, 'start_net.pt')
y = (gstar / model.g_scale).detach()
wts = 1.0 + 50.0 * active.float()                      # the active set matters most
torch.manual_seed(SEED)
opt = torch.optim.Adam(model.net.parameters(), lr=2e-3)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_FIT, eta_min=1e-6)
t0 = time.time()
for it in range(N_FIT):
    opt.zero_grad(set_to_none=True)
    r = (wts * (model.net(s_nodes).squeeze(-1) - y) ** 2).mean()
    r.backward()
    opt.step()
    sch.step()
with torch.no_grad():
    err = model.g() - gstar
print(f'[start net] fitted in {time.time() - t0:.0f} s: |g - g*| on active '
      f'rms {float(err[active].pow(2).mean().sqrt()):.3f} / max '
      f'{float(err[active].abs().max()):.3f} ms; on all S rms '
      f'{float(err.pow(2).mean().sqrt()):.3f} / max {float(err.abs().max()):.3f}'
      f' ms', flush=True)
def lat_loss(g):
    p = model.field_from_g(g, *model.cv)
    t = op.potential(p, *model.cv, n_project=0)
    t = t - t.min()
    return ((t - tau_true) ** 2).mean() / scale ** 2


with torch.no_grad():
    print(f'[start net] LAT loss after regression {float(lat_loss(model.g())):.3e} '
          f'(g* itself {float(lat_loss(gstar)):.3e})', flush=True)
optf = torch.optim.Adam(model.net.parameters(), lr=1e-4)
schf = torch.optim.lr_scheduler.CosineAnnealingLR(optf, T_max=max(N_LATFT, 1), eta_min=1e-6)
for it in range(N_LATFT):
    optf.zero_grad(set_to_none=True)
    L = lat_loss(model.g())
    L.backward()
    optf.step()
    schf.step()
model.zero_grad(set_to_none=True)
with torch.no_grad():
    err = model.g() - gstar
    print(f'[start net] LAT loss after fine-tune {float(lat_loss(model.g())):.3e}; '
          f'|g - g*| active rms {float(err[active].pow(2).mean().sqrt()):.3f} ms, '
          f'all rms {float(err.pow(2).mean().sqrt()):.3f} ms', flush=True)
torch.save(model.state_dict(), START)


def digest(sd):
    """SHA-256 of a state dict's tensors in key order (bit identity)."""
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


start_sd = torch.load(START, map_location=DEVICE)
START_HASH = digest(start_sd)
print(f'start checkpoint {START_HASH}', flush=True)


def score():
    """Scores of the model's own map vs the FIM truth."""
    model.eval()
    with torch.no_grad():
        p = model()
        th = op.potential(p, *model.cv, n_project=0)
        V = ecg.from_tau(th)
        row = accuracy(th, tau_true)
        g = model.g()
        row.update(ecg=float(((V - V_true) ** 2).mean() / (V_true ** 2).mean()),
                   dg_active_rms=float((g - g0).pow(2)[active].mean().sqrt()),
                   dg_all_rms=float((g - g0).pow(2).mean().sqrt()),
                   dg_active_max=float((g - g0).abs()[active].max()),
                   cv_rel_err=[float(c) / t - 1 for c, t in zip(model.cv, v_true)])
    return row


model.load_state_dict(start_sd, strict=False)
with torch.no_grad():
    g0 = model.g().detach().clone()                    # g of the start network
res = []
for v in VARIANTS:
    model.load_state_dict(start_sd, strict=False)
    assert digest(model.state_dict()) == START_HASH, 'start is not identical'
    r0 = score()
    loss = ECGMismatch(grid, op, ecg, V_true, n_project=0, keep_best=False)
    params = [q for q in model.parameters() if q.requires_grad]
    if v['opt'] == 'sgd':
        opt = torch.optim.SGD(params, lr=v['lr'], momentum=v.get('momentum', 0.0))
    else:
        opt = torch.optim.Adam(params, lr=v['lr'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=v['n_iter'], eta_min=v['lr'] * 0.02)
    val = MapValidator(loss, tau_true, name=f"map_{v['name']}", root=RUN,
                       freq=v['n_iter'], dump_f=100000)
    tr = Trainer(n_epochs=v['n_iter'], model=model, device=str(DEVICE),
                 adaptive=False, lr=v['lr'], print_steps=v['n_iter'],
                 patience=None, optimizer=opt, scheduler=sched,
                 ckpt_path=os.path.join(RUN, f"model_{v['name']}.pt"))
    tr.add_loss(loss, weigth=1)
    tr.add_validator(val, freq=v['n_iter'])
    t0 = time.time()
    tr.train()
    r1 = score()
    print(f"{v['name']:14s} start ECG {r0['ecg']:.3e} corr {r0['corr']:.4f} "
          f"MAE {r0['mae']:.2f} | end ECG {r1['ecg']:.3e} corr {r1['corr']:.4f}"
          f" MAE {r1['mae']:.2f} | moved g: active rms {r1['dg_active_rms']:.2f}"
          f" max {r1['dg_active_max']:.2f}, all rms {r1['dg_all_rms']:.2f} ms "
          f"[{time.time() - t0:.0f}s]", flush=True)
    res.append({**v, 'start': r0, 'end': r1})
json.dump({'name': NAME, 'start_hash': START_HASH, 'seed': SEED, 'n_fit': N_FIT,
           'fit_err_active_rms': float(err[active].pow(2).mean().sqrt()),
           'fit_err_all_rms': float(err.pow(2).mean().sqrt()),
           'variants': res}, open(os.path.join(RUN, 'summary.json'), 'w'),
          indent=1, default=float)
print('wrote', RUN)
