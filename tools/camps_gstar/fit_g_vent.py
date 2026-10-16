"""Ventricles (pac2): Method 1 with g0 = tau* on S against the LAT-optimal g*.

Truth, S, electrodes and ECG as in ``problems/screened_poisson/train_method1.py``
(source_face_3d's pair: LV and RV free-wall patches at mid-height, t = 0, FIM at
cv (0.052, 0.017); DESIGN 'recoverable' S = endocardium, 'chamber' S = LV
endocardium, the RV source off S; Berg's electrodes, leads I, II, V1..V6).
Method 1 forward with the TRUE cv everywhere (screened solve by the sparse
factorisation of H at the true tensor, lumped mass, eps; then DeltaPoisson).
Nothing is trained but g:

    g0  = tau* on S
    g*  = free g on S minimising mean((phi - min phi - tau*)^2) / max(tau*)^2,
          started at g0, Adam with constant step LR_G for N_FIT steps

Scores (as fit_g_slab, plus ECG): LAT loss, LAT MAE and rms (means removed),
corr, eikonal residual, basins, spread of phi over the true source patches,
u <= 0, and the relative ECG error against the truth's ECG.

Environment: DESIGN, EPS, N_FIT, LR_G, RUN_OUT.
"""
import json
import os
import time

import numpy as np
import torch

from core import pac2
from core.diagnostics import accuracy, basin_persistence, eikonal_rms
from core.fiber import eikonal_fim
from core.forward import ECGForward, einthoven_wilson, lead_field, upstroke_for_spacing
from core.mesh import one_hop, surface_faces
from core.metric import TransverseIsotropic
from core.operator import DeltaPoisson
from core.output import open_run
from core.screened import ScreenedPoisson
from core.screened_front import ScreenedFront

DESIGN = os.environ.get('DESIGN', 'recoverable')
EPS = float(os.environ.get('EPS', 2.0))
N_FIT = int(os.environ.get('N_FIT', 3000))
LR_G = float(os.environ.get('LR_G', 0.1))
CV_L, CV_T = 0.052, 0.017
MID_AB, PATCH_R, NT = 0.5, 0.035, 80
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RUN = open_run(__file__, f'vent_gstar_{DESIGN}_e{EPS:g}',
               header={'DESIGN': DESIGN, 'EPS': EPS, 'N_FIT': N_FIT, 'LR_G': LR_G})

vent = pac2.ventricles()
grid, verts, tets, f_elem = vent.grid, vent.verts, vent.tets, vent.f_elem
lab = pac2.labels(tets)
lv, rv, endo, ab = lab.lv, lab.rv, lab.endo, lab.ab
surf = surface_faces(tets)
nb_idx, nb_msk = one_hop(grid)


def free_wall(own, other, height=MID_AB):
    from scipy.spatial import cKDTree
    mid = own[np.abs(ab[own] - height) < 0.05]
    d, _ = cKDTree(verts[other]).query(verts[mid])
    return int(mid[np.argmax(d)])


centres = [free_wall(lv, rv), free_wall(rv, lv)]
patches = [endo[np.linalg.norm(verts[endo] - verts[c], axis=1) <= PATCH_R] for c in centres]
src = np.concatenate(patches)
tau = eikonal_fim(grid, f_elem, CV_L, CV_T, src)
tau = tau - tau.min()
scale = float(tau.max())
tau_t = torch.as_tensor(tau, dtype=torch.float32, device=DEVICE)
S = np.sort(endo if DESIGN == 'recoverable' else lv)
sp = ScreenedPoisson(grid, f_elem, CV_L, CV_T, lumped=True)
a_s = sp.surface_weights(S, surf)[S]
fe = np.einsum('ej,ek->ejk', f_elem, f_elem)
D_elem = CV_T ** 2 * np.eye(3)[None] + (CV_L ** 2 - CV_T ** 2) * fe
model = ScreenedFront(grid, f_elem, S, a_s, EPS, g_scale=scale,
                      cv=[(CV_L, CV_L, CV_L), (CV_T, CV_T, CV_T)], learn=False,
                      solver='sparse', screen_D=D_elem).to(DEVICE)
metric = TransverseIsotropic(torch.as_tensor(f_elem, dtype=torch.float32))
op = DeltaPoisson(grid, metric=metric, sparse=str(DEVICE)).to(str(DEVICE))
metric.to(str(DEVICE))
ELECS = pac2.electrodes(vent.scale)
upstroke, _ = upstroke_for_spacing(grid, CV_T)
grad_Z = torch.einsum('ab,bed->aed', einthoven_wilson(torch.eye(len(ELECS))), lead_field(grid, ELECS))
t_grid = np.linspace(0.0, scale * 1.05, NT)
ecg = ECGForward(grid, grad_Z, t_grid, upstroke=upstroke, device=str(DEVICE))
with torch.no_grad():
    V_true = ecg.from_tau(tau_t)
print(f'{grid.N} nodes; design {DESIGN}, S {len(S)} nodes; patches {[len(p) for p in patches]} '
      f'({int(np.isin(src, S).sum())} of {len(src)} source nodes in S); truth max {scale:.1f} ms; eps {EPS}',
      flush=True)


def phi_of(g):
    p = model.field_from_g(g, *model.cv)
    t = op.potential(p, *model.cv, n_project=0)
    return t - t.min()


def score(tag, ph_t, u=None):
    with torch.no_grad():
        a = accuracy(ph_t, tau_t)
        L = float(((ph_t - tau_t) ** 2).mean() / scale ** 2)
        V = ecg.from_tau(ph_t)
        e = float(((V - V_true) ** 2).mean() / (V_true ** 2).mean())
    ph = ph_t.cpu().numpy().astype(np.float64)
    ug = sp.metric_norm(sp.gradient_elem(ph))
    _, nb = basin_persistence(ph, nb_idx, nb_msk)
    tc, pc = tau - tau.mean(), ph - ph.mean()
    row = {'lat_loss': L, 'lat_mae': float(np.abs(pc - tc).mean()),
           'lat_rms': float(np.sqrt(((pc - tc) ** 2).mean())), 'corr': a['corr'],
           'eik': eikonal_rms(ug, sp.meas), 'basins': int(nb),
           'src_spread': float(np.ptp(ph[src])), 'ecg': e,
           'neg_frac': None if u is None else float((u <= 0).float().mean())}
    print(f"[{tag}] LAT loss {L:.3e}  MAE {row['lat_mae']:.3f} ms  rms {row['lat_rms']:.3f} ms  corr {row['corr']:.5f}  "
          f"eik {row['eik']:.3f}  basins {row['basins']}  spread {row['src_spread']:.2f} ms  ECG {e:.3e}"
          + ('' if u is None else f"  u<=0 {100 * row['neg_frac']:.1f}%"), flush=True)
    return row


truth = score('truth', tau_t)
g0 = tau_t[model.s_nodes].clone()
with torch.no_grad():
    row0 = score('g0 = tau* on S', phi_of(g0), model.solve_u(g0, *model.cv))
g = torch.nn.Parameter(g0.clone())
opt = torch.optim.Adam([g], lr=LR_G)
hist = []
t0 = time.time()
for it in range(N_FIT + 1):
    opt.zero_grad(set_to_none=True)
    L = ((phi_of(g) - tau_t) ** 2).mean() / scale ** 2
    hist.append(float(L.detach()))
    if it < N_FIT:
        L.backward()
        opt.step()
    if it % (N_FIT // 6) == 0:
        print(f'  [fit] it {it:5d}  loss {hist[-1]:.3e}', flush=True)
print(f'  [fit] {time.time() - t0:.0f} s', flush=True)
gs = g.detach()
with torch.no_grad():
    row1 = score('g* (LAT fit)', phi_of(gs), model.solve_u(gs, *model.cv))
np.savez_compressed(os.path.join(RUN, 'gstar_vent.npz'), S=S, src=src, g0=g0.cpu().numpy(),
                    gstar=gs.cpu().numpy(), tau=tau, hist=np.array(hist))
json.dump({'setup': {'design': DESIGN, 'eps': EPS, 'n_fit': N_FIT, 'lr_g': LR_G, 'cv': [CV_L, CV_T],
                     'mass': 'lumped', 'n_s': int(len(S))},
           'truth': truth, 'g0': row0, 'gstar': row1}, open(os.path.join(RUN, 'summary.json'), 'w'), indent=1)
print('wrote', RUN)
