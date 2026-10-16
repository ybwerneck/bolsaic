"""Evaluate fit_g_vent results: phi and ECG for g0 and g*, saved for plotting.

OLD DOC of fit_g_vent:

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
RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', f'vent_gstar_{DESIGN}_e{EPS:g}')

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


z = np.load(os.path.join(RUN, 'gstar_vent.npz'))
assert np.array_equal(z['S'], S)
out = {'xyz': np.asarray(verts), 'surf': np.unique(surf), 'S': S, 'src': src, 'tau': tau,
       'V_true': V_true.cpu().numpy(), 't_grid': t_grid}
with torch.no_grad():
    for k in ('g0', 'gstar'):
        g = torch.as_tensor(z[k], dtype=torch.float32, device=DEVICE)
        ph = phi_of(g)
        out[f'phi_{k}'] = ph.cpu().numpy()
        out[f'V_{k}'] = ecg.from_tau(ph).cpu().numpy()
        out[k] = z[k]
np.savez_compressed(os.path.join(RUN, 'fields_vent.npz'), **out)
print('wrote', os.path.join(RUN, 'fields_vent.npz'))
