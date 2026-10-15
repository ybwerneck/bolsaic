"""Slab 'walls' case: Method 1 with g0 = tau* on S against the LAT-optimal g*.

Setup as in ``problems/screened_poisson/slab_cases.py``, case 'walls': slab
2 x 1 x 0.15, fibre from a Laplace solve between the x walls, cv (0.052,
0.017); S = the walls y = 0 and x = 0; active sources = every node of y = 0
(a plane wave), truth tau* = closed-form arrival from them. Method 1 as in
training (``core.screened_front.ScreenedFront``, lumped mass, sparse solve,
then ``DeltaPoisson.potential``). Nothing is trained but g:

    g0  = tau* on S (0 on y = 0)
    g*  = free g on S minimising mean((phi - min phi - tau*)^2) / max(tau*)^2,
          started at g0, Adam with constant step LR_G for N_FIT steps

Scores for both (as slab_cases): LAT MAE and rms (means removed), corr,
eikonal residual (measure-weighted RMS of ||U grad phi|| - 1), basins, spread
of phi over the active nodes, fraction of u <= 0; and how g* moved per wall.

Environment: MESH (e.g. 49,25,5), EPS (ms), N_FIT, LR_G, RUN_OUT.
"""
import json
import os
import time

import numpy as np
import torch

from fisiocomPinn import Grid3D

from core.diagnostics import accuracy, basin_persistence, eikonal_rms
from core.fiber import arrival_floor, poisson_fiber
from core.mesh import one_hop, slab_mesh, surface_faces
from core.metric import TransverseIsotropic
from core.operator import DeltaPoisson
from core.output import open_run
from core.screened import ScreenedPoisson
from core.screened_front import ScreenedFront

CV_L, CV_T = 0.052, 0.017
DIMS = (2.0, 1.0, 0.15)
MESH = tuple(int(x) for x in os.environ.get('MESH', '49,25,5').split(','))
EPS = float(os.environ.get('EPS', 2.0))
N_FIT = int(os.environ.get('N_FIT', 3000))
LR_G = float(os.environ.get('LR_G', 0.1))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RUN = open_run(__file__, f'slab_walls_gstar_{MESH[0]}x{MESH[1]}x{MESH[2]}_e{EPS:g}',
               header={'MESH': MESH, 'EPS': EPS, 'N_FIT': N_FIT, 'LR_G': LR_G})

verts, tets = slab_mesh(*MESH, DIMS)
grid = Grid3D(verts, tets)
on = lambda axis, val: np.isclose(verts[:, axis], val)
_, f_elem = poisson_fiber(grid, np.flatnonzero(on(0, 0.0)), np.flatnonzero(on(0, DIMS[0])))
f = f_elem.mean(axis=0)
f = f / np.linalg.norm(f)
D = CV_T ** 2 * np.eye(3) + (CV_L ** 2 - CV_T ** 2) * np.outer(f, f)
walls = np.flatnonzero(on(1, 0.0) | on(0, 0.0))
in_s = on(1, 0.0) | on(0, 0.0)
act = np.flatnonzero(on(1, 0.0))
tau = arrival_floor(verts, verts[act], D)
tau = tau - tau.min()
scale = float(tau.max())
S = np.flatnonzero(in_s)
surf = surface_faces(tets)
sp = ScreenedPoisson(grid, f_elem, CV_L, CV_T, lumped=True)
a_s = (sp.surface_weights(walls, surf) * in_s)[S]
D_elem = np.einsum('ij,e->eij', D, np.ones(len(tets)))
fe = np.einsum('ej,ek->ejk', f_elem, f_elem)
D_elem = CV_T ** 2 * np.eye(3)[None] + (CV_L ** 2 - CV_T ** 2) * fe
model = ScreenedFront(grid, f_elem, S, a_s, EPS, g_scale=scale,
                      cv=[(CV_L, CV_L, CV_L), (CV_T, CV_T, CV_T)], learn=False,
                      solver='sparse', screen_D=D_elem).to(DEVICE)
metric = TransverseIsotropic(torch.as_tensor(f_elem, dtype=torch.float32))
op = DeltaPoisson(grid, metric=metric, sparse=str(DEVICE)).to(str(DEVICE))
metric.to(str(DEVICE))
tau_t = torch.as_tensor(tau, dtype=torch.float32, device=DEVICE)
nb_idx, nb_msk = one_hop(grid)
pos = np.searchsorted(S, act)                          # active nodes inside S
print(f'mesh {MESH}: {grid.N} nodes; S {len(S)} (y=0: {len(act)}, x=0 only: '
      f'{len(S) - len(act)}); truth max {scale:.1f} ms; eps {EPS}', flush=True)


def phi_of(g):
    p = model.field_from_g(g, *model.cv)
    t = op.potential(p, *model.cv, n_project=0)
    return t - t.min()


def lat_loss(g):
    return ((phi_of(g) - tau_t) ** 2).mean() / scale ** 2


def score(tag, g):
    with torch.no_grad():
        ph_t = phi_of(g)
        u = model.solve_u(g, *model.cv)
        a = accuracy(ph_t, tau_t)
        L = float(((ph_t - tau_t) ** 2).mean() / scale ** 2)
    ph = ph_t.cpu().numpy().astype(np.float64)
    ug = sp.metric_norm(sp.gradient_elem(ph))
    _, nb = basin_persistence(ph, nb_idx, nb_msk)
    tc, pc = tau - tau.mean(), ph - ph.mean()
    row = {'lat_loss': L, 'lat_mae': float(np.abs(pc - tc).mean()),
           'lat_rms': float(np.sqrt(((pc - tc) ** 2).mean())), 'corr': a['corr'],
           'eik': eikonal_rms(ug, sp.meas), 'basins': int(nb),
           'src_spread': float(np.ptp(ph[act])),
           'neg_frac': float((u <= 0).float().mean())}
    print(f"[{tag}] LAT loss {L:.3e}  MAE {row['lat_mae']:.3f} ms  rms {row['lat_rms']:.3f} ms  "
          f"corr {row['corr']:.5f}  eik {row['eik']:.3f}  basins {row['basins']}  "
          f"spread {row['src_spread']:.2f} ms  u<=0 {100 * row['neg_frac']:.1f}%", flush=True)
    return row


truth_ug = sp.metric_norm(sp.gradient_elem(tau))
_, truth_nb = basin_persistence(tau, nb_idx, nb_msk)
truth = {'eik': eikonal_rms(truth_ug, sp.meas), 'basins': int(truth_nb), 'tau_max': scale}
print(f"[truth] eik {truth['eik']:.3f}  basins {truth['basins']}", flush=True)

g0 = tau_t[model.s_nodes].clone()
row0 = score('g0 = tau* on S', g0)
g = torch.nn.Parameter(g0.clone())
opt = torch.optim.Adam([g], lr=LR_G)
hist = []
t0 = time.time()
for it in range(N_FIT + 1):
    opt.zero_grad(set_to_none=True)
    L = lat_loss(g)
    hist.append(float(L.detach()))
    if it < N_FIT:
        L.backward()
        opt.step()
    if it % (N_FIT // 6) == 0:
        print(f'  [fit] it {it:5d}  loss {hist[-1]:.3e}', flush=True)
print(f'  [fit] {time.time() - t0:.0f} s', flush=True)
gs = g.detach()
row1 = score('g* (LAT fit)', gs)
dg = (gs - gs.min()) - (g0 - g0.min())
dg = dg.cpu().numpy()
on_y0 = np.isin(S, act)
moved = {'y0_mean': float(dg[on_y0].mean()), 'y0_absmax': float(np.abs(dg[on_y0]).max()),
         'x0_mean': float(dg[~on_y0].mean()), 'x0_absmax': float(np.abs(dg[~on_y0]).max())}
print(f"g* - g0 (both min-shifted): wall y=0 mean {moved['y0_mean']:+.2f}, max |.| "
      f"{moved['y0_absmax']:.2f} ms; wall x=0 mean {moved['x0_mean']:+.2f}, max |.| "
      f"{moved['x0_absmax']:.2f} ms", flush=True)
np.savez_compressed(os.path.join(RUN, 'gstar_slab.npz'), S=S, act=act, g0=g0.cpu().numpy(),
                    gstar=gs.cpu().numpy(), tau=tau, verts=verts, hist=np.array(hist),
                    phi0=phi_of(g0).detach().cpu().numpy(), phis=phi_of(gs).detach().cpu().numpy())
json.dump({'setup': {'mesh': MESH, 'eps': EPS, 'n_fit': N_FIT, 'lr_g': LR_G, 'cv': [CV_L, CV_T],
                     'mass': 'lumped', 'case': 'walls'},
           'truth': truth, 'g0': row0, 'gstar': row1, 'moved': moved},
          open(os.path.join(RUN, 'summary.json'), 'w'), indent=1)
print('wrote', RUN)
