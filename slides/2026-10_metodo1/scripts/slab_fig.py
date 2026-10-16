"""Figura da laje (caso paredes) para o deck: verdade, phi(g0), phi(g*), erros, perda, g nas paredes.

    python scripts/slab_fig.py data/slab/gstar_slab.npz figures/laje_gstar.png
(entrada: tools/camps_gstar/fit_g_slab.py, MESH 49,25,5, EPS 2)

    python plot_slab_gstar.py results/slab/gstar_slab.npz results/slab/slab_walls_gstar.png
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

z = np.load(sys.argv[1]); out = sys.argv[2]
v, tau = z['verts'], z['tau']
p0, ps = z['phi0'], z['phis']
S, g0, gs = z['S'], z['g0'], z['gstar']
INK2, GRID, BLUE, ORANGE = '#52514e', '#e6e5e1', '#2a78d6', '#eb6834'
plt.rcParams.update({'font.size': 12, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2})
zm = np.unique(v[:, 2])[len(np.unique(v[:, 2])) // 2]
mid = np.isclose(v[:, 2], zm)
xs, ys = np.unique(v[mid, 0]), np.unique(v[mid, 1])


def plane(f):
    a = np.full((len(ys), len(xs)), np.nan)
    a[np.searchsorted(ys, v[mid, 1]), np.searchsorted(xs, v[mid, 0])] = f[mid]
    return a


SRC_C, NSRC_C = '#e8a200', '#7a3db8'


def mark_s(ax, legend=False):
    """S on the mid-height plane: wall y = 0 (true source) and wall x = 0 (in S, not a source)."""
    ax.plot([xs[0], xs[-1]], [ys[0], ys[0]], color=SRC_C, lw=5, clip_on=False, solid_capstyle='butt',
            label='S, parede y = 0 (fonte verdadeira)')
    ax.plot([xs[0], xs[0]], [ys[0], ys[-1]], color=NSRC_C, lw=5, clip_on=False, solid_capstyle='butt',
            label='S, parede x = 0 (não é fonte)')
    if legend:
        ax.legend(frameon=True, fontsize=11, loc='upper right', framealpha=0.9)


ext = (xs[0], xs[-1], ys[0], ys[-1])
cen = lambda f: (f - f.mean()) - (tau - tau.mean())
e0, es = cen(p0), cen(ps)
mae0, maes = np.abs(e0).mean(), np.abs(es).mean()
fig = plt.figure(figsize=(18, 10.5))
gs_ = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.95], width_ratios=[1, 1, 1, 0.05], hspace=0.85, wspace=0.32)
row0, row1 = [], []
for j, (f, ttl) in enumerate(((tau, 'verdade $\\tau^*$'),
                              (p0, '$\\varphi(g_0)$, $g_0 = \\tau^*$ em S'),
                              (ps, '$\\varphi(g^*)$, $g$ ótimo para o LAT'))):
    ax = fig.add_subplot(gs_[0, j]); row0.append(ax)
    im = ax.imshow(plane(f), origin='lower', extent=ext, cmap='Blues', vmin=0, vmax=tau.max(), aspect='equal')
    mark_s(ax, legend=(j == 0))
    ax.set_title(ttl, loc='left', fontsize=12.5); ax.set_xlabel('x'); ax.set_ylabel('y')
fig.colorbar(im, cax=fig.add_subplot(gs_[0, 3]), label='tempo de ativação (ms)')
lim = max(np.abs(e0).max(), np.abs(es).max())
for j, (e, ttl) in enumerate(((e0, f'erro, $g_0$: MAE {mae0:.3f} ms, máx {np.abs(e0).max():.2f} ms'.replace('.', ',')),
                              (es, f'erro, $g^*$: MAE {maes:.3f} ms, máx {np.abs(es).max():.2f} ms'.replace('.', ',')))):
    ax = fig.add_subplot(gs_[1, j]); row1.append(ax)
    im2 = ax.imshow(plane(e), origin='lower', extent=ext, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='equal')
    mark_s(ax)
    ax.set_title(ttl, loc='left', fontsize=12.5); ax.set_xlabel('x'); ax.set_ylabel('y')
cbx = row1[0].inset_axes([0.0, -0.36, 2.25, 0.06]); fig.colorbar(im2, cax=cbx, orientation='horizontal', label='modelo - verdade (ms, médias removidas); vermelho = tarde, azul = cedo')
ax = fig.add_subplot(gs_[1, 2])
ax.semilogy(z['hist'], color=BLUE, lw=1.5)
ax.set_title('erro de LAT no ajuste de g', loc='left', fontsize=12.5); ax.set_xlabel('iteração')
ax.set_ylabel('erro de LAT (log)'); ax.grid(color=GRID, lw=.6)
vs = v[S]
g0n, gsn = g0 - g0.min(), gs - gs.min()
for j, (sel, xax, lab) in enumerate(((np.isclose(vs[:, 1], 0) & np.isclose(vs[:, 2], zm), 0, 'parede y = 0 (fonte), ao longo de x'),
                                     (np.isclose(vs[:, 0], 0) & np.isclose(vs[:, 2], zm) & (vs[:, 1] > 0), 1, 'parede x = 0 (não é fonte), ao longo de y'))):
    ax = fig.add_subplot(gs_[2, j])
    o = np.argsort(vs[sel, xax])
    ax.plot(vs[sel, xax][o], g0n[sel][o], '-', color=ORANGE, lw=2, label='$g_0 = \\tau^*$')
    ax.plot(vs[sel, xax][o], gsn[sel][o], '-', color=BLUE, lw=2, label='$g^*$')
    ax.set_title(f'g na {lab}', loc='left', fontsize=12.5, color=(SRC_C if j == 0 else NSRC_C))
    ax.set_xlabel('x' if xax == 0 else 'y'); ax.set_ylabel('g (ms, mín 0)'); ax.grid(color=GRID, lw=.6)
    ax.legend(frameon=False)
fig.savefig(out, dpi=140, bbox_inches='tight')
# para o deck: duas metades (mapas e erros; g nas paredes), recortadas na fronteira entre as linhas
from PIL import Image
r = fig.canvas.get_renderer()
bb_top = fig.axes[5].get_tightbbox(r)          # painel da perda (linha 2)
import matplotlib.transforms as mt
fig.savefig(out.replace('.png', '_mapas.png'), dpi=140,
            bbox_inches=mt.Bbox.union([a.get_tightbbox(r) for a in fig.axes[:7]]).transformed(fig.dpi_scale_trans.inverted()))
fig.savefig(out.replace('.png', '_g.png'), dpi=140,
            bbox_inches=mt.Bbox.union([a.get_tightbbox(r) for a in fig.axes[7:]]).transformed(fig.dpi_scale_trans.inverted()))
print('wrote', out)
