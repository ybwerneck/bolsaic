"""Figuras do teste direto nos ventrículos (sem treino): g0 = tau* em S contra o g* ótimo para o LAT.

    python scripts/vent_fig.py        # do diretório do deck; lê data/vent/fields_vent_{recoverable,chamber}.npz
(entrada: tools/camps_gstar/fit_g_vent.py + eval_g_vent.py, eps 2 ms)
Saídas: figures/vent_mapas.png, figures/vent_ecg.png.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e6e5e1'
BLUE, ORANGE, SCOL, SRC = '#2a78d6', '#eb6834', '#7a3db8', '#e8a200'
plt.rcParams.update({'font.size': 12, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.edgecolor': GRID})
D = {k: np.load(os.path.join(HERE, 'data', 'vent', f'fields_vent_{k}.npz')) for k in ('recoverable', 'chamber')}
NOME = {'recoverable': 'recuperável\n($S$ = endocárdios)', 'chamber': 'câmara\n($S$ = lvendo)'}
VIEW = (0, 2, 1)                                   # eixo horizontal, vertical, profundidade


def scat(ax, z, field, cmap, vmin, vmax):
    xyz, surf = z['xyz'], z['surf']
    P = xyz[surf]; h, v, d = VIEW
    o = np.argsort(P[:, d])
    sc = ax.scatter(P[o, h], P[o, v], c=field[surf][o], s=4, cmap=cmap, vmin=vmin, vmax=vmax, lw=0, rasterized=True)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    return sc


def mae(ph, tau):
    return np.abs((ph - ph.mean()) - (tau - tau.mean())).mean()


fig, axs = plt.subplots(2, 6, figsize=(21, 8.2))
tmax = max(z['tau'].max() for z in D.values())
for i, (k, z) in enumerate(D.items()):
    xyz, tau, S, src = z['xyz'], z['tau'], z['S'], z['src']
    h, v, _ = VIEW
    ax = axs[i, 0]
    scat(ax, z, np.full(len(xyz), 1.0), 'Greys', 0, 5)
    ax.scatter(xyz[S, h], xyz[S, v], s=2.5, color=SCOL, lw=0, alpha=.55, rasterized=True, label='$S$')
    ax.scatter(xyz[src, h], xyz[src, v], s=60, marker='*', color=SRC, ec=INK, lw=.5, zorder=4, label='fontes verdadeiras')
    ax.set_title('$S$ e fontes verdadeiras', loc='left', fontsize=12)
    if i == 0:
        ax.legend(loc='lower left', fontsize=10, markerscale=2.5, frameon=False, bbox_to_anchor=(0, -0.22))
    ax.text(-0.06, 0.5, NOME[k], transform=ax.transAxes, rotation=90, va='center', ha='right', fontsize=13, fontweight='bold')
    e0, es = z['phi_g0'] - z['phi_g0'].mean() - (tau - tau.mean()), z['phi_gstar'] - z['phi_gstar'].mean() - (tau - tau.mean())
    lim = float(np.percentile(np.abs(np.concatenate([e0, es])), 99))
    for j, (f, ttl) in enumerate(((tau, 'verdade $\\tau^*$'),
                                  (z['phi_g0'], f'$\\varphi(g_0)$, $g_0 = \\tau^*$ em $S$\nMAE {mae(z["phi_g0"], tau):.2f} ms'),
                                  (z['phi_gstar'], f'$\\varphi(g^*)$, ajuste ao LAT\nMAE {mae(z["phi_gstar"], tau):.2f} ms'))):
        sc = scat(axs[i, 1 + j], z, f, 'Blues', 0, tmax)
        axs[i, 1 + j].set_title(ttl.replace('.', ','), loc='left', fontsize=12)
    for j, (e, ttl) in enumerate(((e0, 'erro, $g_0$'), (es, 'erro, $g^*$'))):
        sc2 = scat(axs[i, 4 + j], z, e, 'RdBu_r', -lim, lim)
        axs[i, 4 + j].set_title(ttl, loc='left', fontsize=12)
    cb = fig.colorbar(sc2, ax=axs[i, 4:6], fraction=0.04, pad=0.02)
    cb.set_label(f'modelo - verdade (ms)\nvermelho = tarde')
cbm = fig.colorbar(sc, ax=axs[:, 1:4], orientation='horizontal', fraction=0.035, pad=0.04, aspect=50)
cbm.set_label('tempo de ativação (ms)')
fig.savefig(os.path.join(HERE, 'figures', 'vent_mapas.png'), dpi=140, bbox_inches='tight'); plt.close(fig)

LEADS = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
plt.rcParams.update({'font.size': 14})
fig, axs = plt.subplots(2, 8, figsize=(17, 6.2), sharex=True)
for i, (k, z) in enumerate(D.items()):
    t = z['t_grid']
    e = lambda V: ((V - z['V_true']) ** 2).mean() / (z['V_true'] ** 2).mean()
    for j in range(8):
        ax = axs[i, j]
        ax.plot(t, z['V_true'][j], color=INK, lw=2.4, label='verdade')
        ax.plot(t, z['V_g0'][j], color=ORANGE, lw=1.8, label='$g_0 = \\tau^*$ em $S$')
        ax.plot(t, z['V_gstar'][j], color=BLUE, lw=1.8, label='$g^*$ (ajuste ao LAT)')
        ax.set_title(LEADS[j], loc='left', fontsize=14); ax.grid(color=GRID, lw=.6)
        ax.set_yticks([])
        for s_ in ('top', 'right', 'left'):
            ax.spines[s_].set_visible(False)
        if i == 1:
            ax.set_xlabel('ms')
    axs[i, 0].set_ylabel(NOME[k].split('\n')[0], fontsize=16, fontweight='bold')
    print(k, 'ECG g0 %.3f g* %.3f' % (e(z['V_g0']), e(z['V_gstar'])))
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(os.path.join(HERE, 'figures', 'vent_ecg.png'), dpi=140, bbox_inches='tight'); plt.close(fig)
print('ok')
