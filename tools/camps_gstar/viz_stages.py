"""Figures at each step of: train net on LAT -> ECG inverse (reads stages.npz, summary.json).

    python viz_stages.py DIR OUT      # DIR holds stages.npz and stages_summary.json
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
z = np.load(os.path.join(D, 'stages.npz'))
sm = json.load(open(os.path.join(D, 'stages_summary.json')))
names = [str(n) for n in z['names']]
xyz, surf, roots, S = z['xyz'], z['surf'], z['roots'], z['s_nodes']
tau_true = z['tau_true']; V_true = z['V_true']; t = z['t_grid']
tau = {n: z[f'tau_{i}'] for i, n in enumerate(names)}
V = {n: z[f'V_{i}'] for i, n in enumerate(names)}
g = {n: z[f'g_{i}'] for i, n in enumerate(names)}
rows = {'g0 = truth on S': sm['g0'], 'random-init network (before LAT training)': sm['init'],
        'network after LAT training': sm['lat_net']}
for r in sm['after_ecg']:
    rows[f"after ECG inverse, Adam {r['lr']:g}"] = r
SHORT = {'g0 = truth on S': '1. g0 = truth on S', 'random-init network (before LAT training)': '2. random-init net',
         'network after LAT training': '3. net after LAT training'}
for r in sm['after_ecg']:
    SHORT[f"after ECG inverse, Adam {r['lr']:g}"] = f"4. + ECG inverse, Adam {r['lr']:g}"
INK, INK2, SURF, GRID = '#0b0b0b', '#52514e', '#fcfcfb', '#e6e5e1'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
                     'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.edgecolor': GRID, 'font.size': 9})
P = xyz[surf]
VIEWS = [(0, 2, 1, 'x-z'), (1, 2, 0, 'y-z')]          # horizontal axis, vertical axis, depth axis


def draw(ax, field, vi, cmap, vmin, vmax):
    h, v, d, _ = VIEWS[vi]
    order = np.argsort(P[:, d])
    sc = ax.scatter(P[order, h], P[order, v], c=field[surf][order], s=3.5, cmap=cmap, vmin=vmin, vmax=vmax,
                    linewidths=0, rasterized=True)
    ax.scatter(xyz[roots, h], xyz[roots, v], s=22, facecolors='none', edgecolors=INK, linewidths=1.0)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    return sc


def grid(fields, titles, cmap, vmin, vmax, fn, label, sup):
    n = len(fields)
    fig, axs = plt.subplots(2, n, figsize=(2.35 * n, 5.6))
    for j, (f, ttl) in enumerate(zip(fields, titles)):
        for vi in range(2):
            sc = draw(axs[vi, j], f, vi, cmap, vmin, vmax)
        axs[0, j].set_title(ttl, fontsize=8.5, loc='left')
    cb = fig.colorbar(sc, ax=axs, orientation='horizontal', fraction=0.04, pad=0.03, aspect=50)
    cb.set_label(label)
    fig.suptitle(sup, x=0.01, ha='left', fontsize=11)
    fig.savefig(os.path.join(OUT, fn), dpi=150, bbox_inches='tight'); plt.close(fig)


def mt(n):
    r = rows[n]
    return f"{SHORT[n]}\nLAT MAE {r['mae']:.1f} ms · ECG {r['ecg']:.1e}"


order = names
grid([tau_true] + [tau[n] for n in order], ['0. FIM truth\n(target)'] + [mt(n) for n in order], 'Blues', 0, tau_true.max(),
     'fig6_lat_maps.png', 'activation time (ms)', 'Activation (LAT) map at each step, two views; rings = the 5 true roots')
grid([tau[n] - tau[n].min() - tau_true for n in order], [mt(n) for n in order], 'RdBu_r', -15, 15,
     'fig7_lat_error.png', 'LAT error vs truth (ms): red = model late, blue = early', 'LAT error map at each step')

# ECG leads
LEADS = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
fig, axs = plt.subplots(2, 4, figsize=(11, 4.8), sharex=True)
show = [('g0 = truth on S', BLUE, '1. g0 = truth on S'), ('network after LAT training', ORANGE, '3. net after LAT training'),
        ('after ECG inverse, Adam 0.0001', AQUA, '4. + ECG inverse (Adam 1e-4)')]
for k, ax in enumerate(axs.ravel()):
    ax.plot(t, V_true[k], color=INK, lw=2.2, label='truth ECG')
    for n, c, lab in show:
        ax.plot(t, V[n][k], color=c, lw=1.4, label=lab)
    ax.set_title(LEADS[k], loc='left', fontsize=9); ax.grid(color=GRID, lw=.6)
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
axs[1, 0].set_xlabel('time (ms)'); axs[0, 0].set_ylabel('potential (a.u.)')
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle('ECG leads: truth vs model at steps 1, 3 and 4 (random init omitted, off scale)', x=0.01, ha='left', fontsize=11)
fig.tight_layout(rect=(0, 0.04, 1, 0.96)); fig.savefig(os.path.join(OUT, 'fig8_ecg_leads.png'), dpi=150); plt.close(fig)

# g on S
gt = z['g_true'] - z['g_true'].min(); isr = np.isin(S, roots)
fig, axs = plt.subplots(2, 3, figsize=(10.5, 6.2), sharex=True, sharey=True)
for ax, n in zip(axs.ravel(), order):
    gg = g[n] - g[n].min()
    ax.plot([0, 60], [0, 60], color=GRID, lw=1.2); ax.set_xlim(-1, 60); ax.set_ylim(-1, 60)
    ax.scatter(gt[~isr], gg[~isr], s=7, color='#8a8985', alpha=.45, lw=0)
    ax.scatter(gt[isr], gg[isr], s=40, color=BLUE, ec=SURF, lw=1.0, zorder=3)
    ax.set_title(SHORT[n], loc='left', fontsize=9); ax.grid(color=GRID, lw=.6)
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
for ax in axs[1]: ax.set_xlabel('g0 = truth on S (ms, minus min)')
for ax in axs[:, 0]: ax.set_ylabel('model g on S (ms, minus min)')
fig.suptitle('Source times g on the 601 candidates vs truth, zoomed to 0-60 ms (larger g has no effect: weight exp(-g/2 ms)); blue = true roots; line = identity', x=0.01, ha='left', fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(os.path.join(OUT, 'fig9_g_on_S.png'), dpi=150); plt.close(fig)
print('wrote', os.listdir(OUT))
