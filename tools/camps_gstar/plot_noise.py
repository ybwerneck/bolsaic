"""Plots/tables of the noise sweep: one line per optimizer, colour = start A (orange) or B (blue).

    python plot_noise.py noise_A.json noise_B.json OUT_DIR
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fa, fb, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
R = json.load(open(fa)) + json.load(open(fb))
SIG = sorted({r['sigma'] for r in R})
OPTS = []
for r in R:
    k = (r['opt'], r['lr'])
    if k not in OPTS: OPTS.append(k)
OPTS.sort(key=lambda k: (k[0] != 'adam', k[1]))
COL = {'A': ['#f6a47c', '#ef8450', '#eb6834', '#c24a1c', '#7a2a0a'],
       'B': ['#8ab8ea', '#5a97e0', '#2a78d6', '#1c56a0', '#0e3066']}
MK = ['o', 's', '^', 'D', 'X']
INK, INK2, SURF, GRID = '#0b0b0b', '#52514e', '#fcfcfb', '#e6e5e1'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'text.color': INK,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': GRID,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': .6, 'font.size': 9})


def med(tag, opt, sig, phase, key):
    v = [r[phase][key] for r in R if r['tag'] == tag and (r['opt'], r['lr']) == opt and r['sigma'] == sig]
    v = [x for x in v if np.isfinite(x)]
    return float(np.median(v)) if v else np.nan


x = np.arange(len(SIG))
fig, axs = plt.subplots(2, 2, figsize=(12, 8.2))
panels = [(axs[0, 0], 'ecg', 'ECG error after the ECG inverse (log)', True),
          (axs[0, 1], 'mae', 'LAT MAE after the ECG inverse (ms, log)', True),
          (axs[1, 0], 'corr', 'LAT corr after the ECG inverse', False)]
for ax, key, yl, lg in panels:
    for tag in ('A', 'B'):
        for i, opt in enumerate(OPTS):
            y = [med(tag, opt, s, 'end', key) for s in SIG]
            ax.plot(x, y, '-', color=COL[tag][i], marker=MK[i], ms=6, lw=1.8, mec=SURF, mew=1.0)
    ax.set_xticks(x); ax.set_xticklabels([f'{s:g}' for s in SIG])
    ax.set_xlabel('noise on the start: sigma (x per-tensor weight std)'); ax.set_ylabel(yl)
    if lg: ax.set_yscale('log')
ax = axs[1, 1]
for tag, c in (('A', COL['A'][2]), ('B', COL['B'][2])):
    ax.plot(x, [np.median([r['start']['dg_dom'] for r in R if r['tag'] == tag and r['sigma'] == s]) for s in SIG],
            '-o', color=c, lw=1.8, ms=6, mec=SURF, mew=1.0, label=f'start {tag}')
ax.set_xticks(x); ax.set_xticklabels([f'{s:g}' for s in SIG]); ax.set_xlabel('noise sigma')
ax.set_ylabel('how far the noise moves g (rms, dominant candidates, ms)'); ax.set_yscale('symlog', linthresh=0.5)
ax.legend(frameon=False)
ax.set_title('the noise in ms', loc='left', fontsize=10)
# legend: two groups
from matplotlib.lines import Line2D
h = [Line2D([], [], color=COL[t][i], marker=MK[i], lw=1.8, ms=6, mec=SURF, label=f'{t}: {o} {lr:g}')
     for t in ('A', 'B') for i, (o, lr) in enumerate(OPTS)]
fig.legend(handles=h, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=8.5)
fig.suptitle('Effect of noise on the starting point, per optimizer. A = start from the LAT-trained net (orange), B = start from the net fitted to g* (blue). Median of draws.',
             x=0.01, ha='left', fontsize=10.5)
fig.tight_layout(rect=(0, 0.07, 1, 0.96))
fig.savefig(os.path.join(OUT, 'fig10_noise_sweep.png'), dpi=150); plt.close(fig)

md = ['# Noise sweep (truth s1): median of draws, end of the ECG inverse\n', '| start | optimizer | ' + ' | '.join(f'sigma {s:g}' for s in SIG) + ' |', '|---|---|' + '---|' * len(SIG)]
for key, nm, fmt in (('ecg', 'ECG error', '{:.1e}'), ('mae', 'LAT MAE (ms)', '{:.2f}'), ('corr', 'LAT corr', '{:.4f}')):
    md.append(f'| **{nm}** | | ' + ' | '.join('' for _ in SIG) + ' |')
    for tag in ('A', 'B'):
        for opt in OPTS:
            md.append(f'| {tag} | {opt[0]} {opt[1]:g} | ' + ' | '.join(fmt.format(med(tag, opt, s, 'end', key)) for s in SIG) + ' |')
open(os.path.join(OUT, 'noise_table.md'), 'w').write('\n'.join(md) + '\n')
print('wrote', os.listdir(OUT))
