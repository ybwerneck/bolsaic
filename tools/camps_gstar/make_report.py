"""Tables and plots for the g* runs (train_gstar).

    python make_report.py  DATA_DIR   # DATA_DIR holds g0.npz..g5.npz
Reads results/summary_data.json (summary.json of the six runs) and writes
tables.md, tables.csv and fig1..fig4 PNGs next to it.
"""
import csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results')
DATA = sys.argv[1]
D = {int(k): v for k, v in json.load(open(os.path.join(OUT, 'summary_data.json'))).items()}
# old Method 1, random init, 100 networks, medians (camps_dti004/outputs/set1/SUMMARY.md)
OLD = {0: (0.782, 12.2, 0.031), 1: (0.681, 14.2, 0.017), 2: (0.801, 14.5, 0.022),
       3: (0.523, 15.3, 0.073), 4: (0.895, 11.9, 0.008), 5: (0.577, 14.9, 0.022)}
STAGES = [('g0_truth_on_S', 'g0 = truth|S'), ('gstar', 'g* (LAT fit)'), ('final', 'g* + ECG inverse')]
BLUE, ORANGE, AQUA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#8a8985'
INK, INK2, SURF, GRID = '#0b0b0b', '#52514e', '#fcfcfb', '#e6e5e1'
COL = [BLUE, ORANGE, AQUA]
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
                     'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'text.color': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6})
T = list(range(6))
xl = [f's{k}' for k in T]

# ---- tables ----------------------------------------------------------------
rows = []
for k in T:
    r = {'truth': f'truth5ti_s{k}'}
    for key, lab in STAGES:
        r[f'ecg {lab}'] = D[k][key]['ecg']; r[f'corr {lab}'] = D[k][key]['corr']
        r[f'MAE {lab}'] = D[k][key]['mae']
    r['ecg random-init'] = OLD[k][2]; r['corr random-init'] = OLD[k][0]; r['MAE random-init'] = OLD[k][1]
    r['ranks final'] = str(D[k]['final']['root_rank']); r['sources final'] = D[k]['final']['n_sources']
    r['regression max err (ms)'] = D[k]['reg_err']
    rows.append(r)
with open(os.path.join(OUT, 'tables.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, rows[0].keys()); w.writeheader(); w.writerows(rows)

def med(col): return float(np.median([r[col] for r in rows]))
md = ['# g* runs: tables\n', 'One network seed (0) per truth; 1000 ECG iterations; frozen H, speeds boxed at truth +/-1%.',
      'Random-init column: old Method 1, median of 100 networks (`outputs/set1/SUMMARY.md`).\n']
for title, m, fmt in (('ECG error (relative)', 'ecg', '{:.2e}'), ('LAT correlation with truth', 'corr', '{:.4f}'),
                      ('LAT MAE (ms)', 'MAE', '{:.2f}')):
    labs = [lab for _, lab in STAGES] + ['random-init']
    md += [f'## {title}\n', '| truth | ' + ' | '.join(labs) + ' |', '|---|' + '---|' * len(labs)]
    for r in rows:
        md.append(f"| {r['truth']} | " + ' | '.join(fmt.format(r[f'{m} {l}']) for l in labs) + ' |')
    md.append('| **median** | ' + ' | '.join('**' + fmt.format(med(f'{m} {l}')) + '**' for l in labs) + ' |\n')
md += ['## Sources and root recovery (after ECG inverse)\n', '| truth | detected sources | candidate rank of each true root (0 = exact) | start-to-g* regression max err (ms) |', '|---|---|---|---|']
for r in rows:
    md.append(f"| {r['truth']} | {r['sources final']} | {r['ranks final']} | {r['regression max err (ms)']:.1f} |")
md += ['', '## ECG-error reduction factors\n', '| truth | g0 -> g* | g* -> final | random-init -> final |', '|---|---|---|---|']
for r in rows:
    a, b, c, o = (r['ecg g0 = truth|S'], r['ecg g* (LAT fit)'], r['ecg g* + ECG inverse'], r['ecg random-init'])
    md.append(f"| {r['truth']} | {a / b:.1f}x | {b / c:.1f}x | {o / c:.0f}x |")
open(os.path.join(OUT, 'tables.md'), 'w').write('\n'.join(md) + '\n')

def dots(ax, key, ylabel, logy=True):
    for (_, lab), c, off in zip(STAGES, COL, (-0.18, 0, 0.18)):
        ax.plot([k + off for k in T], [r[f'{key} {lab}'] for r in rows], 'o', ms=7, color=c,
                mec=SURF, mew=1.5, label=lab, ls='none')
    ax.plot([k + 0.36 for k in T], [r[f'{key} random-init'] for r in rows], 'D', ms=6, color=GRAY,
            mec=SURF, mew=1.5, label='random init (old, median of 100)', ls='none')
    for k in T:
        ys = [rows[k][f'{key} {l}'] for _, l in STAGES]
        ax.plot([k - .18, k, k + .18], ys, '-', color=GRID, lw=1.2, zorder=0)
    if logy: ax.set_yscale('log')
    ax.set_xticks(T); ax.set_xticklabels(xl); ax.set_ylabel(ylabel); ax.set_xlabel('truth (5 roots each)')
    ax.legend(frameon=False, fontsize=9, loc='upper left', bbox_to_anchor=(0, -0.18), ncol=2)

# fig1 ECG error, fig2 LAT MAE
for name, key, yl, ttl in (('fig1_ecg_error', 'ecg', 'relative ECG error (log)', 'ECG error at each stage'),
                           ('fig2_lat_mae', 'MAE', 'LAT MAE, ms (log)', 'LAT error at each stage')):
    fig, ax = plt.subplots(figsize=(7.2, 4.6)); dots(ax, key, yl, True); ax.set_title(ttl, loc='left', fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name + '.png'), dpi=160); plt.close(fig)

# fig3 LAT-fit loss + ECG-inverse trajectories
SER = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.3))
for k in T:
    h = np.load(os.path.join(DATA, f'g{k}.npz'))['lat_hist']
    a.plot(h, color=SER[k], lw=2, label=f's{k}')
    hh = D[k]['hist']; b.plot([x[0] for x in hh], [x[1] for x in hh], '-o', color=SER[k], lw=2, ms=5, mec=SURF, mew=1.2, label=f's{k}')
a.set_yscale('log'); a.set_xlabel('LAT-fit step'); a.set_ylabel('LAT misfit / scale$^2$ (log)'); a.set_title('Step 3: LAT fit of g on S', loc='left', fontsize=12)
b.set_yscale('log'); b.set_xlabel('ECG-inverse iteration'); b.set_ylabel('relative ECG error (log)'); b.set_title('Step 4: ECG inverse from g*', loc='left', fontsize=12)
for ax in (a, b): ax.legend(frameon=False, ncol=3, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig3_trajectories.png'), dpi=160); plt.close(fig)

# fig4 where g changes: dg = g* - g0 vs g0, roots highlighted
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={'width_ratios': [1.5, 1]})
dgs_root, dgs_oth = [], []
for k in T:
    z = np.load(os.path.join(DATA, f'g{k}.npz')); g0, gs, S, roots = z['g0'], z['gstar'], z['s_nodes'], z['roots']
    isr = np.isin(S, roots); dg = gs - g0
    a.scatter(g0[~isr], dg[~isr], s=7, color=GRAY, alpha=.35, lw=0, label='other candidates' if k == 0 else None)
    a.scatter(g0[isr], dg[isr], s=46, color=BLUE, ec=SURF, lw=1.2, zorder=3, label='true roots' if k == 0 else None)
    dgs_root += list(np.abs(dg[isr])); dgs_oth += list(np.abs(dg[~isr]))
a.axhline(0, color=INK2, lw=0.8); a.set_xlabel('g0 = truth|S  (ms)'); a.set_ylabel('g* - g0  (ms)')
a.set_title('How the LAT fit moves g (all six truths)', loc='left', fontsize=12); a.legend(frameon=False, fontsize=9)
b.hist([dgs_oth], bins=40, color=GRAY, alpha=.8, label='other candidates (|dg|)')
b.hist([dgs_root], bins=12, color=BLUE, alpha=.9, label='true roots (|dg|)')
b.set_yscale('log'); b.set_xlabel('|g* - g0|  (ms)'); b.set_ylabel('candidates (log)'); b.legend(frameon=False, fontsize=9)
b.set_title('Size of the change', loc='left', fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig4_g_change.png'), dpi=160); plt.close(fig)
print('roots: mean|dg| %.2f ms; others %.2f ms' % (np.mean(dgs_root), np.mean(dgs_oth)))
print(open(os.path.join(OUT, 'tables.md')).read())
