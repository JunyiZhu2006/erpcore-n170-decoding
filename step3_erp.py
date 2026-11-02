"""Step 3: ERPs for faces vs cars, ERP CORE N170, sub-001.

3.1 Split the clean epochs from step 2 into intact faces and intact cars
    (scrambled images are not used here)
3.2 Average each set per electrode: the face ERP and the car ERP
3.3 At PO7 and PO8: face, car and the face - car difference wave
3.4 Mean of the difference wave over 110-150 ms, with a 95% CI band
3.5 Topography: face - car in 110-150 ms at every electrode
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
FIGS = ROOT / 'figs'

# N170 measurement window used by ERP CORE (Kappenman et al., 2021)
N170_WIN = (0.110, 0.150)
COLORS = dict(face='#c2385a', car='#1c8cc6')
# Head outline for topomaps, same as step 2: the dataset's electrodes sit on
# an 85 mm sphere centred at the origin
SPHERE = (0, 0, 0, 0.085)

# --- 3.1 Two sets of trials: faces and cars ---------------------------------
epochs = mne.read_epochs(DATA / 'step2_clean-epo.fif')
sets = {name: epochs[name].copy().pick('eeg') for name in ['face', 'car']}
for name, ep in sets.items():
    print(f'{name}: {len(ep)} trials, {len(ep.ch_names)} EEG channels')

# --- 3.2 Average each set -----------------------------------------------------
evokeds = {name: ep.average() for name, ep in sets.items()}
for name, evk in evokeds.items():
    evk.comment = name
mne.write_evokeds(DATA / 'step3_face_car-ave.fif', list(evokeds.values()),
                  overwrite=True)
print('Saved: data/step3_face_car-ave.fif')

# --- 3.3 PO7 and PO8: face, car, and face - car ------------------------------
# Faces and cars share most of the visual response; subtracting leaves what
# is specific to faces. Expected: negative difference around 110-150 ms.
DIFF_CHANNELS = ['PO7', 'PO8']
diff = mne.combine_evoked([evokeds['face'], evokeds['car']], weights=[1, -1])

# --- 3.4 One number per electrode, plus an error band ------------------------
# Error band: 95% confidence interval of the difference between two
# independent sets of trials, mean_face - mean_car +/- 1.96 * SE with
# SE = sqrt(var_face / n_face + var_car / n_car), at every time point.
# The 110-150 ms window is fixed in advance and does not change with results.
t = epochs.times
win = (t >= N170_WIN[0]) & (t <= N170_WIN[1])


def diff_se(data_face, data_car):
    """SE of face - car for per-trial values (trials along axis 0)."""
    return np.sqrt(data_face.var(axis=0, ddof=1) / len(data_face)
                   + data_car.var(axis=0, ddof=1) / len(data_car))


print('Face - car, mean over 110-150 ms (µV), with 95% CI:')
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, ch in zip(axes, DIFF_CHANNELS):
    i = diff.ch_names.index(ch)
    xf = sets['face'].get_data()[:, i, :] * 1e6
    xc = sets['car'].get_data()[:, i, :] * 1e6
    d = diff.data[i] * 1e6
    se = diff_se(xf, xc)

    # The number: per-trial window means, then the same difference and SE
    wf, wc = xf[:, win].mean(axis=1), xc[:, win].mean(axis=1)
    d_win = wf.mean() - wc.mean()
    se_win = diff_se(wf, wc)
    print(f'  {ch}: {d_win:.2f} µV, 95% CI [{d_win - 1.96 * se_win:.2f}, '
          f'{d_win + 1.96 * se_win:.2f}]')

    ax.fill_between(t * 1000, d - 1.96 * se, d + 1.96 * se, color='k',
                    alpha=0.15, lw=0, label='face - car 95% CI')
    for name in ['face', 'car']:
        ax.plot(t * 1000, evokeds[name].data[i] * 1e6, color=COLORS[name],
                lw=1.5, label=f'{name} (n={evokeds[name].nave})')
    ax.plot(t * 1000, d, color='k', lw=2, label='face - car')
    ax.axvspan(N170_WIN[0] * 1000, N170_WIN[1] * 1000, color='#f2d98c',
               alpha=0.5, zorder=0)
    ax.axvline(0, color='k', lw=0.5)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(f'{ch}: face - car in 110-150 ms = {d_win:.2f} µV '
                 f'[{d_win - 1.96 * se_win:.2f}, {d_win + 1.96 * se_win:.2f}]')
    ax.set_xlabel('time (ms)')
axes[0].set_ylabel('µV')
axes[0].legend(fontsize=8, loc='upper left')
fig.suptitle('Yellow band = ERP CORE N170 window (110-150 ms); '
             'grey = 95% CI of the difference wave')
fig.tight_layout()
fig.savefig(FIGS / 'step3_diff_PO7_PO8.png', dpi=100)
plt.close('all')

# --- 3.5 Topography of face - car in 110-150 ms ------------------------------
# One number per electrode, drawn on the head. Expected: negative over both
# sides of the back of the head; positive over the forehead (a consequence of
# the average reference).
topo = diff.data[:, win].mean(axis=1) * 1e6
print('Face - car, mean over 110-150 ms, all electrodes (µV, most negative first):')
for i in np.argsort(topo):
    print(f'  {diff.ch_names[i]:>4} {topo[i]:6.2f}')

lim = np.abs(topo).max()
fig, ax = plt.subplots(figsize=(6, 5.5))
im, _ = mne.viz.plot_topomap(topo, diff.info, axes=ax, sphere=SPHERE,
                             cmap='RdBu_r', vlim=(-lim, lim), contours=6,
                             names=diff.ch_names, show=False)
fig.colorbar(im, ax=ax, shrink=0.7, label='face - car (µV)')
ax.set_title('Face - car, 110-150 ms')
fig.savefig(FIGS / 'step3_topo_diff.png', dpi=100, bbox_inches='tight')
plt.close('all')
