"""Step 3: ERPs for faces vs cars, ERP CORE N170, sub-001.

3.1 Split the clean epochs from step 2 into intact faces and intact cars
    (scrambled images are not used here)
3.2 Average each set per electrode: the face ERP and the car ERP
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
# Electrodes shown in figures only; not used to pick or test anything
SHOW_CHANNELS = ['P7', 'P8', 'PO7', 'PO8', 'P9', 'P10']
COLORS = dict(face='#c2385a', car='#1c8cc6')

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

# How much averaging cuts the noise: SD over the baseline (-200..0 ms, where
# there is no stimulus response), single trials vs the average. Random
# noise should shrink by about sqrt(number of trials).
t = epochs.times
b0, b1 = epochs.baseline
in_base = (t >= b0) & (t <= b1)
print('Baseline noise (SD over -200..0 ms, mean over 30 channels):')
for name, ep in sets.items():
    single = ep.get_data()[:, :, in_base].std(axis=2).mean() * 1e6
    avg = evokeds[name].data[:, in_base].std(axis=1).mean() * 1e6
    print(f'  {name}: single trial {single:.2f} µV, average {avg:.2f} µV, '
          f'ratio {single / avg:.1f} (sqrt(n) = {np.sqrt(len(ep)):.1f})')

win = (t >= N170_WIN[0]) & (t <= N170_WIN[1])
print('Mean amplitude 110-150 ms (µV):')
for ch in SHOW_CHANNELS:
    i = evokeds['face'].ch_names.index(ch)
    f, c = (evokeds[n].data[i, win].mean() * 1e6 for n in ['face', 'car'])
    print(f'  {ch:>4}: face {f:6.2f}, car {c:6.2f}, face-car {f - c:6.2f}')

# Figure 1: face vs car ERP at the display electrodes, single trials faint
fig, axes = plt.subplots(2, 3, figsize=(14, 6.5), sharex=True, sharey=True)
for ax, ch in zip(axes.ravel(), SHOW_CHANNELS):
    for name, ep in sets.items():
        i = ep.ch_names.index(ch)
        trials = ep.get_data()[:, i, :] * 1e6
        ax.plot(t * 1000, trials.T, color=COLORS[name], lw=0.3, alpha=0.08)
    for name, evk in evokeds.items():
        i = evk.ch_names.index(ch)
        ax.plot(t * 1000, evk.data[i] * 1e6, color=COLORS[name], lw=2,
                label=f'{name} average (n={evk.nave})')
    ax.axvspan(N170_WIN[0] * 1000, N170_WIN[1] * 1000, color='0.85', zorder=0)
    ax.axvline(0, color='k', lw=0.5)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylim(-25, 25)
    ax.set_title(ch)
for ax in axes[1]:
    ax.set_xlabel('time (ms)')
for ax in axes[:, 0]:
    ax.set_ylabel('µV')
axes[0, 0].legend(fontsize=8, loc='lower right')
fig.suptitle('Face vs car: thin lines = single trials, thick = average; '
             'grey band = ERP CORE N170 window (110-150 ms)')
fig.tight_layout()
fig.savefig(FIGS / 'step3_erp_face_car.png', dpi=100)

# Figure 2: all 30 electrodes laid out on the head
fig = mne.viz.plot_evoked_topo(list(evokeds.values()),
                               color=[COLORS['face'], COLORS['car']],
                               title='Face (red) vs car (blue), all electrodes',
                               show=False)
fig.set_size_inches(12, 9)
fig.savefig(FIGS / 'step3_erp_topo.png', dpi=100, facecolor='white')
plt.close('all')
