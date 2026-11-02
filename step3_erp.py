"""Step 3: ERPs for faces vs cars, ERP CORE N170, sub-001.

3.1 Split the clean epochs from step 2 into intact faces and intact cars
    (scrambled images are not used here)
3.2 Average each set per electrode: the face ERP and the car ERP
3.3 At PO7 and PO8: face, car and the face - car difference wave
3.4 Mean of the difference wave over 110-150 ms, with a 95% CI band
3.5 Topography: face - car in 110-150 ms at every electrode
3.6 Redo with the low-pass at 30 Hz, 40 Hz and none; compare PO8
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

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

# --- 3.6 Same analysis with a different low-pass -----------------------------
# Redo step 2 from the 0.1 Hz high-passed data with the low-pass at 30 Hz,
# 40 Hz, or none. Everything else stays the same: the same ICA solution and
# excluded components, Oz interpolation, average reference, epochs, and the
# same rejected trials (taken from step 2's log, not re-decided).
ica = mne.preprocessing.read_ica(DATA / 'step2_ica-ica.fif')
ica.exclude = [0, 3]   # as confirmed in step 2 (ICA_EOG_CONFIRMED)
reject_log = pd.read_csv(DATA / 'step2_reject_log.csv', index_col='epoch')
raw_hp = mne.io.read_raw_fif(DATA / 'step2_hp-raw.fif', preload=True)
filter_picks = mne.pick_types(raw_hp.info, eeg=True, eog=True, exclude=[])
LOWPASS = {'30 Hz': 30, '40 Hz': 40, 'no low-pass': None}

diff_po8 = {}
for label, h_freq in LOWPASS.items():
    raw_v = raw_hp.copy()
    if h_freq is not None:
        raw_v.filter(l_freq=None, h_freq=h_freq, picks=filter_picks)
    raw_v = ica.apply(raw_v)
    raw_v.interpolate_bads(reset_bads=True, origin=(0.0, 0.0, 0.0))
    raw_v.set_eeg_reference('average', projection=False)
    ev, ids = mne.events_from_annotations(raw_v)
    stim_id = {f'{name}/{d}': c for d, c in ids.items()
               for lo, hi, name in [(1, 40, 'face'), (41, 80, 'car'),
                                    (101, 140, 'scrambled_face'),
                                    (141, 180, 'scrambled_car')]
               if lo <= int(d) <= hi}
    ep_v = mne.Epochs(raw_v, ev, event_id=stim_id, tmin=-0.2, tmax=0.8,
                      baseline=(-0.2, 0), picks=['eeg', 'eog'], reject=None,
                      preload=True)
    ep_v.drop(np.where(reject_log.rejected.to_numpy())[0], reason='STEP2_LOG')
    if h_freq == 30:
        # The 30 Hz version must reproduce step 2's clean epochs. Not bit for
        # bit: step2_hp-raw.fif is stored as float32, while step 2 kept going
        # in float64 (differences ~0.00003 µV), hence the 0.001 µV tolerance
        assert np.allclose(ep_v.get_data(), epochs.get_data(), rtol=0,
                           atol=1e-9), '30 Hz redo does not match step2_clean-epo.fif'
    d_v = mne.combine_evoked([ep_v['face'].average(picks='eeg'),
                              ep_v['car'].average(picks='eeg')], weights=[1, -1])
    diff_po8[label] = d_v.copy().pick('PO8').data[0] * 1e6
    print(f'{label}: {len(ep_v["face"])} face, {len(ep_v["car"])} car trials')

print('PO8 face - car, mean over 110-150 ms (µV):')
fig, ax = plt.subplots(figsize=(9, 4.5))
for (label, d), col in zip(diff_po8.items(), ['k', '#c9690c', '#6a4dd6']):
    print(f'  {label}: {d[win].mean():.2f}')
    ax.plot(t * 1000, d, color=col, lw=1.8 if label == '30 Hz' else 1.1,
            label=f'{label}: {d[win].mean():.2f} µV in 110-150 ms')
ax.axvspan(N170_WIN[0] * 1000, N170_WIN[1] * 1000, color='#f2d98c',
           alpha=0.5, zorder=0)
ax.axvline(0, color='k', lw=0.5)
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('time (ms)')
ax.set_ylabel('face - car (µV)')
ax.set_title('PO8 difference wave with three low-pass settings')
ax.legend(fontsize=9, loc='upper left')
fig.tight_layout()
fig.savefig(FIGS / 'step3_filter_PO8.png', dpi=100)
plt.close('all')
