"""Step 2: preprocessing of ERP CORE N170, sub-001.

2.1 Mark the 3 EOG electrodes (HEOG_left, HEOG_right, VEOG_lower) as EOG type;
    compute bipolar EOG: HEOG = HEOG_left - HEOG_right, VEOG = VEOG_lower - FP2;
    mark Oz as a bad channel (see step2_check_oz.py)
2.2 Average reference over the 29 good scalp EEG channels (Oz and EOG excluded)
2.3.1 High-pass filter at 0.1 Hz
2.3.2 Low-pass filter at 30 Hz
2.3.3a ICA on a 1 Hz high-passed copy; find EOG component candidates
2.3.3b Remove the EOG components (ICA000, ICA003) from the 0.1-30 Hz data
2.3.4 Interpolate Oz, re-reference to the average of all 30 scalp channels
2.4 Epochs from -200 to 800 ms around each stimulus, baseline -200..0 ms
2.5 Reject epochs (EEG amplitude, blinks / eye movements during the stimulus,
    behaviour), recording the reason(s) for every rejected epoch
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
RAW_FILE = (DATA / 'step1_erpcore_N170' / 'erpcore' / 'N170'
            / 'sub-001' / 'eeg' / 'sub-001_task-N170_eeg.set')
ELEC_FILE = RAW_FILE.with_name('sub-001_task-N170_electrodes.tsv')
EVENTS_FILE = RAW_FILE.with_name('sub-001_task-N170_events.tsv')

# Epoch rejection (2.5)
STIM_DURATION = 0.3          # s the image stays on screen (events.tsv duration)
EEG_Z_THRESHOLD = 3.0        # per-channel peak-to-peak, mean + 3 SD
HEOG_STEP_THRESHOLD = 32.0   # µV, about 2 degrees of gaze
RT_RANGE_MS = (200, 1000)    # ERP CORE behavioural limits
# Head outline for topomaps: the dataset's electrodes sit on an 85 mm sphere
# centred at the origin, so electrodes at z = 0 (FP1/FP2, O1/O2) fall on it
SPHERE = (0, 0, 0, 0.085)
# N170 measurement window used by ERP CORE (Kappenman et al., 2021), fixed in
# advance rather than chosen from this subject's data
N170_WIN = (0.110, 0.150)

EOG_CHANNELS = ['HEOG_left', 'HEOG_right', 'VEOG_lower']
BAD_CHANNELS = ['Oz']  # intermittent jumps up to ~1700 µV, low-frequency only


def save(raw, name):
    out = DATA / name
    raw.save(out, overwrite=True)
    print(f'Saved: {out.relative_to(ROOT)}')


# --- 2.1 EOG channels, bipolar EOG, bad channels ----------------------------
raw = mne.io.read_raw_eeglab(RAW_FILE, preload=True)

# The .set file loads every channel as EEG; channels.tsv says these are EOG
raw.set_channel_types({ch: 'eog' for ch in EOG_CHANNELS})

# Bipolar derivations (anode - cathode); keep the original channels,
# FP2 is still needed as an EEG channel
raw = mne.set_bipolar_reference(
    raw,
    anode=['HEOG_left', 'VEOG_lower'],
    cathode=['HEOG_right', 'FP2'],
    ch_name=['HEOG', 'VEOG'],
    drop_refs=False,
    copy=False,
)
raw.set_channel_types({'HEOG': 'eog', 'VEOG': 'eog'})

# The .set file has no electrode positions; take them from the dataset's own
# electrodes.tsv (needed for ICA topomaps and for interpolating bad channels).
# These are idealised positions on an 85 mm sphere, not digitised per subject.
# coordsystem.json says ARS, but left electrodes (FP1, F7, ...) have +y, so
# the axes are x = nose, y = left ear, z = up. MNE's head frame is x = right,
# y = nose, z = up, in metres.
elec = pd.read_csv(ELEC_FILE, sep='\t', na_values='n/a').dropna().set_index('name')
ch_pos = {name: np.array([-r.y, r.x, r.z]) / 1000 for name, r in elec.iterrows()}
raw.set_montage(mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head'))

# Bad channels stay in the data but are left out of the average reference
# and, later, ICA; they get interpolated from their neighbours afterwards
raw.info['bads'] = BAD_CHANNELS
save(raw, 'step2_eog-raw.fif')

# --- 2.2 Average reference ---------------------------------------------------
# Only good channels typed 'eeg' (29 scalp electrodes, Oz excluded) enter the
# average; EOG channels are left untouched. Bipolar EOG was computed before
# this, so it is unaffected (both of its electrodes shared a reference).
raw.set_eeg_reference('average', projection=False)
save(raw, 'step2_avgref-raw.fif')

# Filter all EEG and EOG channels, bad ones included, so that Oz stays on the
# same footing as its neighbours until it is interpolated
FILTER_PICKS = mne.pick_types(raw.info, eeg=True, eog=True, exclude=[])

# --- 2.3.1 High-pass 0.1 Hz --------------------------------------------------
# Removes slow drifts (sweat, skin potentials, electrode drift). MNE default
# FIR: zero-phase, Hamming window, 0.1 Hz transition band (-6 dB at 0.05 Hz),
# ~33 s kernel. EOG is filtered too so it stays comparable with EEG for ICA
# and epoch rejection later.
raw.filter(l_freq=0.1, h_freq=None, picks=FILTER_PICKS)
save(raw, 'step2_hp-raw.fif')

# --- 2.3.2 Low-pass 30 Hz ----------------------------------------------------
# Removes 60 Hz line noise and part of the muscle activity; also needed
# before downsampling to 100 Hz later (Nyquist 50 Hz). MNE default FIR:
# 7.5 Hz transition band (-6 dB at 33.75 Hz), ~0.44 s kernel.
raw.filter(l_freq=None, h_freq=30, picks=FILTER_PICKS)
save(raw, 'step2_lp-raw.fif')

# --- 2.3.3a ICA: fit and find EOG candidates ---------------------------------
# Fit on a copy high-passed at 1 Hz: slow drifts do not have a fixed scalp
# distribution and would degrade the decomposition. The unmixing found here
# is applied to the 0.1-30 Hz data later.
raw_ica = raw.copy().filter(l_freq=1, h_freq=None, picks=FILTER_PICKS)

# Only the 29 good scalp channels are used (EOG and bad Oz are left out).
# After average reference their rank is 28, so at most 28 components.
# 2-s segments with any channel > 500 µV peak-to-peak are skipped during
# fitting (about 1% of the recording); blinks stay well below this.
ica = mne.preprocessing.ICA(n_components=None, method='fastica',
                            random_state=97, max_iter='auto')
ica.fit(raw_ica, picks='eeg', reject=dict(eeg=500e-6), tstep=2.0)
ica.save(DATA / 'step2_ica-ica.fif', overwrite=True)

# Bipolar EOG is used here for the first time: components that correlate
# strongly with HEOG or VEOG are candidates, to be confirmed by eye
eog_idx, eog_scores = ica.find_bads_eog(raw_ica, ch_name=['HEOG', 'VEOG'])
eog_idx = [int(i) for i in eog_idx]
print(f'ICA components: {ica.n_components_}')
print(f'EOG candidates: {eog_idx}')
for ch, sc in zip(['HEOG', 'VEOG'], eog_scores):
    top = np.argsort(np.abs(sc))[::-1][:3]
    print(f'  {ch} top |r|: ' + ', '.join(f'ICA{i:03d} {sc[i]:+.2f}' for i in top))

# Figures for checking the candidates
for i, fig in enumerate(np.atleast_1d(ica.plot_components(sphere=SPHERE,
                                                          show=False))):
    fig.savefig(FIGS / f'step2_ica_components_{i + 1}.png', dpi=100)
fig = ica.plot_scores(eog_scores, exclude=eog_idx, labels=['HEOG', 'VEOG'],
                      show=False)
fig.savefig(FIGS / 'step2_ica_eog_scores.png', dpi=100)
for comp, fig in zip(eog_idx, ica.plot_properties(
        raw_ica, picks=eog_idx, topomap_args=dict(sphere=SPHERE), show=False)):
    fig.savefig(FIGS / f'step2_ica_props_ICA{comp:03d}.png', dpi=100)

# Candidate time courses next to the bipolar EOG, 30 s window
t0, t1 = 20.0, 50.0
eog = raw_ica.copy().crop(t0, t1)
src = ica.get_sources(eog)
rows = [('VEOG', eog.get_data(picks='VEOG')[0])]
rows += [('HEOG', eog.get_data(picks='HEOG')[0])]
rows += [(f'ICA{c:03d}', src.get_data(picks=[c])[0]) for c in eog_idx]
fig, axes = plt.subplots(len(rows), 1, figsize=(12, 1.6 * len(rows)),
                         sharex=True)
for ax, (name, y) in zip(np.atleast_1d(axes), rows):
    ax.plot(src.times + t0, y, lw=0.7, color='#c2385a' if 'EOG' in name else '#6a4dd6')
    ax.set_ylabel(name, rotation=0, ha='right', va='center')
    ax.set_yticks([])
np.atleast_1d(axes)[-1].set_xlabel('time (s)')
fig.tight_layout()
fig.savefig(FIGS / 'step2_ica_sources_vs_eog.png', dpi=100)
plt.close('all')

# --- 2.3.3b ICA: remove the EOG components from the 0.1-30 Hz data ----------
# Confirmed by eye (topomaps, time courses, step2_check_ica003.py):
# ICA000 = blinks, ICA003 = horizontal eye movements. All other components
# are kept, including ICA010 (bilateral P9/P10, the N170 face effect).
ica.exclude = [0, 3]
raw_clean = ica.apply(raw.copy())
# Only the 29 ICA channels change; EOG channels keep the uncorrected signal,
# which is what epoch rejection must look at later. Oz (bad) is untouched.
save(raw_clean, 'step2_ica-raw.fif')

# Check 1: blink-locked average at FP1/FP2 before vs after
blinks = mne.preprocessing.find_eog_events(raw, ch_name='VEOG')
blink_before = mne.Epochs(raw, blinks, tmin=-0.5, tmax=0.5, baseline=(-0.5, -0.3),
                          picks=['FP1', 'FP2'], preload=True).average()
blink_after = mne.Epochs(raw_clean, blinks, tmin=-0.5, tmax=0.5,
                         baseline=(-0.5, -0.3), picks=['FP1', 'FP2'],
                         preload=True).average()

# Check 2: stimulus-locked ERP at the N170 sites before vs after
annot_events, annot_ids = mne.events_from_annotations(raw)
faces = [v for k, v in annot_ids.items() if 1 <= int(k) <= 40]
cars = [v for k, v in annot_ids.items() if 41 <= int(k) <= 80]


def erp(r, ids):
    ev = annot_events[np.isin(annot_events[:, 2], ids)]
    return mne.Epochs(r, ev, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0),
                      picks=['P7', 'P8', 'PO7', 'PO8'], preload=True).average()


erps = {(cond, when): erp(r, ids)
        for cond, ids in [('faces', faces), ('cars', cars)]
        for when, r in [('before', raw), ('after', raw_clean)]}

print(f'Blinks found in VEOG: {len(blinks)}')
for ch in ['FP1', 'FP2']:
    b = blink_before.copy().pick(ch).data[0] * 1e6
    a = blink_after.copy().pick(ch).data[0] * 1e6
    print(f'  {ch} blink peak-to-peak: before {np.ptp(b):.1f} µV, after {np.ptp(a):.1f} µV')
t_erp = erps['faces', 'before'].times
win = (t_erp >= N170_WIN[0]) & (t_erp <= N170_WIN[1])
for ch in ['P7', 'P8', 'PO7', 'PO8']:
    vals = []
    for cond in ['faces', 'cars']:
        b = erps[cond, 'before'].copy().pick(ch).data[0] * 1e6
        a = erps[cond, 'after'].copy().pick(ch).data[0] * 1e6
        vals.append(f'{cond} mean {b[win].mean():.2f} -> {a[win].mean():.2f} µV '
                    f'(r={np.corrcoef(b, a)[0, 1]:.3f})')
    diff = {when: (erps['faces', when].copy().pick(ch).data[0]
                   - erps['cars', when].copy().pick(ch).data[0])[win].mean() * 1e6
            for when in ['before', 'after']}
    vals.append(f'face-car {diff["before"]:.2f} -> {diff["after"]:.2f} µV')
    print(f'  {ch}: ' + '; '.join(vals))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
for ax, ch in zip(axes, ['FP1', 'FP2']):
    t = blink_before.times * 1000
    ax.plot(t, blink_before.copy().pick(ch).data[0] * 1e6, color='#8b98a3', label='before ICA')
    ax.plot(t, blink_after.copy().pick(ch).data[0] * 1e6, color='#6a4dd6', label='after ICA')
    ax.set_title(f'{ch}, average of {len(blinks)} blinks')
    ax.set_xlabel('time from blink peak (ms)')
axes[0].set_ylabel('µV')
axes[0].legend()
fig.tight_layout()
fig.savefig(FIGS / 'step2_ica_blinks_before_after.png', dpi=100)

fig, axes = plt.subplots(1, 4, figsize=(15, 3.5), sharey=True)
for ax, ch in zip(axes, ['P7', 'PO7', 'P8', 'PO8']):
    for cond, col in [('faces', '#c2385a'), ('cars', '#1c8cc6')]:
        t = erps[cond, 'before'].times * 1000
        ax.plot(t, erps[cond, 'before'].copy().pick(ch).data[0] * 1e6, color=col,
                lw=2.5, alpha=0.35, label=f'{cond} before')
        ax.plot(t, erps[cond, 'after'].copy().pick(ch).data[0] * 1e6, color=col,
                lw=1, label=f'{cond} after')
    ax.axvline(0, color='k', lw=0.5)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(ch)
    ax.set_xlabel('time (ms)')
axes[0].set_ylabel('µV')
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGS / 'step2_ica_erp_before_after.png', dpi=100)
plt.close('all')

# --- 2.3.4 Interpolate Oz, re-reference to the average of all 30 ------------
# Done after ICA so Oz's jumps never entered the decomposition. Spherical
# spline interpolation from the neighbours; the dataset's electrodes sit on a
# sphere centred at the origin, so that is the interpolation origin.
oz_original = raw_clean.get_data(picks='Oz')[0].copy()
raw_clean.interpolate_bads(reset_bads=True, origin=(0.0, 0.0, 0.0))
# Oz is now a mix of the other 29 (already average-referenced); re-reference
# so all 30 scalp channels sum to zero again. The 30 channels have rank 28.
raw_clean.set_eeg_reference('average', projection=False)
save(raw_clean, 'step2_interp-raw.fif')

eeg_now = raw_clean.get_data(picks='eeg')
oz_new = raw_clean.get_data(picks='Oz')[0]
o12 = raw_clean.get_data(picks=['O1', 'O2']).mean(0)
print(f'Bad channels now: {raw_clean.info["bads"]}')
print(f'Sum of 30 EEG, max |.|: {np.abs(eeg_now.sum(0)).max() * 1e6:.4f} µV')
print(f'Oz std: original {oz_original.std() * 1e6:.1f} µV, '
      f'interpolated {oz_new.std() * 1e6:.1f} µV; '
      f'O1 {raw_clean.get_data(picks="O1")[0].std() * 1e6:.1f}, '
      f'O2 {raw_clean.get_data(picks="O2")[0].std() * 1e6:.1f} µV')
print(f'corr(interpolated Oz, mean of O1/O2): {np.corrcoef(oz_new, o12)[0, 1]:.2f}')

# --- 2.4 Epochs: -200 to 800 ms around each stimulus, baseline correction ---
# Event codes (task-N170_events.json): 1-40 faces, 41-80 cars,
# 101-140 scrambled faces, 141-180 scrambled cars; 201/202 are responses.
# Each epoch keeps the stimulus code as a tag, e.g. 'face/12', so that
# epochs['face'] selects a whole category.
CATEGORIES = [(1, 40, 'face'), (41, 80, 'car'),
              (101, 140, 'scrambled_face'), (141, 180, 'scrambled_car')]
annot_events, annot_ids = mne.events_from_annotations(raw_clean)
event_id = {}
for desc, code in annot_ids.items():
    for lo, hi, name in CATEGORIES:
        if lo <= int(desc) <= hi:
            event_id[f'{name}/{desc}'] = code

# Baseline: for every epoch and channel, subtract the mean of -200..0 ms.
# Done after filtering because the 0.1 Hz high-pass kernel is ~33 s long.
# No rejection yet (that is the next step); EOG channels are kept, still
# uncorrected, for rejecting blinks/eye movements during the stimulus.
epochs = mne.Epochs(raw_clean, annot_events, event_id=event_id,
                    tmin=-0.2, tmax=0.8, baseline=(-0.2, 0),
                    picks=['eeg', 'eog'], reject=None, preload=True)
epochs.save(DATA / 'step2_epochs-epo.fif', overwrite=True)
print(f'Saved: data/step2_epochs-epo.fif')

print(epochs)
for _, _, name in CATEGORIES:
    print(f'  {name}: {len(epochs[name])}')
# At 1024 Hz the first sample is -200.2 ms, which lies just outside the
# baseline window; use the same samples MNE used for the baseline
b0, b1 = epochs.baseline
in_base = (epochs.times >= b0) & (epochs.times <= b1)
base = epochs.get_data(picks='eeg')[:, :, in_base].mean(2)
print(f'Baseline mean per epoch/channel, max |.|: {np.abs(base).max() * 1e6:.2e} µV')

# Quick look: average ERP per category at the N170 sites
fig, axes = plt.subplots(1, 4, figsize=(15, 3.5), sharey=True)
colors = dict(face='#c2385a', car='#1c8cc6', scrambled_face='#e8a0b0',
              scrambled_car='#a0c8e0')
for ax, ch in zip(axes, ['P7', 'PO7', 'P8', 'PO8']):
    for _, _, name in CATEGORIES:
        evk = epochs[name].average(picks=ch)
        ax.plot(evk.times * 1000, evk.data[0] * 1e6, color=colors[name],
                lw=1.8 if 'scrambled' not in name else 1, label=name)
    ax.axvline(0, color='k', lw=0.5)
    ax.axhline(0, color='k', lw=0.5)
    ax.axvspan(N170_WIN[0] * 1000, N170_WIN[1] * 1000, color='0.9', zorder=0)
    ax.set_title(ch)
    ax.set_xlabel('time (ms)')
axes[0].set_ylabel('µV')
axes[0].legend(fontsize=8)
fig.suptitle('All epochs, no rejection yet '
             '(grey band = ERP CORE N170 window, 110-150 ms)')

# N170 face effect in the ERP CORE window, per category and face - car
t = epochs.times
win = (t >= N170_WIN[0]) & (t <= N170_WIN[1])
print('Mean amplitude 110-150 ms (µV):')
for ch in ['P7', 'PO7', 'P8', 'PO8']:
    m = {name: epochs[name].average(picks=ch).data[0][win].mean() * 1e6
         for _, _, name in CATEGORIES}
    print(f'  {ch}: ' + ', '.join(f'{k} {v:.2f}' for k, v in m.items())
          + f'; face-car {m["face"] - m["car"]:.2f}')
fig.tight_layout()
fig.savefig(FIGS / 'step2_epochs_erp_by_category.png', dpi=100)
plt.close('all')

# --- 2.5 Reject epochs, recording every reason ------------------------------
# Each epoch is checked against all rules; an epoch can fail several. All
# measures and flags go into epochs.metadata and data/step2_reject_log.csv,
# and each dropped epoch carries its reasons in epochs.drop_log.
t = epochs.times
sf = epochs.info['sfreq']
in_stim = (t >= 0) & (t <= STIM_DURATION)
log = pd.DataFrame({
    'onset_s': epochs.events[:, 0] / sf,
    'code': [int(k.split('/')[1]) for k in
             [{v: k for k, v in epochs.event_id.items()}[e] for e in epochs.events[:, 2]]],
})
log['category'] = [name for c in log.code for lo, hi, name in CATEGORIES if lo <= c <= hi]

# 2.5.1 Scalp EEG amplitude: peak-to-peak of each channel over the whole
# epoch; an epoch fails if any channel exceeds that channel's own
# mean + 3 SD across epochs (3 SD is the strictest allowed).
eeg_names = epochs.copy().pick('eeg').ch_names
ptp = np.ptp(epochs.get_data(picks='eeg'), axis=2) * 1e6       # epochs x channels
z = (ptp - ptp.mean(0)) / ptp.std(0)
log['eeg_max_z'] = z.max(1)
log['eeg_worst_ch'] = [eeg_names[i] for i in z.argmax(1)]
log['eeg_worst_ptp_uv'] = ptp[np.arange(len(ptp)), z.argmax(1)]
log['EEG_AMPLITUDE'] = log.eeg_max_z > EEG_Z_THRESHOLD

# 2.5.2a Blinks during the stimulus: VEOG (uncorrected) peak-to-peak within
# the 300 ms the image is on screen, against half of this subject's median
# blink size (blinks found in the continuous VEOG).
blink_events = mne.preprocessing.find_eog_events(raw_clean, ch_name='VEOG')
veog_cont = raw_clean.get_data(picks='VEOG')[0] * 1e6
half_win = int(0.25 * sf)
blink_size = np.median([np.ptp(veog_cont[max(s - half_win, 0):s + half_win])
                        for s in blink_events[:, 0]])
veog_threshold = blink_size / 2
veog = epochs.get_data(picks='VEOG')[:, 0] * 1e6
log['veog_ptp_stim_uv'] = np.ptp(veog[:, in_stim], axis=1)
log['BLINK'] = log.veog_ptp_stim_uv > veog_threshold

# 2.5.2b Eye movements during the stimulus: step in HEOG (uncorrected),
# mean of the 200 ms after minus the 200 ms before, for every centre point
# within the stimulus period. 32 µV is about 2 degrees of gaze (~16 µV/deg).
heog = epochs.get_data(picks='HEOG')[:, 0] * 1e6
step_half = int(0.2 * sf)
steps = np.array([heog[:, i:i + step_half].mean(1) - heog[:, i - step_half:i].mean(1)
                  for i in np.where(in_stim)[0]])
log['heog_step_stim_uv'] = np.abs(steps).max(0)
log['SACCADE'] = log.heog_step_stim_uv > HEOG_STEP_THRESHOLD

# 2.5.3 Behaviour: the response is the first response event after the
# stimulus and before the next one (events.tsv; codes 201 correct, 202 error)
beh = pd.read_csv(EVENTS_FILE, sep='\t')
beh_rows = beh[['sample', 'trial_type', 'value']].to_numpy()
resp_by_sample = {}
for k, (sample, ttype, value) in enumerate(beh_rows):
    if ttype != 'stimulus':
        continue
    resp, rt = np.nan, np.nan
    if k + 1 < len(beh_rows) and beh_rows[k + 1][1] == 'response':
        resp = int(beh_rows[k + 1][2])
        rt = (beh_rows[k + 1][0] - sample) / sf * 1000
    resp_by_sample[int(sample)] = (resp, rt)
# events.tsv samples are 1-based; MNE's are 0-based
log['response'], log['rt_ms'] = zip(*[resp_by_sample[s + 1] for s in epochs.events[:, 0]])
log['NO_RESPONSE'] = log.response.isna()
log['WRONG'] = log.response == 202
log['RT_FAST'] = log.rt_ms < RT_RANGE_MS[0]
log['RT_SLOW'] = log.rt_ms > RT_RANGE_MS[1]

REASONS = ['EEG_AMPLITUDE', 'BLINK', 'SACCADE', 'NO_RESPONSE', 'WRONG',
           'RT_FAST', 'RT_SLOW']
log['reasons'] = log[REASONS].apply(lambda r: ';'.join(c for c in REASONS if r[c]), axis=1)
log['rejected'] = log.reasons != ''
log.index.name = 'epoch'
log.to_csv(DATA / 'step2_reject_log.csv', float_format='%.2f')
print(f'Saved: data/step2_reject_log.csv')

epochs.metadata = log
epochs.save(DATA / 'step2_epochs-epo.fif', overwrite=True)  # all 320, with metadata

# Drop from the highest index down so the remaining indices stay valid
clean = epochs.copy()
for i in log.index[log.rejected][::-1]:
    clean.drop([i], reason=tuple(log.reasons[i].split(';')))
clean.save(DATA / 'step2_clean-epo.fif', overwrite=True)
print(f'Saved: data/step2_clean-epo.fif')

print(f'EEG z threshold {EEG_Z_THRESHOLD} -> per-channel ptp limits '
      f'{(ptp.mean(0) + EEG_Z_THRESHOLD * ptp.std(0)).min():.0f}-'
      f'{(ptp.mean(0) + EEG_Z_THRESHOLD * ptp.std(0)).max():.0f} µV')
print(f'Blinks in continuous VEOG: {len(blink_events)}, median size '
      f'{blink_size:.0f} µV -> VEOG threshold {veog_threshold:.0f} µV')
print('Epochs failing each rule (an epoch can fail several):')
for r in REASONS:
    print(f'  {r}: {log[r].sum()}')
print(f'Rejected: {log.rejected.sum()} of {len(log)}; kept {len(clean)}')
print(log.groupby('category').rejected.agg(['sum', 'count'])
      .assign(kept=lambda d: d['count'] - d['sum']))
