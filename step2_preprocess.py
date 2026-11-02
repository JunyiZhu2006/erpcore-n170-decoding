"""Step 2: preprocessing of ERP CORE N170, sub-001.

2.1 Mark the 3 EOG electrodes (HEOG_left, HEOG_right, VEOG_lower) as EOG type;
    compute bipolar EOG: HEOG = HEOG_left - HEOG_right, VEOG = VEOG_lower - FP2;
    mark Oz as a bad channel (see step2_check_oz.py)
2.2 Average reference over the 29 good scalp EEG channels (Oz and EOG excluded)
2.3.1 High-pass filter at 0.1 Hz
2.3.2 Low-pass filter at 30 Hz
2.3.3a ICA on a 1 Hz high-passed copy; find EOG component candidates
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
# Head outline for topomaps: the dataset's electrodes sit on an 85 mm sphere
# centred at the origin, so electrodes at z = 0 (FP1/FP2, O1/O2) fall on it
SPHERE = (0, 0, 0, 0.085)

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
