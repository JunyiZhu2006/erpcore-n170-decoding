"""Step 2: preprocessing of ERP CORE N170, sub-001.

2.1 Mark the 3 EOG electrodes (HEOG_left, HEOG_right, VEOG_lower) as EOG type;
    compute bipolar EOG: HEOG = HEOG_left - HEOG_right, VEOG = VEOG_lower - FP2;
    mark Oz as a bad channel (see step2_check_oz.py)
2.2 Average reference over the 29 good scalp EEG channels (Oz and EOG excluded)
2.3.1 High-pass filter at 0.1 Hz
2.3.2 Low-pass filter at 30 Hz
"""
from pathlib import Path

import mne

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RAW_FILE = (DATA / 'step1_erpcore_N170' / 'erpcore' / 'N170'
            / 'sub-001' / 'eeg' / 'sub-001_task-N170_eeg.set')

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
