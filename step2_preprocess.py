"""Step 2: preprocessing of ERP CORE N170, sub-001.

- Mark the 3 EOG electrodes (HEOG_left, HEOG_right, VEOG_lower) as EOG type
- Compute bipolar EOG: HEOG = HEOG_left - HEOG_right, VEOG = VEOG_lower - FP2
"""
from pathlib import Path

import mne

ROOT = Path(__file__).resolve().parent
RAW_FILE = (ROOT / 'data' / 'step1_erpcore_N170' / 'erpcore' / 'N170'
            / 'sub-001' / 'eeg' / 'sub-001_task-N170_eeg.set')
OUT_FILE = ROOT / 'data' / 'step2_eog-raw.fif'

EOG_CHANNELS = ['HEOG_left', 'HEOG_right', 'VEOG_lower']

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

raw.save(OUT_FILE, overwrite=True)

print(dict(zip(raw.ch_names, raw.get_channel_types())))
print(f'Saved: {OUT_FILE.relative_to(ROOT)}')
