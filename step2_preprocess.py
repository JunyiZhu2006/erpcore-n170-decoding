"""Step 2: preprocessing of ERP CORE N170, sub-001.

2.1 Mark the 3 EOG electrodes (HEOG_left, HEOG_right, VEOG_lower) as EOG type;
    compute bipolar EOG: HEOG = HEOG_left - HEOG_right, VEOG = VEOG_lower - FP2
2.2 Average reference over the 30 scalp EEG channels (EOG excluded)
"""
from pathlib import Path

import mne

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RAW_FILE = (DATA / 'step1_erpcore_N170' / 'erpcore' / 'N170'
            / 'sub-001' / 'eeg' / 'sub-001_task-N170_eeg.set')

EOG_CHANNELS = ['HEOG_left', 'HEOG_right', 'VEOG_lower']


def save(raw, name):
    out = DATA / name
    raw.save(out, overwrite=True)
    print(f'Saved: {out.relative_to(ROOT)}')


# --- 2.1 EOG channels and bipolar EOG ---------------------------------------
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
save(raw, 'step2_eog-raw.fif')

# --- 2.2 Average reference ---------------------------------------------------
# Only channels typed 'eeg' (the 30 scalp electrodes) enter the average and
# get re-referenced; EOG channels are left untouched. Bipolar EOG was computed
# before this, so it is unaffected (both of its electrodes shared a reference).
raw.set_eeg_reference('average', projection=False)
save(raw, 'step2_avgref-raw.fif')
