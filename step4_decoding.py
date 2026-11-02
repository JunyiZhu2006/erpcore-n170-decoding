"""Step 4: time-resolved decoding of faces vs cars, ERP CORE N170, sub-001.

4.1 Downsample the clean epochs from step 2 to 100 Hz
"""
from pathlib import Path

import mne
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'

# Sampling rate for decoding. 100 Hz first for a quick pass; the data were
# low-passed at 30 Hz in step 2, below the 50 Hz Nyquist limit, so no aliasing.
DECODE_SFREQ = 100

# --- 4.1 Downsample to 100 Hz ------------------------------------------------
epochs = mne.read_epochs(DATA / 'step2_clean-epo.fif')
print(f'Before: {epochs.info["sfreq"]:.0f} Hz, {len(epochs.times)} time points, '
      f'{epochs.times[0] * 1000:.1f} to {epochs.times[-1] * 1000:.1f} ms')

epochs_ds = epochs.copy().resample(DECODE_SFREQ)
epochs_ds.save(DATA / f'step4_epochs_{DECODE_SFREQ}hz-epo.fif', overwrite=True)
print(f'Saved: data/step4_epochs_{DECODE_SFREQ}hz-epo.fif')

t = epochs_ds.times
print(f'After: {epochs_ds.info["sfreq"]:.0f} Hz, {len(t)} time points, '
      f'{t[0] * 1000:.1f} to {t[-1] * 1000:.1f} ms, step {np.diff(t).mean() * 1000:.1f} ms')
print(f'Epochs: {len(epochs_ds)} '
      f'(face {len(epochs_ds["face"])}, car {len(epochs_ds["car"])})')

# Check: the averages before and after downsampling should look the same
for name in ['face', 'car']:
    for ch in ['PO7', 'PO8']:
        hi = epochs[name].average(picks=ch)
        lo = epochs_ds[name].average(picks=ch)
        hi_at_lo = np.interp(lo.times, hi.times, hi.data[0])
        r = np.corrcoef(hi_at_lo, lo.data[0])[0, 1]
        dmax = np.abs(hi_at_lo - lo.data[0]).max() * 1e6
        print(f'  {name} {ch}: r = {r:.4f}, max difference {dmax:.2f} µV')
