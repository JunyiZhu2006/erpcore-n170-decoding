"""Step 4: time-resolved decoding of faces vs cars, ERP CORE N170, sub-001.

4.1 Downsample the clean epochs from step 2 to 100 Hz
4.2 X (trials x 30 scalp electrodes x time points) and y (face 1, car 0)
4.3 Balance the classes: draw 48 of the 60 faces at random
"""
from pathlib import Path

import mne
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'

# Sampling rate for decoding. 100 Hz first for a quick pass; the data were
# low-passed at 30 Hz in step 2, below the 50 Hz Nyquist limit, so no aliasing.
DECODE_SFREQ = 100
# Seed for every random draw in step 4, so results are reproducible
RANDOM_SEED = 42

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

# --- 4.2 X and y -------------------------------------------------------------
# X: trials x 30 scalp electrodes x time points; y: face = 1, car = 0.
# EOG channels are left out, so the classifier cannot separate the classes
# by eye movements instead of brain activity.
fc = epochs_ds[['face', 'car']].copy().pick('eeg')
X = fc.get_data()
code_to_tag = {code: tag for tag, code in fc.event_id.items()}
y = np.array([1 if code_to_tag[code].startswith('face/') else 0
              for code in fc.events[:, 2]])
# sample = stimulus position in the original 1024 Hz recording (events keep
# their original sample numbers after resample())
np.savez(DATA / f'step4_Xy_{DECODE_SFREQ}hz.npz', X=X, y=y, times=fc.times,
         ch_names=np.array(fc.ch_names), sample=fc.events[:, 0])
print(f'Saved: data/step4_Xy_{DECODE_SFREQ}hz.npz')
print(f'X: {X.shape} (trials x channels x time points), y: {y.shape}, '
      f'face (1) {int((y == 1).sum())}, car (0) {int((y == 0).sum())}')
print(f'Channels ({len(fc.ch_names)}): {", ".join(fc.ch_names)}')
eog_in_x = [ch for ch in fc.ch_names if 'EOG' in ch]
print(f'EOG channels in X: {eog_in_x if eog_in_x else "none"}')

# --- 4.3 Same number of trials in both classes ------------------------------
# 60 faces vs 48 cars: draw 48 faces at random so that chance is exactly 50%
# and the classifier is not pushed towards the bigger class. Fixed seed so
# the draw is the same every run.
rng = np.random.default_rng(RANDOM_SEED)
face_idx = np.where(y == 1)[0]
car_idx = np.where(y == 0)[0]
n_per_class = min(len(face_idx), len(car_idx))
face_keep = np.sort(rng.choice(face_idx, size=n_per_class, replace=False))
keep = np.sort(np.r_[face_keep, car_idx[:n_per_class]])
X_bal, y_bal = X[keep], y[keep]
np.savez(DATA / f'step4_balanced_{DECODE_SFREQ}hz.npz', X=X_bal, y=y_bal,
         times=fc.times, ch_names=np.array(fc.ch_names),
         sample=fc.events[keep, 0], kept_from_Xy=keep)
print(f'Saved: data/step4_balanced_{DECODE_SFREQ}hz.npz')
print(f'Balanced: X {X_bal.shape}, face {int((y_bal == 1).sum())}, '
      f'car {int((y_bal == 0).sum())} (seed {RANDOM_SEED})')
dropped = np.setdiff1d(face_idx, face_keep)
print(f'Faces left out ({len(dropped)}), index in X: {dropped.tolist()}')
# events keep counting samples at the original 1024 Hz after resample()
print('  their stimulus onsets (s): '
      + ', '.join(f'{s / epochs.info["sfreq"]:.1f}' for s in fc.events[dropped, 0]))
