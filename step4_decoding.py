"""Step 4: time-resolved decoding of faces vs cars, ERP CORE N170, sub-001.

4.1 Downsample the clean epochs from step 2 to 100 Hz
4.2 X (trials x 30 scalp electrodes x time points) and y (face 1, car 0)
4.3 Outer loop: draw 48 of the 60 faces at random, 20 times
4.4 Inner loop: per draw, shuffle within each class and average every 4
    trials into a trace (12 per class); regroup 100 times

Decoding (4.5-4.8) is in step4_run.py; shared settings and functions are in
step4_lib.py.
"""
import mne
import numpy as np

from step4_lib import (DATA, DECODE_SFREQ, N_FACE_DRAWS, N_GROUPINGS,
                       RANDOM_SEED, TRIALS_PER_TRACE, main_seeds, make_groups)

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

# --- 4.3 Outer loop: draw 48 of the 60 faces, 20 times -----------------------
# --- 4.4 Inner loop: group into traces of 4 trials, 100 times per draw -------
# Only indices into X are stored here (which faces each draw keeps, which
# trials go into each trace); the traces themselves are averaged on the fly
# when decoding, since 2000 sets of traces would take ~1 GB.
# Each draw gets its own random stream spawned from RANDOM_SEED, so every
# draw is reproducible on its own, whatever order (or process) runs it in.
# make_groups (step4_lib.py) is also what the permutation test calls, with
# shuffled trial labels instead of y.
face_idx = np.where(y == 1)[0]
car_idx = np.where(y == 0)[0]
face_draws, groups = make_groups(y, main_seeds())
# groups[draw, grouping, class (0 = face, 1 = car), trace, trial in trace]
n_traces = groups.shape[3]

np.savez(DATA / f'step4_draws_{DECODE_SFREQ}hz.npz', face_draws=face_draws,
         groups=groups, random_seed=RANDOM_SEED)
print(f'Saved: data/step4_draws_{DECODE_SFREQ}hz.npz')
print(f'{N_FACE_DRAWS} face draws x {N_GROUPINGS} groupings; each grouping: '
      f'{n_traces} face traces + {n_traces} car traces of {TRIALS_PER_TRACE} trials')

# Checks
left_out = [np.setdiff1d(face_idx, fd) for fd in face_draws]
print(f'Faces left out per draw: {[len(lo) for lo in left_out][0]} '
      f'(draw 1: {left_out[0].tolist()})')
print(f'Distinct face draws: {len({tuple(fd) for fd in face_draws})} of {N_FACE_DRAWS}')
times_left_out = np.bincount(np.concatenate(left_out), minlength=len(y))[face_idx]
print(f'How often each face is left out over the 20 draws: '
      f'min {times_left_out.min()}, max {times_left_out.max()}, '
      f'mean {times_left_out.mean():.1f}')
ok_cover = all(np.array_equal(np.sort(groups[d, g, 0].ravel()), face_draws[d])
               and np.array_equal(np.sort(groups[d, g, 1].ravel()), car_idx)
               for d in range(N_FACE_DRAWS) for g in range(N_GROUPINGS))
print(f'Every grouping uses each drawn trial exactly once: {ok_cover}')
ok_class = (y[groups[:, :, 0]] == 1).all() and (y[groups[:, :, 1]] == 0).all()
print(f'Traces only mix trials of one class: {ok_class}')
n_distinct = len({groups[d, g].tobytes() for d in range(N_FACE_DRAWS)
                  for g in range(N_GROUPINGS)})
print(f'Distinct groupings: {n_distinct} of {N_FACE_DRAWS * N_GROUPINGS}')

# Example: the traces of draw 1, grouping 1
traces_face = X[groups[0, 0, 0]].mean(axis=1)
traces_car = X[groups[0, 0, 1]].mean(axis=1)
print(f'Example traces (draw 1, grouping 1): face {traces_face.shape}, '
      f'car {traces_car.shape} (traces x channels x time points)')
