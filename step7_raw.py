"""Step 7: the same decoding and permutation test on the raw, uncleaned EEG,
ERP CORE N170, sub-001.

Raw here means the recording as it is in the dataset:
  - no re-referencing, no high-pass, no low-pass, no ICA, no interpolation
    of Oz (Oz stays in as recorded), no baseline correction
  - the 3 EOG channels are dropped; the 30 scalp electrodes are used
  - labels from the dataset's stimulus codes (1-40 faces, 41-80 cars)
  - outlier trials are rejected with step 2's EEG amplitude rule (2.5.1):
    peak-to-peak of a channel over the epoch above that channel's mean + 3 SD
    across all 320 epochs AND above 100 µV. Step 2's blink and eye-movement
    rules need the EOG channels, and its behaviour rules are not about the
    EEG, so they are not used.
  - the only filtering is the anti-aliasing inside the resampling to 100 Hz,
    as in step 4 (without it, everything above 50 Hz would fold back into
    the 0-50 Hz range)

Everything after that is as in steps 4-6 (traces of 4 trials,
leave-one-pair-out, GPU linear SVM, the same random streams, 1000
permutations, FDR over all 100 time points), except that, to save time, the
number of curves is cut to a tenth: main analysis 20 draws x 10 groupings
(steps 4/6: 20 x 100), each permutation 20 draws x 1 grouping (step 5/6:
20 x 10). With ~75 trials per class there are 18 traces per class instead of
12, so each curve costs about 10 times more than in step 6.
Both classes are drawn down to the same number of trials (a multiple of 4).

7.1 raw -> epochs -> outlier rejection (reasons logged) -> 100 Hz -> X, y
7.2 groups (20 draws x 10 groupings)
7.3 main analysis on the GPU (200 curves)
7.4 permutations on the GPU (1000 x 20 curves), one file each, resumable
7.5 summary and figure (step5_permutation.summarise)

Run:      .venv\\Scripts\\python.exe step7_raw.py
Summary:  .venv\\Scripts\\python.exe step7_raw.py --summary-only
Progress: data/step7_perm_raw_100hz/progress.log
"""
import argparse
import time
from datetime import timedelta

import mne
import numpy as np
import pandas as pd

from step4_lib import (DATA, DECODE_SFREQ, FIGS, N_FACE_DRAWS, keep_awake,
                       main_seeds, make_groups, permutation_seeds,
                       shuffle_labels)
from step5_permutation import N_PERM, log, perm_file, summarise
from step6_gpu_lib import decode_curves_gpu

RAW_FILE = (DATA / 'step1_erpcore_N170' / 'erpcore' / 'N170'
            / 'sub-001' / 'eeg' / 'sub-001_task-N170_eeg.set')
EOG_CHANNELS = ['HEOG_left', 'HEOG_right', 'VEOG_lower']
# Event codes (task-N170_events.json), as in step 2
CATEGORIES = [(1, 40, 'face'), (41, 80, 'car'),
              (101, 140, 'scrambled_face'), (141, 180, 'scrambled_car')]
# Outlier rule, same values as step 2 (2.5.1)
EEG_Z_THRESHOLD = 3.0
EEG_PTP_FLOOR_UV = 100.0
# A tenth of the curves of steps 4-6 (see above)
GROUPINGS = 10        # per draw, main analysis (steps 4/6: 100)
PERM_GROUPINGS = 1    # per draw, each permutation (steps 5/6: 10)
# Groupings per GPU batch: 200 groupings of 18 traces per class measured at
# 11.4 GB of the 16 GB (step 6's 400 is for 12 traces per class)
CHUNK = 200

REJECT_LOG = DATA / 'step7_reject_log_raw.csv'
XY = DATA / f'step7_Xy_raw_{DECODE_SFREQ}hz.npz'
DRAWS = DATA / f'step7_draws_raw_{DECODE_SFREQ}hz.npz'
ACC = DATA / f'step7_accuracy_raw_{DECODE_SFREQ}hz.npz'
RESULTS = DATA / f'step7_perm_raw_{DECODE_SFREQ}hz'
NULL = DATA / f'step7_null_raw_{DECODE_SFREQ}hz.npz'
FIG = FIGS / f'step7_permutation_raw_{DECODE_SFREQ}hz.png'
LOG_EVERY = 10


def say(msg):
    log(msg, RESULTS)


def build_xy():
    """7.1: raw recording -> rejection log and the face/car epochs at 100 Hz."""
    raw = mne.io.read_raw_eeglab(RAW_FILE, preload=True)
    raw.drop_channels(EOG_CHANNELS)                 # 30 scalp electrodes left
    events, ids = mne.events_from_annotations(raw)
    event_id = {f'{name}/{desc}': code for desc, code in ids.items()
                for lo, hi, name in CATEGORIES if lo <= int(desc) <= hi}
    # all four categories, so the outlier rule sees the same 320 epochs as
    # in step 2; no baseline correction
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-0.2, tmax=0.8,
                        baseline=None, reject=None, preload=True)

    sf = epochs.info['sfreq']
    code_to_tag = {c: t for t, c in epochs.event_id.items()}
    tags = [code_to_tag[c] for c in epochs.events[:, 2]]
    rej = pd.DataFrame({'onset_s': epochs.events[:, 0] / sf,
                        'code': [int(t.split('/')[1]) for t in tags],
                        'category': [t.split('/')[0] for t in tags]})
    ptp = np.ptp(epochs.get_data(), axis=2) * 1e6       # epochs x channels, µV
    z = (ptp - ptp.mean(0)) / ptp.std(0)
    too_big = (z > EEG_Z_THRESHOLD) & (ptp > EEG_PTP_FLOOR_UV)
    names = epochs.ch_names
    rej['eeg_max_ptp_ch'] = [names[i] for i in ptp.argmax(1)]
    rej['eeg_max_ptp_uv'] = ptp.max(1)
    rej['eeg_max_z'] = z.max(1)
    rej['eeg_bad_chs'] = [','.join(names[j] for j in np.where(r)[0]) for r in too_big]
    rej['EEG_AMPLITUDE'] = too_big.any(1)
    rej['rejected'] = rej.EEG_AMPLITUDE
    rej.index.name = 'epoch'

    keep = epochs.copy().drop(np.where(rej.rejected)[0], reason='EEG_AMPLITUDE')
    fc = keep[['face', 'car']].copy().resample(DECODE_SFREQ)
    return rej, fc


def prepare():
    """7.1-7.2: save the rejection log, X / y and the groups."""
    rej, fc = build_xy()
    rej.to_csv(REJECT_LOG, float_format='%.2f')
    say(f'7.1 Saved: data/{REJECT_LOG.name}')
    say(f'7.1 rejected (EEG amplitude) {int(rej.rejected.sum())} of {len(rej)} epochs; '
        + ', '.join(f'{c} {int(g.rejected.sum())}/{len(g)}'
                    for c, g in rej.groupby('category', sort=False)))
    bad = rej.loc[rej.rejected, 'eeg_bad_chs'].str.split(',').explode()
    say('7.1 channels over the limit in rejected epochs: '
        + ', '.join(f'{ch} {n}' for ch, n in bad.value_counts().items()))

    X = fc.get_data()
    code_to_tag = {c: t for t, c in fc.event_id.items()}
    y = np.array([1 if code_to_tag[c].startswith('face/') else 0 for c in fc.events[:, 2]])
    np.savez(XY, X=X, y=y, times=fc.times, ch_names=np.array(fc.ch_names),
             sample=fc.events[:, 0])
    say(f'7.1 Saved: data/{XY.name}; X {X.shape} (trials x channels x time points), '
        f'face {int((y == 1).sum())}, car {int((y == 0).sum())}, '
        f'{fc.times[0] * 1000:.0f} to {fc.times[-1] * 1000:.0f} ms at {fc.info["sfreq"]:.0f} Hz')

    face_draws, groups = make_groups(y, main_seeds(), n_groupings=GROUPINGS)
    np.savez(DRAWS, face_draws=face_draws, groups=groups)
    say(f'7.2 Saved: data/{DRAWS.name}; {N_FACE_DRAWS} draws x {GROUPINGS} groupings, '
        f'{groups.shape[3]} traces per class of {groups.shape[4]} trials '
        f'({groups.shape[3] * groups.shape[4]} trials per class used per draw)')


def perm_groups(y, p):
    """The groups of permutation p: step 5's random streams, 1 grouping per draw."""
    shuffle_seed, draw_seeds = permutation_seeds(p)
    y_perm = shuffle_labels(y, shuffle_seed)
    _, groups = make_groups(y_perm, draw_seeds, n_groupings=PERM_GROUPINGS)
    return groups                                   # (20, 1, 2, n_traces, 4)


def main_analysis(X, groups, chunk):
    """7.3: the 200 curves of the main analysis."""
    t0 = time.time()
    acc = decode_curves_gpu(X, groups, chunk=chunk)       # (draw, grouping, time)
    mean_curve = acc.mean(axis=(0, 1))
    draw_means = acc.mean(axis=1)
    times = np.load(XY)['times']
    np.savez(ACC, acc=acc, mean_curve=mean_curve, draw_means=draw_means,
             sd_within=acc.std(axis=1, ddof=1).mean(axis=0),
             sd_between=draw_means.std(axis=0, ddof=1), times=times)
    t_ms = times * 1000
    post, pre = t_ms >= 0, t_ms < 0
    i_peak = np.where(post)[0][mean_curve[post].argmax()]
    say(f'7.3 main analysis: {acc.shape[0] * acc.shape[1]} curves in '
        f'{timedelta(seconds=round(time.time() - t0))}; peak {mean_curve[i_peak]:.3f} at '
        f'{t_ms[i_peak]:.0f} ms; before stimulus mean {mean_curve[pre].mean():.3f}')
    say(f'7.3 Saved: data/{ACC.name}')


def permutations(X, y, chunk):
    """7.4: 1000 permutations, each saved to its own file."""
    per_batch = max(1, chunk // (N_FACE_DRAWS * PERM_GROUPINGS))
    todo = [p for p in range(N_PERM) if not perm_file(p, RESULTS).exists()]
    say(f'7.4 start: {len(todo)} of {N_PERM} permutations to do '
        f'({N_PERM - len(todo)} already saved), {N_FACE_DRAWS} x {PERM_GROUPINGS} '
        f'curves each, {per_batch} permutation(s) per GPU call, {chunk} groupings '
        f'per GPU batch')
    t0 = time.time()
    n_done = 0
    for b in range(0, len(todo), per_batch):
        batch = todo[b:b + per_batch]
        groups = np.stack([perm_groups(y, p) for p in batch])
        acc = decode_curves_gpu(X, groups, chunk=chunk)    # (perm, 20, 1, time)
        for p, acc_p in zip(batch, acc):
            tmp = perm_file(p, RESULTS).with_suffix('.tmp.npy')
            np.save(tmp, acc_p.mean(axis=1))               # 20 draw means
            tmp.replace(perm_file(p, RESULTS))             # only complete files count
        before, n_done = n_done, n_done + len(batch)
        if n_done // LOG_EVERY > before // LOG_EVERY or n_done == len(todo):
            elapsed = time.time() - t0
            eta = elapsed / n_done * (len(todo) - n_done)
            say(f'  {n_done}/{len(todo)} permutations done, '
                f'elapsed {timedelta(seconds=round(elapsed))}, '
                f'about {timedelta(seconds=round(eta))} left')
    say('7.4 all permutations done')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary-only', action='store_true',
                        help='only summarise the permutations finished so far')
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if not args.summary_only:
        keep_awake()
        # 7.1-7.3 are only done once; delete their files to redo them
        if not (XY.exists() and DRAWS.exists()):
            prepare()
        xy = np.load(XY)
        X, y = xy['X'], xy['y']
        groups = np.load(DRAWS)['groups']
        if not ACC.exists():
            main_analysis(X, groups, CHUNK)
        permutations(X, y, CHUNK)
    summarise(results=RESULTS, real_path=ACC, null_path=NULL, fig_path=FIG)


if __name__ == '__main__':
    main()
