"""Step 8: decoding with time windows (method 2), ERP CORE N170, sub-001.

Method 1 (steps 4-6): one sample = one time point, 30 electrodes -> 30 values.
Method 2 (this step), as in Nemrodov et al. (2018): one sample = a short
time window, the electrodes of WIN_WIDTH neighbouring time points put one
after the other -> 30 x 5 = 150 values. The window moves on by WIN_STEP
point(s): 100 time points -> 96 windows. A window is labelled by its centre.

At 100 Hz one point is 10 ms, so the paper's 10 ms window would be a single
point (= method 1); here the window is 5 points = 50 ms.

Everything else is step 6, unchanged:
  - the cleaned step 4 data (data/step4_Xy_100hz.npz)
  - main analysis: the step 4 groups, 20 draws x 100 groupings = 2000 curves
  - permutations: 1000, the step 5/6 random streams, 20 x 10 curves each
  - traces of 4 trials, leave-one-pair-out (12 folds), standardise with the
    training traces, linear SVM C = 1 (GPU, step6_gpu_lib)
  - p values and FDR over all windows (step5_permutation.summarise)

How the windows get into the step 6 code: the window's 150 values are put
where the 30 electrodes were, so X becomes (trials, 150, 96 windows) and
decode_curves_gpu treats each window like one time point. A trace is the
mean of 4 trials, and taking the mean does not care whether the values were
put into windows before or after, so this is the same as building windows
from the traces.

Run:      .venv\\Scripts\\python.exe step8_window.py
Summary:  .venv\\Scripts\\python.exe step8_window.py --summary-only
Progress: data/step8_perm_win5_100hz/progress.log
"""
import argparse
import time
from datetime import timedelta

import numpy as np

import step6_gpu_lib
from step4_lib import DATA, DECODE_SFREQ, FIGS, N_FACE_DRAWS, N_GROUPINGS, keep_awake
from step5_permutation import N_PERM, PERM_GROUPINGS, log, perm_file, summarise
from step6_gpu_lib import decode_curves_gpu
from step6_permutation import perm_groups

WIN_WIDTH = 5     # time points per window (5 x 10 ms = 50 ms)
WIN_STEP = 1      # the window moves on by this many points
# The training traces are 5 times bigger than in step 6 (150 instead of 30
# values), so the kernels are built 10 instead of 50 groupings at a time
step6_gpu_lib.BUILD_CHUNK = 10
CHUNK = step6_gpu_lib.CHUNK                     # 400 groupings per GPU batch
PERMS_PER_BATCH = CHUNK // (N_FACE_DRAWS * PERM_GROUPINGS)   # 2

TAG = f'win{WIN_WIDTH}_{DECODE_SFREQ}hz'
ACC = DATA / f'step8_accuracy_{TAG}.npz'
RESULTS = DATA / f'step8_perm_{TAG}'
NULL = DATA / f'step8_null_{TAG}.npz'
FIG = FIGS / f'step8_permutation_{TAG}.png'
LOG_EVERY = 10


def say(msg):
    log(msg, RESULTS)


def make_windows(X, times, width=WIN_WIDTH, step=WIN_STEP):
    """(trials, electrodes, time points) -> (trials, electrodes x width, windows).

    Window w holds time points start .. start + width - 1; its values are
    electrode 1 at those points, then electrode 2, and so on. Its time is the
    mean of those points (the centre).
    """
    n_trials, n_ch, n_times = X.shape
    starts = np.arange(0, n_times - width + 1, step)
    X_win = np.stack([X[:, :, s:s + width].reshape(n_trials, n_ch * width)
                      for s in starts], axis=2)
    win_times = np.array([times[s:s + width].mean() for s in starts])
    return X_win, win_times, starts


def load_windows():
    xy = np.load(DATA / f'step4_Xy_{DECODE_SFREQ}hz.npz')
    X_win, win_times, _ = make_windows(xy['X'], xy['times'])
    return X_win, xy['y'], win_times


def main_analysis(X_win, win_times):
    """The 2000 curves of the main analysis, with the step 4 groups."""
    groups = np.load(DATA / f'step4_draws_{DECODE_SFREQ}hz.npz')['groups']
    assert groups.shape[:2] == (N_FACE_DRAWS, N_GROUPINGS), groups.shape
    t0 = time.time()
    acc = decode_curves_gpu(X_win, groups, chunk=CHUNK)     # (draw, grouping, window)
    mean_curve = acc.mean(axis=(0, 1))
    draw_means = acc.mean(axis=1)
    np.savez(ACC, acc=acc, mean_curve=mean_curve, draw_means=draw_means,
             sd_within=acc.std(axis=1, ddof=1).mean(axis=0),
             sd_between=draw_means.std(axis=0, ddof=1), times=win_times,
             win_width=WIN_WIDTH, win_step=WIN_STEP)
    t_ms = win_times * 1000
    post, pre = t_ms >= 0, t_ms < 0
    i_peak = np.where(post)[0][mean_curve[post].argmax()]
    say(f'main analysis: {acc.shape[0] * acc.shape[1]} curves x {len(win_times)} '
        f'windows in {timedelta(seconds=round(time.time() - t0))}; peak '
        f'{mean_curve[i_peak]:.3f} at {t_ms[i_peak]:.0f} ms (window centre); '
        f'before stimulus mean {mean_curve[pre].mean():.3f}')
    say(f'Saved: data/{ACC.name}')


def permutations(X_win, y):
    """1000 permutations, step 5/6 random streams, each saved to its own file."""
    todo = [p for p in range(N_PERM) if not perm_file(p, RESULTS).exists()]
    say(f'start: {len(todo)} of {N_PERM} permutations to do '
        f'({N_PERM - len(todo)} already saved), {N_FACE_DRAWS} x {PERM_GROUPINGS} '
        f'curves each, {PERMS_PER_BATCH} permutations per GPU batch')
    t0 = time.time()
    n_done = 0
    for b in range(0, len(todo), PERMS_PER_BATCH):
        batch = todo[b:b + PERMS_PER_BATCH]
        groups = np.stack([perm_groups(y, p) for p in batch])   # (perm, 20, 10, ...)
        acc = decode_curves_gpu(X_win, groups, chunk=CHUNK)      # (perm, 20, 10, window)
        for p, acc_p in zip(batch, acc):
            tmp = perm_file(p, RESULTS).with_suffix('.tmp.npy')
            np.save(tmp, acc_p.mean(axis=1))                     # 20 draw means
            tmp.replace(perm_file(p, RESULTS))                   # only complete files count
        before, n_done = n_done, n_done + len(batch)
        if n_done // LOG_EVERY > before // LOG_EVERY or n_done == len(todo):
            elapsed = time.time() - t0
            eta = elapsed / n_done * (len(todo) - n_done)
            say(f'  {n_done}/{len(todo)} permutations done, '
                f'elapsed {timedelta(seconds=round(elapsed))}, '
                f'about {timedelta(seconds=round(eta))} left')
    say('all permutations done')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary-only', action='store_true',
                        help='only summarise the permutations finished so far')
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if not args.summary_only:
        keep_awake()
        X_win, y, win_times = load_windows()
        say(f'windows: {WIN_WIDTH} points ({WIN_WIDTH * 1000 // DECODE_SFREQ} ms), '
            f'step {WIN_STEP}; X {X_win.shape} (trials x values x windows), '
            f'centres {win_times[0] * 1000:.0f} to {win_times[-1] * 1000:.0f} ms')
        # the main analysis is only done once; delete its file to redo it
        if not ACC.exists():
            main_analysis(X_win, win_times)
        permutations(X_win, y)
    # FDR over all windows, including those centred before the stimulus
    summarise(results=RESULTS, real_path=ACC, null_path=NULL, fig_path=FIG)


if __name__ == '__main__':
    main()
