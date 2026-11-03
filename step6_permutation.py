"""Step 6: the step 5 permutation test, with the GPU classifier.

Same design and the same random streams as step5_permutation.py (1000
permutations, trial-level shuffle, 20 draws x 10 groupings each, FDR over all
time points), so permutation p here shuffles the labels exactly as
permutation p there. Only the classifier differs: decode_curves_gpu.
The real curve is the GPU main curve from step6_main.py (run that first).

Several permutations go to the GPU together (PERMS_PER_BATCH x 200 curves);
each one is still saved to its own file, so an interrupted run picks up where
it stopped.

Run:      .venv\\Scripts\\python.exe step6_permutation.py   (about 1.5 h)
Summary:  .venv\\Scripts\\python.exe step6_permutation.py --summary-only
Progress: data/step6_perm_gpu_100hz/progress.log
"""
import argparse
import time
from datetime import timedelta

import numpy as np

from step4_lib import (DATA, DECODE_SFREQ, FIGS, N_FACE_DRAWS, keep_awake,
                       make_groups, permutation_seeds, shuffle_labels)
from step5_permutation import (N_PERM, PERM_GROUPINGS, log, perm_file,
                               summarise)
from step6_gpu_lib import CHUNK, decode_curves_gpu

RESULTS = DATA / f'step6_perm_gpu_{DECODE_SFREQ}hz'
PERMS_PER_BATCH = CHUNK // (N_FACE_DRAWS * PERM_GROUPINGS)   # 400 / 200 = 2
LOG_EVERY = 10


def perm_groups(y, p):
    """The groups of permutation p, exactly as in step5_permutation.run_perm."""
    shuffle_seed, draw_seeds = permutation_seeds(p)
    y_perm = shuffle_labels(y, shuffle_seed)
    _, groups = make_groups(y_perm, draw_seeds, n_groupings=PERM_GROUPINGS)
    return groups                                   # (20, 10, 2, 12, 4)


def run():
    xy = np.load(DATA / f'step4_Xy_{DECODE_SFREQ}hz.npz')
    X, y = xy['X'], xy['y']
    todo = [p for p in range(N_PERM) if not perm_file(p, RESULTS).exists()]
    log(f'start (GPU): {len(todo)} of {N_PERM} permutations to do '
        f'({N_PERM - len(todo)} already saved), {N_FACE_DRAWS} x {PERM_GROUPINGS} '
        f'curves each, {PERMS_PER_BATCH} permutations per GPU batch', RESULTS)
    t0 = time.time()
    n_done = 0
    for b in range(0, len(todo), PERMS_PER_BATCH):
        batch = todo[b:b + PERMS_PER_BATCH]
        groups = np.stack([perm_groups(y, p) for p in batch])   # (perm, 20, 10, ...)
        acc = decode_curves_gpu(X, groups)                       # (perm, 20, 10, time)
        for p, acc_p in zip(batch, acc):
            # as in step 5: save the 20 draw means (draws x time)
            tmp = perm_file(p, RESULTS).with_suffix('.tmp.npy')
            np.save(tmp, acc_p.mean(axis=1))
            tmp.replace(perm_file(p, RESULTS))      # only complete files count
        before, n_done = n_done, n_done + len(batch)
        if n_done // LOG_EVERY > before // LOG_EVERY or n_done == len(todo):
            elapsed = time.time() - t0
            eta = elapsed / n_done * (len(todo) - n_done)
            log(f'  {n_done}/{len(todo)} permutations done, '
                f'elapsed {timedelta(seconds=round(elapsed))}, '
                f'about {timedelta(seconds=round(eta))} left', RESULTS)
    log('all permutations done', RESULTS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary-only', action='store_true',
                        help='only summarise the permutations finished so far')
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if not args.summary_only:
        keep_awake()
        run()
    summarise(results=RESULTS,
              real_path=DATA / f'step6_accuracy_gpu_{DECODE_SFREQ}hz.npz',
              null_path=DATA / f'step6_null_gpu_{DECODE_SFREQ}hz.npz',
              fig_path=FIGS / f'step6_permutation_gpu_{DECODE_SFREQ}hz.png')


if __name__ == '__main__':
    main()
