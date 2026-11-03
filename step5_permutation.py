"""Step 5: permutation test for the step 4 decoding, ERP CORE N170, sub-001.

Each permutation:
  1. shuffle the 108 trial labels (still 60 faces, 48 cars)
  2. with the shuffled labels, draw faces and group into traces again
     (make_groups, 20 draws x 10 groupings)
  3. decode with the same decode_curve, and average the 200 curves
This gives, at every time point, what the main curve looks like when there is
no real face/car information. 1000 permutations.

Why 1000 x (20 x 10) and not 100 x (20 x 100) like the main analysis: with N
permutations the smallest p is 1/(1+N), and FDR over 100 time points needs
the k-th smallest p <= 0.05 * k / 100. With 100 permutations nothing could
pass unless 20+ time points were all at the minimum p. Fewer curves per
permutation make the null wider, so the test is if anything conservative.

Each permutation is one task saved to its own file, so an interrupted run
picks up where it stopped. Results can be summarised at any time from the
finished permutations.

Run:      .venv\\Scripts\\python.exe step5_permutation.py [--jobs 8]
Summary:  .venv\\Scripts\\python.exe step5_permutation.py --summary-only
Progress: data/step5_perm_100hz/progress.log
"""
import argparse
import os
import time
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from mne.stats import fdr_correction

from step4_lib import (DATA, DECODE_SFREQ, FIGS, N170_WIN, N_FACE_DRAWS,
                       decode_curve, keep_awake, make_groups,
                       permutation_seeds, shuffle_labels)

N_PERM = 1000
# Groupings per face draw within one permutation (main analysis: 100)
PERM_GROUPINGS = 10
# Multiple comparisons: FDR (Benjamini-Hochberg) over all 100 time points
FDR_Q = 0.05
RESULTS = DATA / f'step5_perm_{DECODE_SFREQ}hz'
LOG = RESULTS / 'progress.log'
LOG_EVERY = 10   # write a progress line every this many permutations


def log(msg, results=RESULTS):
    line = f'{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}'
    print(line, flush=True)
    with open(results / LOG.name, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def perm_file(p, results=RESULTS):
    return results / f'perm_{p:04d}.npy'


def run_perm(X, y, p):
    """Permutation p: shuffle labels, regroup, decode 20 x 10 curves.

    Saves the 20 draw means (draws x time); the permutation's curve is their
    mean.
    """
    shuffle_seed, draw_seeds = permutation_seeds(p)
    y_perm = shuffle_labels(y, shuffle_seed)
    _, groups = make_groups(y_perm, draw_seeds, n_groupings=PERM_GROUPINGS)
    draw_means = np.array([np.mean([decode_curve(X, g[0], g[1]) for g in groups_d],
                                   axis=0) for groups_d in groups])
    tmp = perm_file(p).with_suffix('.tmp.npy')
    np.save(tmp, draw_means)
    tmp.replace(perm_file(p))           # only complete files count
    return p


def finished_perms(results=RESULTS):
    return [p for p in range(N_PERM) if perm_file(p, results).exists()]


def run(n_jobs):
    xy = np.load(DATA / f'step4_Xy_{DECODE_SFREQ}hz.npz')
    X, y = xy['X'], xy['y']
    todo = [p for p in range(N_PERM) if not perm_file(p).exists()]
    log(f'start: {len(todo)} of {N_PERM} permutations to do '
        f'({N_PERM - len(todo)} already saved), {N_FACE_DRAWS} x {PERM_GROUPINGS} '
        f'curves each, {n_jobs} parallel jobs')
    t0 = time.time()
    jobs = (delayed(run_perm)(X, y, p) for p in todo)
    for n_done, _ in enumerate(Parallel(n_jobs=n_jobs,
                                        return_as='generator_unordered')(jobs), 1):
        if n_done % LOG_EVERY == 0 or n_done == len(todo):
            elapsed = time.time() - t0
            eta = elapsed / n_done * (len(todo) - n_done)
            log(f'  {n_done}/{len(todo)} permutations done, '
                f'elapsed {timedelta(seconds=round(elapsed))}, '
                f'about {timedelta(seconds=round(eta))} left')
    log('all permutations done')


def summarise(results=RESULTS,
              real_path=DATA / f'step4_accuracy_{DECODE_SFREQ}hz.npz',
              null_path=DATA / f'step5_null_{DECODE_SFREQ}hz.npz',
              fig_path=FIGS / f'step5_permutation_{DECODE_SFREQ}hz.png'):
    """p values, FDR, figure. The defaults are step 5's files; step 6 (GPU)
    passes its own."""
    def say(msg):
        log(msg, results)

    perms = finished_perms(results)
    if not perms:
        say('summary: no finished permutations yet')
        return
    real = np.load(real_path)
    real_curve, times = real['mean_curve'], real['times']
    t_ms = times * 1000
    # null[perm, time]: mean of the permutation's 200 curves
    null = np.stack([np.load(perm_file(p, results)).mean(axis=0) for p in perms])
    n = len(null)
    # p at every time point: share of permutations at least as high as the
    # real curve, counting the real result itself (+1), uncorrected
    p_val = (1 + (null >= real_curve).sum(axis=0)) / (1 + n)
    # FDR (Benjamini-Hochberg), q = 0.05, over all time points together,
    # including the pre-stimulus ones: after correction none of those should
    # be significant (self-check)
    sig_fdr, p_fdr = fdr_correction(p_val, alpha=FDR_Q, method='indep')
    null_mean = null.mean(axis=0)               # actual chance level of this pipeline
    null_p95 = np.percentile(null, 95, axis=0)  # how high chance alone gets (one-sided)
    np.savez(null_path, null=null, perms=np.array(perms),
             real_curve=real_curve, p_val=p_val, p_fdr=p_fdr, sig_fdr=sig_fdr,
             null_mean=null_mean, null_p95=null_p95, times=times)

    pre, post = t_ms < 0, t_ms >= 0
    sig = p_val < 0.05
    say(f'summary from {n} permutations (smallest possible p = {1 / (1 + n):.4f})')
    say(f'  null mean (actual chance level): overall {null_mean.mean():.3f}, before '
        f'stimulus {null_mean[pre].mean():.3f}, 400-790 ms '
        f'{null_mean[t_ms >= 400].mean():.3f}, range {null_mean.min():.3f}-'
        f'{null_mean.max():.3f}')
    say(f'  null 95th percentile: range {null_p95.min():.3f}-{null_p95.max():.3f}')
    say(f'  self-check, before stimulus ({int(pre.sum())} points): FDR-significant '
        f'{int(sig_fdr[pre].sum())} (should be 0); uncorrected p < 0.05 '
        f'{int(sig[pre].sum())}; smallest p {p_val[pre].min():.4f} at '
        f'{t_ms[pre][p_val[pre].argmin()]:.0f} ms')
    say(f'  after stimulus, FDR-significant (q = {FDR_Q}, all {len(p_val)} points): '
        + (', '.join(f'{x:.0f}' for x in t_ms[post & sig_fdr]) or 'none') + ' ms')
    say(f'  after stimulus, uncorrected p < 0.05: '
        + (', '.join(f'{x:.0f}' for x in t_ms[post & sig]) or 'none') + ' ms')
    i_peak = np.where(post)[0][real_curve[post].argmax()]
    say(f'  real peak {real_curve[i_peak]:.3f} at {t_ms[i_peak]:.0f} ms: p = '
        f'{p_val[i_peak]:.4f}, FDR-adjusted {p_fdr[i_peak]:.4f}; null at that time: '
        f'mean {null_mean[i_peak]:.3f}, 95th percentile {null_p95[i_peak]:.3f}')

    fig, (ax, ax_p) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                   gridspec_kw=dict(height_ratios=[3, 1.3]))
    ax.fill_between(t_ms, np.percentile(null, 2.5, axis=0),
                    np.percentile(null, 97.5, axis=0), color='0.85',
                    label=f'null, middle 95% ({n} permutations)')
    ax.plot(t_ms, null_mean, color='0.3', lw=1.5, label='null mean (actual chance level)')
    ax.plot(t_ms, null_p95, color='0.3', lw=1, ls=':', label='null 95th percentile')
    ax.plot(t_ms, real_curve, color='#6a4dd6', lw=2, label='real (mean of 2000)')
    ax.plot(t_ms[sig_fdr], np.full(sig_fdr.sum(), real_curve.max() + 0.03), 'k.',
            label=f'FDR q < {FDR_Q}')
    ax.axhline(0.5, color='k', lw=0.8, ls='--')
    ax.set_ylabel('accuracy')
    ax.legend(fontsize=8, loc='lower left')
    ax.set_title('Face vs car decoding against the permutation null')
    ax_p.plot(t_ms, p_val, color='0.5', lw=1, label='p (uncorrected)')
    ax_p.plot(t_ms, p_fdr, color='k', lw=1, label='p (FDR-adjusted)')
    ax_p.axhline(FDR_Q, color='#c2385a', lw=0.8, ls='--')
    ax_p.set_ylabel('p')
    ax_p.set_xlabel('time (ms)')
    ax_p.legend(fontsize=8, loc='upper right')
    for a in (ax, ax_p):
        a.axvline(0, color='k', lw=0.8)
        a.axvspan(N170_WIN[0] * 1000, N170_WIN[1] * 1000, color='#f2d98c',
                  alpha=0.5, zorder=0)
        a.set_xlim(t_ms[0], t_ms[-1])
    fig.tight_layout()
    fig.savefig(fig_path, dpi=100)
    plt.close('all')
    say(f'Saved: {null_path.parent.name}/{null_path.name}, '
        f'{fig_path.parent.name}/{fig_path.name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help='parallel processes (default: one per physical core)')
    parser.add_argument('--summary-only', action='store_true',
                        help='only summarise the permutations finished so far')
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if not args.summary_only:
        keep_awake()
        run(args.jobs)
    summarise()


if __name__ == '__main__':
    main()
