"""Step 4: run the decoding and summarise it, ERP CORE N170, sub-001.

Needs step4_decoding.py to have been run first (X, y and the groups).

4.5 Decode every grouping: 20 face draws x 100 groupings = 2000 accuracy
    curves. Draws run in parallel; each finished draw is saved to its own
    file, so an interrupted run picks up where it stopped.
4.6 Main result: mean of the 2000 curves at every time point
4.7 Robustness: spread across groupings within a draw, and across the 20
    draw means (stability only, not an error bar)
4.8 Figures: main curve; the two kinds of spread

Run:  .venv\\Scripts\\python.exe step4_run.py
Progress: data/step4_results_100hz/progress.log
"""
import os
import time
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from step4_lib import (DATA, DECODE_SFREQ, FIGS, N170_WIN, N_FACE_DRAWS,
                       N_GROUPINGS, decode_curve, keep_awake)

N_JOBS = max(1, (os.cpu_count() or 2) // 2)   # one per physical core
RESULTS = DATA / f'step4_results_{DECODE_SFREQ}hz'
LOG = RESULTS / 'progress.log'


def log(msg):
    line = f'{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}'
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def draw_file(d):
    return RESULTS / f'draw_{d:02d}.npy'


def run_draw(X, groups_d, d):
    """Decode the 100 groupings of one face draw and save them."""
    acc = np.array([decode_curve(X, groups_d[g, 0], groups_d[g, 1])
                    for g in range(len(groups_d))])        # (groupings, time)
    tmp = draw_file(d).with_suffix('.tmp.npy')
    np.save(tmp, acc)
    tmp.replace(draw_file(d))      # only complete files get the final name
    return d


def main():
    RESULTS.mkdir(exist_ok=True)
    keep_awake()
    xy = np.load(DATA / f'step4_Xy_{DECODE_SFREQ}hz.npz')
    X, times = xy['X'], xy['times']
    groups = np.load(DATA / f'step4_draws_{DECODE_SFREQ}hz.npz')['groups']
    assert groups.shape[:2] == (N_FACE_DRAWS, N_GROUPINGS), groups.shape

    # --- 4.5 Decode every grouping -------------------------------------------
    todo = [d for d in range(N_FACE_DRAWS) if not draw_file(d).exists()]
    log(f'4.5 start: {len(todo)} of {N_FACE_DRAWS} draws to do '
        f'({N_FACE_DRAWS - len(todo)} already saved), {N_JOBS} parallel jobs')
    t0 = time.time()
    jobs = (delayed(run_draw)(X, groups[d], d) for d in todo)
    for n_done, d in enumerate(Parallel(n_jobs=N_JOBS,
                                        return_as='generator_unordered')(jobs), 1):
        elapsed = time.time() - t0
        eta = elapsed / n_done * (len(todo) - n_done)
        log(f'  draw {d + 1:2d} done ({n_done}/{len(todo)}), '
            f'elapsed {timedelta(seconds=round(elapsed))}, '
            f'about {timedelta(seconds=round(eta))} left')
    log('4.5 done')

    # --- 4.6 Main result: mean of all 2000 curves ----------------------------
    acc = np.stack([np.load(draw_file(d)) for d in range(N_FACE_DRAWS)])
    # acc[draw, grouping, time]
    mean_curve = acc.mean(axis=(0, 1))
    t_ms = times * 1000
    post = t_ms >= 0
    i_peak = np.where(post)[0][mean_curve[post].argmax()]
    log(f'4.6 main result: {acc.shape[0] * acc.shape[1]} curves; peak accuracy '
        f'{mean_curve[i_peak]:.3f} at {t_ms[i_peak]:.0f} ms')
    # Self-check: before the stimulus there is nothing to decode, so the
    # main curve should sit around 0.5 there
    pre = t_ms < 0
    i_pre = np.where(pre)[0][mean_curve[pre].argmax()]
    log(f'4.6 self-check, before stimulus (-200 to 0 ms, {pre.sum()} points): '
        f'mean {mean_curve[pre].mean():.3f}, max {mean_curve[i_pre]:.3f} '
        f'at {t_ms[i_pre]:.0f} ms (should be near 0.500)')

    # --- 4.7 Robustness: two kinds of random spread --------------------------
    # Within a draw: SD across its 100 groupings, averaged over the 20 draws.
    # Across draws: each draw's mean over its groupings, then SD across 20.
    # These show stability only; all 2000 curves reuse the same data, so the
    # spread is much smaller than the real uncertainty. Significance comes
    # from the permutation test.
    sd_within = acc.std(axis=1, ddof=1).mean(axis=0)        # (time,)
    draw_means = acc.mean(axis=1)                            # (draw, time)
    sd_between = draw_means.std(axis=0, ddof=1)              # (time,)
    peak_idx = np.where(post)[0][draw_means[:, post].argmax(axis=1)]
    log(f'4.7 spread within a draw (SD over groupings): mean {sd_within.mean():.3f}, '
        f'max {sd_within.max():.3f}')
    log(f'4.7 spread across draws (SD of the 20 draw means): mean '
        f'{sd_between.mean():.3f}, max {sd_between.max():.3f}')
    log(f'4.7 peak of each draw mean: time {t_ms[peak_idx].min():.0f}-'
        f'{t_ms[peak_idx].max():.0f} ms, accuracy '
        f'{draw_means[np.arange(N_FACE_DRAWS), peak_idx].min():.3f}-'
        f'{draw_means[np.arange(N_FACE_DRAWS), peak_idx].max():.3f}')
    for d in range(N_FACE_DRAWS):
        log(f'    draw {d + 1:2d}: peak {draw_means[d, peak_idx[d]]:.3f} '
            f'at {t_ms[peak_idx[d]]:.0f} ms')

    np.savez(DATA / f'step4_accuracy_{DECODE_SFREQ}hz.npz', acc=acc,
             mean_curve=mean_curve, draw_means=draw_means, sd_within=sd_within,
             sd_between=sd_between, times=times)
    log(f'Saved: data/step4_accuracy_{DECODE_SFREQ}hz.npz')

    # --- 4.8 Figures ------------------------------------------------------------
    def references(ax):
        ax.axhline(0.5, color='k', lw=0.8, ls='--')
        ax.axvline(0, color='k', lw=0.8)
        ax.axvspan(N170_WIN[0] * 1000, N170_WIN[1] * 1000, color='#f2d98c',
                   alpha=0.5, zorder=0)
        ax.set_xlim(t_ms[0], t_ms[-1])
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('accuracy')

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t_ms, mean_curve, color='#6a4dd6', lw=2)
    references(ax)
    ax.set_title(f'Face vs car decoding, mean of {acc.shape[0] * acc.shape[1]} '
                 f'curves (dashed = 50% chance, yellow = 110-150 ms)')
    fig.tight_layout()
    fig.savefig(FIGS / f'step4_accuracy_{DECODE_SFREQ}hz.png', dpi=100)

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), sharey=True)
    axes[0].plot(t_ms, acc[0].T, color='0.6', lw=0.4, alpha=0.3)
    axes[0].plot(t_ms, draw_means[0], color='#6a4dd6', lw=2, label='mean of draw 1')
    axes[0].set_title(f'Within one draw: its {N_GROUPINGS} groupings (draw 1)')
    axes[1].plot(t_ms, draw_means.T, color='0.5', lw=0.8, alpha=0.6)
    axes[1].plot(t_ms, mean_curve, color='#6a4dd6', lw=2, label='mean of all')
    axes[1].set_title(f'Across draws: the {N_FACE_DRAWS} draw means')
    for ax in axes:
        references(ax)
        ax.legend(loc='upper right', fontsize=8)
    fig.suptitle('Spread from random draws and groupings: stability only, '
                 'not a confidence interval')
    fig.tight_layout()
    fig.savefig(FIGS / f'step4_variability_{DECODE_SFREQ}hz.png', dpi=100)
    plt.close('all')
    log(f'Saved: figs/step4_accuracy_{DECODE_SFREQ}hz.png, '
        f'figs/step4_variability_{DECODE_SFREQ}hz.png')


if __name__ == '__main__':
    main()
