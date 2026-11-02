"""Step 2 check: does ICA003 follow horizontal eye movements?

Finds the 3 largest sustained HEOG shifts in the 0.1-30 Hz data and plots
HEOG next to the ICA003 time course around them.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np

ROOT = Path(__file__).resolve().parent
D = ROOT / 'data'
raw = mne.io.read_raw_fif(D / 'step2_lp-raw.fif', preload=True, verbose=False)
ica = mne.preprocessing.read_ica(D / 'step2_ica-ica.fif', verbose=False)

# HEOG at 0.1-30 Hz, smoothed to 0.2 s so we find sustained gaze shifts, not EMG
sf = raw.info['sfreq']
heog = raw.get_data(picks='HEOG')[0] * 1e6
k = int(0.2 * sf)
sm = np.convolve(heog, np.ones(k) / k, mode='same')
step = np.abs(sm[int(0.3 * sf):] - sm[:-int(0.3 * sf)])  # change over 0.3 s
order = np.argsort(step)[::-1]
picks_t = []
for i in order:
    t = i / sf
    if 20 < t < raw.times[-1] - 20 and all(abs(t - p) > 10 for p in picks_t):
        picks_t.append(t)
    if len(picks_t) == 3:
        break
print('largest HEOG shifts at (s):', [round(t, 1) for t in picks_t])

src = ica.get_sources(raw).get_data(picks=[0, 3])
corr_lp = [np.corrcoef(src[i], heog)[0, 1] for i in range(2)]
print('corr with HEOG on 0.1-30 Hz data: ICA000 %.2f, ICA003 %.2f' % tuple(corr_lp))
# correlation after smoothing (low-frequency part only, < ~5 Hz)
sm3 = np.convolve(src[1], np.ones(k) / k, mode='same')
print('corr smoothed (0.2 s) HEOG vs ICA003: %.2f' % np.corrcoef(sm3, sm)[0, 1])

fig, axes = plt.subplots(2, 3, figsize=(14, 4.5), sharex='col')
for j, t in enumerate(picks_t):
    a, b = int((t - 4) * sf), int((t + 4) * sf)
    tt = raw.times[a:b]
    axes[0, j].plot(tt, heog[a:b], color='#c2385a', lw=0.8)
    axes[0, j].set_title(f'HEOG around {t:.1f} s')
    axes[1, j].plot(tt, src[1, a:b], color='#6a4dd6', lw=0.8)
    axes[1, j].set_title('ICA003')
    axes[1, j].set_xlabel('time (s)')
fig.tight_layout()
out = ROOT / 'figs' / 'step2_ica_ICA003_vs_HEOG.png'
fig.savefig(out, dpi=100)
print('saved', out)

