"""Step 2 check: is Oz a bad channel? Compare its power spectrum with O1 and O2.

Uses the high-passed data (before the 30 Hz low-pass) so the full spectrum is
visible.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import mne

ROOT = Path(__file__).resolve().parent
CHANNELS = ['O1', 'Oz', 'O2']
COLORS = {'O1': '#1c8cc6', 'Oz': '#c2385a', 'O2': '#d9820d'}

raw = mne.io.read_raw_fif(ROOT / 'data' / 'step2_hp-raw.fif', preload=True)
psd = raw.compute_psd(picks=CHANNELS, fmin=0.1, fmax=100)

fig = psd.plot(picks=CHANNELS, spatial_colors=False, show=False)
ax = fig.axes[0]
traces = [ln for ln in ax.get_lines() if len(ln.get_xdata()) == len(psd.freqs)]
for line, ch in zip(traces, psd.ch_names):
    line.set_color(COLORS[ch])
    line.set_linewidth(2 if ch == 'Oz' else 1.2)
    line.set_label(ch)
ax.legend()
ax.set_title('Power spectrum: O1 / Oz / O2 (0.1 Hz high-passed)')
fig.set_size_inches(9, 5)

out = ROOT / 'figs' / 'step2_psd_Oz_O1_O2.png'
fig.savefig(out, dpi=120, bbox_inches='tight')
print(f'Saved: {out.relative_to(ROOT)}')

data = psd.get_data() * 1e12
for lo, hi in [(0.1, 1), (1, 4), (4, 8), (8, 13), (13, 30), (30, 45), (45, 100)]:
    m = (psd.freqs >= lo) & (psd.freqs < hi)
    vals = '  '.join(f'{ch} {data[i, m].mean():9.2f}' for i, ch in enumerate(psd.ch_names))
    print(f'{lo:>5}-{hi:<4} Hz  {vals}')
