"""Step 6: the step 4 main analysis again, with the GPU classifier.

Same 20 face draws x 100 groupings as step 4 (data/step4_draws_100hz.npz),
decoded with decode_curves_gpu instead of sklearn. The main curve and the
permutation null (step6_permutation.py) then both come from the same GPU
classifier.

Saves data/step6_accuracy_gpu_100hz.npz with the same fields as
data/step4_accuracy_100hz.npz, and compares the two.

Run: .venv\\Scripts\\python.exe step6_main.py   (about 1 minute)
"""
import time

import numpy as np
import torch

from step4_lib import DATA, DECODE_SFREQ, N_FACE_DRAWS, N_GROUPINGS
from step6_gpu_lib import decode_curves_gpu

OUT = DATA / f'step6_accuracy_gpu_{DECODE_SFREQ}hz.npz'

xy = np.load(DATA / f'step4_Xy_{DECODE_SFREQ}hz.npz')
X, times = xy['X'], xy['times']
groups = np.load(DATA / f'step4_draws_{DECODE_SFREQ}hz.npz')['groups']
assert groups.shape[:2] == (N_FACE_DRAWS, N_GROUPINGS), groups.shape
print(f'GPU: {torch.cuda.get_device_name(0)}')

t0 = time.time()
acc = decode_curves_gpu(X, groups)                  # (draw, grouping, time)
print(f'Decoded {acc.shape[0] * acc.shape[1]} curves in {time.time() - t0:.0f} s')

# Same summaries as step 4.6-4.7
mean_curve = acc.mean(axis=(0, 1))
draw_means = acc.mean(axis=1)
sd_within = acc.std(axis=1, ddof=1).mean(axis=0)
sd_between = draw_means.std(axis=0, ddof=1)
np.savez(OUT, acc=acc, mean_curve=mean_curve, draw_means=draw_means,
         sd_within=sd_within, sd_between=sd_between, times=times)
print(f'Saved: data/{OUT.name}')

t_ms = times * 1000
pre, post = t_ms < 0, t_ms >= 0
i_peak = np.where(post)[0][mean_curve[post].argmax()]
print(f'GPU main curve: peak {mean_curve[i_peak]:.3f} at {t_ms[i_peak]:.0f} ms; '
      f'before stimulus mean {mean_curve[pre].mean():.3f}')

# Compare with the sklearn (CPU) result of step 4
cpu = np.load(DATA / f'step4_accuracy_{DECODE_SFREQ}hz.npz')
cpu_curve = cpu['mean_curve']
j_peak = np.where(post)[0][cpu_curve[post].argmax()]
d = np.abs(mean_curve - cpu_curve)
print(f'CPU main curve: peak {cpu_curve[j_peak]:.3f} at {t_ms[j_peak]:.0f} ms; '
      f'before stimulus mean {cpu_curve[pre].mean():.3f}')
print(f'GPU vs CPU main curve: max difference {d.max():.4f} at {t_ms[d.argmax()]:.0f} ms, '
      f'mean {d.mean():.4f}')
print(f'GPU vs CPU single curves: identical {np.mean(acc == cpu["acc"]) * 100:.2f}% '
      f'of curve points, max difference {np.abs(acc - cpu["acc"]).max():.4f}')
