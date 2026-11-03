"""Step 6 check: does the GPU classifier give the same answers as sklearn's SVC?

Uses the first N_CHECK groupings of face draw 1 from the main analysis
(N_CHECK x 100 time points x 12 folds SVMs, 2 test traces each).
  1. sklearn: decision values of SVC, the same loop as decode_curve
  2. GPU: decode_curves_gpu with decision values
  3. compare predictions, decision values, and the accuracy curves (also
     against the curves saved by step4_run.py)
  4. time the GPU on one permutation's worth of curves (200)

Run: .venv\\Scripts\\python.exe step6_check.py
"""
import time

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from step4_lib import DATA, DECODE_SFREQ, SVM_C
from step6_gpu_lib import decode_curves_gpu

N_CHECK = 50


def sklearn_decision(X, face_groups, car_groups):
    """Like decode_curve, but returns the decision values (time, fold, 2)."""
    face = X[face_groups].mean(axis=1)
    car = X[car_groups].mean(axis=1)
    n_traces, _, n_times = face.shape
    y_train = np.r_[np.ones(n_traces - 1), np.zeros(n_traces - 1)]
    dec = np.empty((n_times, n_traces, 2))
    for t in range(n_times):
        f, c = face[:, :, t], car[:, :, t]
        for k in range(n_traces):
            train = np.r_[np.delete(f, k, axis=0), np.delete(c, k, axis=0)]
            test = np.stack([f[k], c[k]])
            scaler = StandardScaler().fit(train)
            clf = SVC(kernel='linear', C=SVM_C).fit(scaler.transform(train), y_train)
            dec[t, k] = clf.decision_function(scaler.transform(test))
    return dec


xy = np.load(DATA / f'step4_Xy_{DECODE_SFREQ}hz.npz')
X = xy['X']
groups = np.load(DATA / f'step4_draws_{DECODE_SFREQ}hz.npz')['groups']
saved_acc = np.load(DATA / f'step4_results_{DECODE_SFREQ}hz' / 'draw_00.npy')
print(f'GPU: {torch.cuda.get_device_name(0)}, PyTorch {torch.__version__}')

check = groups[0, :N_CHECK]                     # (N_CHECK, 2, 12, 4)
n_svm = N_CHECK * X.shape[2] * check.shape[2]
print(f'Comparing {N_CHECK} groupings = {n_svm} SVMs, {2 * n_svm} test predictions')

t0 = time.time()
dec_sk = np.stack([sklearn_decision(X, g[0], g[1]) for g in check])
print(f'1. sklearn done in {time.time() - t0:.1f} s')

t0 = time.time()
acc_gpu, dec_gpu = decode_curves_gpu(X, check, return_decision=True)
torch.cuda.synchronize()
print(f'2. GPU done in {time.time() - t0:.1f} s (first call, includes start-up)')

# 3. Compare. sklearn: positive decision = class 1 = face; same on the GPU
truth = np.array([1, -1])
pred_sk = np.where(dec_sk > 0, 1, -1)
pred_gpu = np.where(dec_gpu > 0, 1, -1)
agree = pred_sk == pred_gpu
print(f'3. Predictions agree: {agree.sum()} of {agree.size} '
      f'({agree.mean() * 100:.4f}%)')
if not agree.all():
    print(f'   |decision| where they disagree: max {np.abs(dec_sk[~agree]).max():.2e} '
          f'(sklearn), {np.abs(dec_gpu[~agree]).max():.2e} (GPU)')
d = np.abs(dec_sk - dec_gpu)
print(f'   Decision values: max difference {d.max():.2e}, median {np.median(d):.2e}; '
      f'typical size |decision| median {np.median(np.abs(dec_sk)):.2f}')
acc_sk = (pred_sk == truth).mean(axis=(2, 3))
print(f'   Accuracy curves: GPU vs sklearn max difference '
      f'{np.abs(acc_gpu - acc_sk).max():.4f}; sklearn here vs saved step 4 '
      f'{np.abs(acc_sk - saved_acc[:N_CHECK]).max():.4f}; GPU vs saved step 4 '
      f'{np.abs(acc_gpu - saved_acc[:N_CHECK]).max():.4f}')
print(f'   Mean curve of the {N_CHECK}: max difference GPU vs sklearn '
      f'{np.abs(acc_gpu.mean(0) - acc_sk.mean(0)).max():.4f}')

# 4. Speed: one permutation = 20 draws x 10 groupings = 200 curves
batch = groups[:2].reshape(-1, *groups.shape[2:])   # 200 groupings
t0 = time.time()
decode_curves_gpu(X, batch)
torch.cuda.synchronize()
sec = time.time() - t0
print(f'4. GPU, 200 curves: {sec:.1f} s -> 1000 permutations about '
      f'{sec * 1000 / 3600:.1f} h, main analysis (2000 curves) about {sec * 10 / 60:.1f} min')
