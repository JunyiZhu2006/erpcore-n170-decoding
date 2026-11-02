"""Step 4 shared settings and functions (used by step4_decoding.py,
step4_run.py and step5_permutation.py).

make_groups:  labels -> which trials form which face / car trace
decode_curve: those groups -> accuracy at every time point

decode_curve never sees the labels, only the groups. The main analysis builds
the groups from the true labels; a permutation builds them from shuffled
trial labels. Both then go through exactly the same decoding.
"""
import ctypes
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
FIGS = ROOT / 'figs'

# Sampling rate for decoding; the data were low-passed at 30 Hz in step 2,
# below the 50 Hz Nyquist limit, so no aliasing
DECODE_SFREQ = 100
# Seed for every random draw in step 4, so results are reproducible
RANDOM_SEED = 42
# Repetitions, fixed in advance (see notes.md): 20 face draws x 100 groupings
N_FACE_DRAWS = 20
N_GROUPINGS = 100
TRIALS_PER_TRACE = 4
# Linear SVM with scikit-learn's default regularisation, not tuned
SVM_C = 1.0
# N170 window used by ERP CORE, for reference lines in figures only
N170_WIN = (0.110, 0.150)


def main_seeds():
    """One independent random stream per face draw of the main analysis."""
    return np.random.SeedSequence(RANDOM_SEED).spawn(N_FACE_DRAWS)


# The permutation test gets its own random source, separate from main_seeds:
# entropy [RANDOM_SEED, 1] instead of RANDOM_SEED
PERM_SEED_TAG = 1


def permutation_seeds(perm):
    """Random streams for permutation number perm (0-based).

    Returns (shuffle_seed, draw_seeds): one stream to shuffle the trial
    labels, and one per face draw for make_groups. Each permutation, and each
    draw within it, can be recomputed on its own.
    """
    root = np.random.SeedSequence([RANDOM_SEED, PERM_SEED_TAG])
    perm_seed = root.spawn(perm + 1)[perm]
    shuffle_seed, groups_seed = perm_seed.spawn(2)
    return shuffle_seed, groups_seed.spawn(N_FACE_DRAWS)


def shuffle_labels(y, shuffle_seed):
    """Trial-level permutation: same labels (60 faces, 48 cars), new order."""
    return np.random.default_rng(shuffle_seed).permutation(y)


def make_groups(y, seeds, n_groupings=N_GROUPINGS, per_trace=TRIALS_PER_TRACE):
    """Draw faces and group trials into traces, from the labels y.

    For each seed (one per face draw): pick as many face trials as there are
    car trials, then n_groupings times shuffle each class and cut it into
    groups of per_trace trials.

    Returns
    -------
    face_draws : (n_draws, n_per_class) indices of the faces kept per draw
    groups : (n_draws, n_groupings, 2, n_traces, per_trace) trial indices;
        axis 2 is the class, 0 = face, 1 = car
    """
    face_idx = np.where(y == 1)[0]
    car_idx = np.where(y == 0)[0]
    assert len(face_idx) >= len(car_idx), 'expects at least as many faces as cars'
    n_per_class = len(car_idx)
    n_traces = n_per_class // per_trace
    face_draws = np.empty((len(seeds), n_per_class), dtype=int)
    groups = np.empty((len(seeds), n_groupings, 2, n_traces, per_trace), dtype=int)
    for d, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        face_draws[d] = np.sort(rng.choice(face_idx, size=n_per_class, replace=False))
        for g in range(n_groupings):
            groups[d, g, 0] = rng.permutation(face_draws[d]).reshape(-1, per_trace)
            groups[d, g, 1] = rng.permutation(car_idx).reshape(-1, per_trace)
    return face_draws, groups


def decode_curve(X, face_groups, car_groups, svm_c=SVM_C):
    """Accuracy at every time point for one set of traces.

    X : (n_trials, n_channels, n_times)
    face_groups, car_groups : (n_traces, per_trace) trial indices

    At each time point, each trace is 30 numbers (one per electrode). Fold k
    tests face trace k and car trace k and trains on the other 11 + 11:
    standardise with the training traces only, fit a linear SVM, predict the
    two test traces (score 1, 0.5 or 0). Accuracy = mean over the folds.
    """
    face = X[face_groups].mean(axis=1)            # (n_traces, ch, time)
    car = X[car_groups].mean(axis=1)
    n_traces, _, n_times = face.shape
    y_train = np.r_[np.ones(n_traces - 1), np.zeros(n_traces - 1)]
    y_test = np.array([1, 0])
    acc = np.empty(n_times)
    for t in range(n_times):
        f, c = face[:, :, t], car[:, :, t]
        score = 0.0
        for k in range(n_traces):
            train = np.r_[np.delete(f, k, axis=0), np.delete(c, k, axis=0)]
            test = np.stack([f[k], c[k]])
            scaler = StandardScaler().fit(train)
            clf = SVC(kernel='linear', C=svm_c).fit(scaler.transform(train), y_train)
            score += (clf.predict(scaler.transform(test)) == y_test).mean()
        acc[t] = score / n_traces
    return acc


def keep_awake():
    """Stop Windows from sleeping while this process runs (no-op elsewhere).

    The setting ends by itself when the process exits.
    """
    if sys.platform == 'win32':
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
