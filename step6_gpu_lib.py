"""Step 6: the step 4 decoding on the GPU, ERP CORE N170, sub-001.

Same procedure as decode_curve in step4_lib.py: traces of 4 trials,
leave-one-pair-out folds, standardise with the training traces only, linear
SVM with C = 1, accuracy = mean over the folds. The difference is that many
curves are done at once: every grouping x time point x fold is one small SVM
problem (22 training traces x 30 electrodes), and all of them are solved
together on the GPU.

The SVM solver is libsvm's own algorithm, the one scikit-learn's SVC runs:
SMO with second-order working set selection, stopping tolerance 1e-3, and
the same rule for the intercept. It is only run on many problems in parallel,
so the answers should agree with SVC up to that tolerance. step6_check.py
compares the two.
"""
import numpy as np
import torch

from step4_lib import SVM_C

SVM_TOL = 1e-3       # libsvm / SVC default stopping tolerance
TAU = 1e-12          # libsvm: used when the curvature is not positive
MAX_ITER = 100_000   # safety stop only; libsvm itself has no lower limit
# Groupings per GPU batch (400 x 100 x 12 = 480 000 SVMs, about 7 GB of the
# 16 GB at float64; measured: 200 curves in about 5.5 s)
CHUNK = 400
BUILD_CHUNK = 50     # groupings per step when building the kernels
CHECK_EVERY = 32    # SMO iterations between "all finished?" checks


def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError('no CUDA GPU visible to PyTorch')
    return torch.device('cuda')


def make_folds(X, groups):
    """Traces and leave-one-pair-out folds for a batch of groupings.

    X : (n_trials, n_ch, n_times) tensor
    groups : (n_groupings, 2, n_traces, per_trace) long tensor, class 0 = face

    Returns train (P, 2 * (n_traces - 1), n_ch) and test (P, 2, n_ch), with
    P = n_groupings * n_times * n_traces problems in the order
    (grouping, time, fold). Training rows are the 11 face traces then the 11
    car traces (as in decode_curve); test rows are (face k, car k).
    """
    traces = X[groups].mean(dim=3)                 # (G, 2, K, ch, T)
    n_groupings, _, n_traces, n_ch, n_times = traces.shape
    traces = traces.permute(0, 4, 1, 2, 3)         # (G, T, 2, K, ch)
    keep = torch.tensor([[j for j in range(n_traces) if j != k]
                         for k in range(n_traces)], device=X.device)
    train = traces[:, :, :, keep]                  # (G, T, 2, fold, K-1, ch)
    train = train.permute(0, 1, 3, 2, 4, 5).reshape(-1, 2 * (n_traces - 1), n_ch)
    test = traces.permute(0, 1, 3, 2, 4).reshape(-1, 2, n_ch)
    return train, test


def standardise(train, test):
    """StandardScaler per problem: mean and SD (ddof 0) of the training rows."""
    mean = train.mean(dim=1, keepdim=True)
    sd = train.std(dim=1, correction=0, keepdim=True)
    sd = torch.where(sd == 0, torch.ones_like(sd), sd)   # constant feature: scale 1
    return (train - mean) / sd, (test - mean) / sd


def smo(K, y, c=SVM_C, tol=SVM_TOL, max_iter=MAX_ITER):
    """libsvm's C-SVC dual solver, run on P problems at once.

    K : (P, n, n) linear kernels of the training rows
    y : (n,) labels +1 / -1, the same for every problem

    Returns alpha (P, n), rho (P,) and the number of iterations. The decision
    value of x is sum_i alpha_i y_i K(x_i, x) - rho.
    Follows Solver::select_working_set, Solver::Solve and
    Solver::calculate_rho in libsvm's svm.cpp.
    """
    n_prob, n, _ = K.shape
    alpha_all = torch.zeros(n_prob, n, dtype=K.dtype, device=K.device)
    G_all = -torch.ones_like(alpha_all)            # gradient of the dual
    # the loop works on the unfinished problems only: idx says which they are
    idx = torch.arange(n_prob, device=K.device)
    QD = torch.diagonal(K, dim1=1, dim2=2)
    alpha, G = alpha_all.clone(), G_all.clone()
    rows = idx.clone()
    pos = y > 0
    for it in range(max_iter):
        # working set: i = most violating in I_up, j = best second-order gain
        # in I_low; on ties libsvm keeps the last index, hence the flips
        v = -y * G
        up = (pos & (alpha < c)) | (~pos & (alpha > 0))
        low = (pos & (alpha > 0)) | (~pos & (alpha < c))
        v_up = torch.where(up, v, -torch.inf)
        i = n - 1 - v_up.flip(1).argmax(dim=1)
        g_max = v_up[rows, i]
        g_min = torch.where(low, v, torch.inf).min(dim=1).values
        grad_diff = g_max[:, None] - v
        quad = QD[rows, i][:, None] + QD - 2 * K[rows, i]
        quad = torch.where(quad > 0, quad, TAU)
        obj = torch.where(low & (grad_diff > 0), -grad_diff ** 2 / quad, torch.inf)
        obj_min, j = obj.flip(1).min(dim=1)
        j = n - 1 - j
        active = (g_max - g_min >= tol) & torch.isfinite(obj_min)
        # every CHECK_EVERY iterations, store the finished problems and go on
        # with the rest only (asking the GPU makes the CPU wait, so not every
        # iteration; in between, finished problems just stay frozen)
        if it % CHECK_EVERY == 0:
            done = ~active
            if done.any():
                alpha_all[idx[done]] = alpha[done]
                G_all[idx[done]] = G[done]
                idx, K, QD = idx[active], K[active], QD[active]
                alpha, G = alpha[active], G[active]
                rows = torch.arange(len(idx), device=K.device)
                if len(idx) == 0:
                    break
                continue          # working set is chosen again on the new set

        # update alpha_i, alpha_j (C_i = C_j = c)
        ai, aj = alpha[rows, i], alpha[rows, j]
        gi, gj = G[rows, i], G[rows, j]
        q = QD[rows, i] + QD[rows, j] - 2 * K[rows, i, j]
        q = torch.where(q > 0, q, TAU)
        # y_i != y_j: alpha_i - alpha_j stays fixed
        delta = (-gi - gj) / q
        d = ai - aj
        ni, nj = ai + delta, aj + delta
        fix = (d > 0) & (nj < 0)
        ni, nj = torch.where(fix, d, ni), torch.where(fix, 0., nj)
        fix = (d <= 0) & (ni < 0)
        ni, nj = torch.where(fix, 0., ni), torch.where(fix, -d, nj)
        fix = (d > 0) & (ni > c)
        ni, nj = torch.where(fix, c, ni), torch.where(fix, c - d, nj)
        fix = (d <= 0) & (nj > c)
        ni, nj = torch.where(fix, c + d, ni), torch.where(fix, c, nj)
        diff_i, diff_j = ni, nj
        # y_i == y_j: alpha_i + alpha_j stays fixed
        delta = (gi - gj) / q
        s = ai + aj
        ni, nj = ai - delta, aj + delta
        fix = (s > c) & (ni > c)
        ni, nj = torch.where(fix, c, ni), torch.where(fix, s - c, nj)
        fix = (s <= c) & (nj < 0)
        ni, nj = torch.where(fix, s, ni), torch.where(fix, 0., nj)
        fix = (s > c) & (nj > c)
        ni, nj = torch.where(fix, s - c, ni), torch.where(fix, c, nj)
        fix = (s <= c) & (ni < 0)
        ni, nj = torch.where(fix, 0., ni), torch.where(fix, s, nj)
        same = y[i] == y[j]
        ni, nj = torch.where(same, ni, diff_i), torch.where(same, nj, diff_j)
        # finished problems stay as they are
        ni, nj = torch.where(active, ni, ai), torch.where(active, nj, aj)

        # G += Q[:, i] dA_i + Q[:, j] dA_j, with Q_ik = y_i y_k K_ik
        G += y * (K[rows, i] * (y[i] * (ni - ai))[:, None]
                  + K[rows, j] * (y[j] * (nj - aj))[:, None])
        alpha[rows, i] = ni
        alpha[rows, j] = nj
    else:
        raise RuntimeError(f'SMO did not converge in {max_iter} iterations')
    alpha, G = alpha_all, G_all

    # intercept: mean over free support vectors, else middle of the bounds
    yG = y * G
    at_upper, at_lower = alpha >= c, alpha <= 0
    free = ~at_upper & ~at_lower
    ub_set = (at_upper & ~pos) | (at_lower & pos)
    lb_set = (at_upper & pos) | (at_lower & ~pos)
    ub = torch.where(ub_set, yG, torch.inf).min(dim=1).values
    lb = torch.where(lb_set, yG, -torch.inf).max(dim=1).values
    n_free = free.sum(dim=1)
    rho = torch.where(n_free > 0, (yG * free).sum(dim=1) / n_free.clamp(min=1),
                      (ub + lb) / 2)
    return alpha, rho, it


def decode_curves_gpu(X, groups, svm_c=SVM_C, chunk=CHUNK, dtype=torch.float64,
                      return_decision=False):
    """Accuracy curves for many groupings at once (GPU version of decode_curve).

    X : (n_trials, n_ch, n_times) array
    groups : (..., 2, n_traces, per_trace) trial indices, class 0 = face

    Returns acc (..., n_times). With return_decision, also the SVM decision
    values (..., n_times, n_traces, 2) for the test traces (face k, car k);
    positive means "face".
    """
    device = get_device()
    lead = groups.shape[:-3]
    groups = torch.as_tensor(np.asarray(groups).reshape(-1, *groups.shape[-3:]),
                             device=device)
    Xt = torch.as_tensor(X, dtype=dtype, device=device)
    n_traces = groups.shape[2]
    y = torch.cat([torch.ones(n_traces - 1), -torch.ones(n_traces - 1)]).to(device, dtype)
    y_test = torch.tensor([1., -1.], device=device, dtype=dtype)
    acc, dec = [], []
    for start in range(0, len(groups), chunk):
        # the SVMs only need the kernels (train x train, train x test), so
        # build them a few groupings at a time and drop the traces
        K, K_test = [], []
        for sub in range(start, min(start + chunk, len(groups)), BUILD_CHUNK):
            train, test = make_folds(Xt, groups[sub:min(sub + BUILD_CHUNK,
                                                        start + chunk)])
            train, test = standardise(train, test)
            K.append(train @ train.transpose(1, 2))
            K_test.append(train @ test.transpose(1, 2))
            del train, test
        K, K_test = torch.cat(K), torch.cat(K_test)
        alpha, rho, _ = smo(K, y, c=svm_c)
        # decision value of each test trace: sum_i alpha_i y_i <x_i, x> - rho
        dec_c = torch.einsum('pn,pnm->pm', alpha * y, K_test) - rho[:, None]
        del K, K_test
        pred = torch.where(dec_c > 0, 1., -1.)
        correct = (pred == y_test).to(dtype)                  # (P, 2)
        n_g = min(chunk, len(groups) - start)
        correct = correct.reshape(n_g, X.shape[2], n_traces, 2)
        acc.append(correct.mean(dim=(2, 3)).cpu().numpy())
        if return_decision:
            dec.append(dec_c.reshape(n_g, X.shape[2], n_traces, 2).cpu().numpy())
    acc = np.concatenate(acc).reshape(*lead, X.shape[2])
    if return_decision:
        dec = np.concatenate(dec).reshape(*lead, X.shape[2], n_traces, 2)
        return acc, dec
    return acc
