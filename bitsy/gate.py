"""Gate: is a fine-tune worth paying for?"""
import numpy as np
from .core import ci
from .models import _is_units, Fn


def fit(model, data, query=None):
    """Does the data fit the window with a query (text, or its length in tokens)? If it does, the prompt holds all of
    it, so on average a fine-tune can beat the prompt only by the prompt's decoding loss (what the model fails to use from it), which fit
    alone cannot measure: see bakeoff and diagnose.unused. If not, the prompt drops what falls outside; a fine-tune
    can use it."""
    w = model.window
    if isinstance(model, Fn):  # any other model: a window, if any, counts items
        if w is None:
            return dict(tokens=None, window=None, fits=None, times=None,
                        read='no window: the memory reads every item; whether it keeps all of it cannot be checked from outside')
        return dict(tokens=len(data), window=w, fits=len(data) <= w, times=len(data) / w,
                    read='no item is dropped, though the memory may still lose detail' if len(data) <= w else
                    f'the memory drops {len(data) - w} items; a fine-tune can use them')
    if w is None:
        raise ValueError('set the model\'s window (in tokens)')
    q = query if isinstance(query, (int, np.integer)) else (model.n(query) if query is not None else 0)
    n = 1 + (sum(model.n(u) + 1 for u in data) if _is_units(data) else model.n(data)) + q
    return dict(tokens=n, window=w, fits=n <= w, times=n / w,
                read='fits: a fine-tune can only win by what the model fails to use from the prompt' if n <= w else
                f'{n / w:.1f}x the window: the prompt drops {n - w} tokens; a fine-tune can use them')


def curves(model, units, lr=3e-5, seed=0):
    """Both learning curves read from the data itself. The units are shuffled (seed=None keeps your order;
    the identity needs exchangeable units). Unit j is scored by the prompt holding the units before it (window rule)
    and by an online fine-tune before its step on unit j, so entry j estimates each curve at j units. One ordering is
    noisy: average a few seeds. For any model a unit is what it reads (text, or a (spec, observed) example); the
    fine-tune curve needs a local model with online steps, otherwise it is None. Returns per-unit bits (prompt,
    fine-tune)."""
    if seed is not None:
        units = [units[i] for i in np.random.default_rng(seed).permutation(len(units))]
    icl = np.array([model.bits(units[:j], [u])[0].sum() for j, u in enumerate(units)])
    if not hasattr(model, 'step'):
        return icl, None
    ft = model.copy()
    tun = np.array([ft.step(u, lr).sum() for u in units])
    return icl, tun


def bins(icl, tun):
    """Curves averaged over sizes 0, 1, 2-3, 4-7, ... -> rows (first, last, prompt, fine-tune, lower). tun may be None."""
    rows, lo = [], 0
    while lo < len(icl):
        hi = min(max(2 * lo - 1, lo), len(icl) - 1)
        a = icl[lo:hi + 1].mean()
        b = None if tun is None else float(tun[lo:hi + 1].mean())
        rows.append((lo, hi, float(a), b, None if b is None else ('prompt' if a < b else 'fine-tune')))
        lo = hi + 1
    return rows


def info(P, exact=200_000, samples=4096, seed=0):
    """I(Y_1:K; D) in bits for D uniform over M variants and outputs independent given D. P: [M, K, C]."""
    M, K, C = P.shape
    lp = np.log2(np.clip(P, 1e-300, 1))
    h = -(P * lp).sum((1, 2)).mean()
    if C ** K <= exact:
        mix = np.zeros(C ** K)
        for m in range(M):
            j = P[m, 0]
            for k in range(1, K):
                j = np.multiply.outer(j, P[m, k]).ravel()
            mix += j
        mix /= M
        return float(-(mix * np.log2(np.clip(mix, 1e-300, 1))).sum() - h)
    rng, tot = np.random.default_rng(seed), 0.0
    for _ in range(samples):
        m = rng.integers(M)
        y = [rng.choice(C, p=P[m, k] / P[m, k].sum()) for k in range(K)]
        l = lp[:, np.arange(K), y].sum(1)
        tot += l[m] - (np.logaddexp2.reduce(l) - np.log2(M))
    return float(tot / samples)


def size(model, variants, prompts, cands=None, bits=None):
    """Storage budget. variants: the context with its content resampled; prompts: the workload (for any other
    model, the specs, whose outcomes are its law's bins); cands: for language models, the outputs read, with distinct
    first tokens (the rest lumped as 'other'). I is what the prompt's outputs reveal about
    which variant it holds. A fixed state built from the context (adapter, compiled or compressed prompt) that stores
    at most `bits` bits about it (e.g. log2 of its number of states) misses the prompt's outputs on this workload by at
    least I - bits in total, on average over the variants. Capped at log2(len(variants)); with variants drawn from real
    data, I is a lower bound in expectation. Needs full probabilities (local, or a law with finite outcomes)."""
    I = info(np.array([model.dist(v, prompts, cands) for v in variants]))
    out = dict(I=I, cap=float(np.log2(len(variants))))
    if bits is not None:
        out.update(bits=bits, miss=max(I - bits, 0.0), too_small=I > bits)
    return out


def capacity(prompt, prior, C, z=1.96):
    """Certificate: the prompt beats every memory that stores at most C bits about its context (a fine-tune, adapter,
    compiled or compressed prompt of that size, with any decoder). Such a memory costs at least H(Q) - C bits on average,
    where H(Q) is the entropy of the answers before any context is read. One entry per episode, each a fresh task,
    context and query block drawn from a design whose true laws you know. prompt: its code lengths; prior: the code
    lengths under the true law with no context, whose mean estimates H(Q). Issued when the paired margin prior - prompt
    - C has a normal interval (z) above zero. The interval is pointwise: raise z when testing many settings."""
    m, lo, hi = ci(np.asarray(prior, float) - np.asarray(prompt, float) - C, z)
    return dict(margin=m, lo=lo, hi=hi, issued=lo > 0)


def window(memory, best, z=1.96):
    """Certificate: this memory (a fine-tune, or a model that reads everything) beats every memory that reads only the
    window (the last W items or tokens), with any decoder. Any of those costs at least H(Q | window) bits on average.
    One entry per episode, as for capacity. memory: its code lengths; best: the code lengths under the true law given
    only what the window holds, whose mean estimates H(Q | window). Issued when the paired margin best - memory has a
    normal interval (z) above zero."""
    m, lo, hi = ci(np.asarray(best, float) - np.asarray(memory, float), z)
    return dict(margin=m, lo=lo, hi=hi, issued=lo > 0)
