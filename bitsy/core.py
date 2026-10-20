"""Bits. All logs are base 2."""
import numpy as np

CLIP = 1e-12


def d2(a, b):
    """Binary divergence d2(a||b) in bits."""
    a, b = np.clip(a, CLIP, 1 - CLIP), np.clip(b, CLIP, 1 - CLIP)
    return np.maximum(a * np.log2(a / b) + (1 - a) * np.log2((1 - a) / (1 - b)), 0.0)


def budget(q, t=0.95):
    """Bits a claim needs to rise from probability q to target t: d2(t||q) if q < t, else 0. Evidence that lifts a
    claim from q to t or above supplies at least this many bits."""
    q = np.asarray(q, float)
    return np.where(q < t, d2(t, q), 0.0)


def mixed(tok_bits, V, eps=0.01):
    """Code length of the eps-uniform mixture, from per-token bits [..., L] -> [...]. Never above L log2(V/eps)."""
    p = 2.0 ** -np.asarray(tok_bits, float)
    return -np.log2((1 - eps) * p + eps / V).sum(-1)


def ci(x, z=1.96):
    """(mean, low, high) of x, normal approximation."""
    x = np.asarray(x, float)
    m = x.mean()
    s = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.inf
    return float(m), float(m - z * s), float(m + z * s)


def eprocess(x, lam_max=0.75, n0=1.0, s0=0.01):
    """log E_t of the betting test on increments x = Delta/c >= -1, one row per stream [R,T].
    Predictable bet lambda = clip(m/(v+m^2), 0, lam_max) from past increments, mean shrunk by n0
    pseudo-observations, pseudo-variance s0. When the increments have conditional mean <= 0, the capital is a
    non-negative supermartingale, so it ever reaches 1/alpha with probability at most alpha."""
    x = np.atleast_2d(np.asarray(x, float))
    if x.ndim != 2 or not x.size or not np.isfinite(x).all() or (x < -1 - 1e-12).any():
        raise ValueError('betting increments must be finite, nonempty and >= -1')
    if not (0 <= lam_max < 1 and n0 > 0 and s0 > 0):
        raise ValueError('need 0 <= lam_max < 1 and positive pseudo-count/variance')
    x = np.maximum(x, -1)
    R, T = x.shape
    out, s1, s2, cur = np.zeros((R, T)), np.zeros(R), np.zeros(R), np.zeros(R)
    for t in range(T):
        n = t + n0
        m = s1 / n
        v = (s2 - 2 * m * s1 + t * m ** 2 + n0 * s0) / n
        lam = np.clip(m / (v + m ** 2), 0.0, lam_max)
        cur = cur + np.log1p(lam * x[:, t])
        out[:, t] = cur
        s1 += x[:, t]
        s2 += x[:, t] ** 2
    return out


def eprocess_wor(x, population):
    """Bet on a fixed finite cohort revealed in uniformly random order, without replacement.

    Under cohort mean <= 0, E[x_t | past] <= -sum(past)/(N-t+1).
    Subtract that conditional null mean and use its predictable lower bound.
    Infinite wealth marks a deterministic certificate: even -1 for every
    remaining item cannot make the cohort mean non-positive.
    """
    x = np.atleast_2d(np.asarray(x, float))
    if x.ndim != 2 or not x.size or not np.isfinite(x).all() or (x < -1 - 1e-12).any():
        raise ValueError('betting increments must be finite, nonempty and >= -1')
    if not isinstance(population, (int, np.integer)) or population < x.shape[1]:
        raise ValueError('population must be an integer at least as large as the stream')
    x = np.maximum(x, -1)
    total = np.cumsum(x, axis=1)
    past = np.concatenate([np.zeros((len(x), 1)), total[:, :-1]], axis=1)
    remaining = population - np.arange(x.shape[1])
    shift = past / remaining
    lower = 1 - shift
    z = np.zeros_like(x)
    np.divide(x + shift, lower, out=z, where=lower > 0)
    wealth = eprocess(np.maximum(z, -1))
    certain = total > (remaining - 1) + 1e-12 * population
    return np.where(np.maximum.accumulate(certain, axis=1), np.inf, wealth)


def race(da, db, ca, cb, alpha=0.05, population=None):
    """Two tests on one query stream, each at level alpha/2; the first to reach 2/alpha makes the call.
    da = cost_B - cost_A^mixed >= -ca favours A; db = cost_A - cost_B^mixed >= -cb favours B. Rows are streams.
    Returns call [R] in {'A','B',''}, queries used [R], log E for A and for B.
    population=N targets the mean of a fixed cohort of N items sampled without
    replacement in random order; the default targets an independent query stream."""
    da, db = np.asarray(da, float), np.asarray(db, float)
    if da.shape != db.shape:
        raise ValueError('both directions must score the same paired queries')
    if not (np.isfinite(ca) and np.isfinite(cb) and ca > 0 and cb > 0 and 0 < alpha < 1):
        raise ValueError('need positive finite bounds and 0 < alpha < 1')
    thr = np.log(2 / alpha)
    process = eprocess if population is None else lambda x: eprocess_wor(x, population)
    ea, eb = process(da / ca), process(db / cb)
    T = ea.shape[1]
    first = lambda e: np.where((e >= thr).any(1), (e >= thr).argmax(1) + 1, T + 1)
    ta, tb = first(ea), first(eb)
    call = np.where((ta <= T) & (ta <= tb), 'A', np.where(tb <= T, 'B', ''))
    return call, np.minimum(ta, tb), ea, eb
