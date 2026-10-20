"""Predictive laws for any output, per site: yes/no (Bernoulli), bins (Categorical), values (Gaussian).
Each gives bits(observed), kl(other), prob(claim) and probs() over its outcomes. Bits are log base 2."""
import math
import numpy as np
from .core import CLIP, d2

_erfc = np.vectorize(math.erfc, otypes=[float])
_S2, _LN2 = math.sqrt(2.0), math.log(2.0)


def interval(a, b):
    """P(a < Z < b) for a standard normal Z, from whichever tail avoids cancellation."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.where(a > 0, 0.5 * (_erfc(a / _S2) - _erfc(b / _S2)), 0.5 * (_erfc(-b / _S2) - _erfc(-a / _S2)))


def _logq(z):
    """log P(Z > z), exact via erfc below z = 20 and by the asymptotic series beyond, so far tails never underflow."""
    z = np.asarray(z, float)
    out = np.full(z.shape, -np.inf)
    s, b = z < 20, (z >= 20) & np.isfinite(z)
    out[s] = np.log(0.5 * _erfc(z[s] / _S2))
    zb = z[b]
    out[b] = -zb ** 2 / 2 - np.log(zb) - 0.5 * math.log(2 * math.pi) + np.log1p(-1 / zb ** 2 + 3 / zb ** 4 - 15 / zb ** 6)
    return out


def _logmass(a, b):
    """log P(a < Z < b), stable in both tails."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    up, lo = a >= 0, b <= 0
    hi_a, hi_b = np.where(up, a, -b), np.where(up, b, -a)               # mirror the lower tail onto the upper
    tail = _logq(hi_a) + np.log(-np.expm1(np.minimum(_logq(hi_b) - _logq(hi_a), -1e-300)))
    mid = np.log(np.maximum(1 - np.exp(_logq(-a)) - np.exp(_logq(b)), 1e-300))
    return np.where(up | lo, tail, mid)


class Bernoulli:
    """Yes/no per site, e.g. occupancy. A claim is True (yes) or False (no), per site or for all."""
    V, grid = 2, 'bernoulli'

    def __init__(self, p):
        self.p = np.clip(np.asarray(p, float), CLIP, 1 - CLIP)

    def bits(self, y):
        return -np.log2(np.where(np.asarray(y, bool), self.p, 1 - self.p))

    def kl(self, o):
        return np.maximum(d2(self.p, o.p), 0.0)

    def prob(self, claim=True):
        return np.where(np.asarray(claim, bool), self.p, 1 - self.p)

    def probs(self):
        return np.stack([1 - self.p, self.p], -1)


class Categorical:
    """A law over fixed bins per site, e.g. depth hypotheses. P: [sites, bins]. A claim is a mask over the bins."""

    def __init__(self, P):
        P = np.clip(np.atleast_2d(np.asarray(P, float)), CLIP, None)
        self.P, self.V = P / P.sum(-1, keepdims=True), P.shape[-1]
        self.grid = ('bins', self.V)

    def bits(self, y):
        return -np.log2(np.take_along_axis(self.P, np.asarray(y).reshape(-1, 1), -1)[:, 0])

    def kl(self, o):
        return np.maximum((self.P * np.log2(self.P / o.P)).sum(-1), 0.0)

    def prob(self, claim):
        return (self.P * np.asarray(claim, float)).sum(-1)

    def probs(self):
        return self.P


class Gaussian:
    """A Gaussian per site, e.g. pressure. A claim is an interval (lo, hi) per site. Code lengths need grid=(lo, hi, h):
    values are binned at width h on [lo, hi] plus two open tail bins (continuous outputs are discretised), and
    both sides of a comparison must use the same grid. If values are recorded at resolution h, offset lo by h/2 so they
    do not sit on bin edges. The divergence uses the densities."""

    def __init__(self, mu, var, grid=None):
        self.mu = np.asarray(mu, float)
        self.var = np.broadcast_to(np.maximum(np.asarray(var, float), 1e-300), self.mu.shape)
        self.grid, self.V = grid, (None if grid is None else int(round((grid[1] - grid[0]) / grid[2])) + 2)

    def _logmass(self, k):
        if self.grid is None:
            raise ValueError('code lengths and outcome probabilities need grid=(lo, hi, h)')
        e = self.grid[0] + self.grid[2] * np.arange(self.V - 1)
        a = np.where(k == 0, -np.inf, e[np.clip(k - 1, 0, self.V - 2)])
        b = np.where(k == self.V - 1, np.inf, e[np.clip(k, 0, self.V - 2)])
        s = np.sqrt(self.var)
        return _logmass((a - self.mu) / s, (b - self.mu) / s)

    def bits(self, y):
        if self.grid is None:
            raise ValueError('scoring observed values needs grid=(lo, hi, h)')
        z = (np.asarray(y, float) - self.grid[0]) / self.grid[2]
        if np.isnan(z).any():
            raise ValueError('observed values contain NaN')
        k = np.clip(np.floor(z), -1, self.V - 2).astype(int) + 1
        return -self._logmass(k) / _LN2

    def kl(self, o):
        return np.maximum((np.log(o.var / self.var) + (self.var + (self.mu - o.mu) ** 2) / o.var - 1) / (2 * _LN2), 0.0)

    def prob(self, claim):
        s = np.sqrt(self.var)
        return interval((np.asarray(claim[0], float) - self.mu) / s, (np.asarray(claim[1], float) - self.mu) / s)

    def probs(self):
        return np.stack([np.exp(self._logmass(np.full(self.mu.shape, k))) for k in range(self.V)], -1)


def _var(r2, floor, tiny, pooled):
    v = np.mean(np.concatenate([x.ravel() for x in r2])) if pooled or len({x.shape for x in r2}) > 1 else np.mean(r2, 0)
    return np.maximum(np.maximum(v, floor * np.median(v)), tiny)


def spread(point, calib, floor=0.05, tiny=1e-12, pooled=False, grid=None):
    """Give a point predictor an error-based Gaussian spread, fixed per site. point(evidence, spec) -> values. calib: (evidence, spec,
    observed) cases disjoint from the predictor's training data and from any queries scored later. Variance: the mean
    squared error per site position (pooled=True, or cases of different sizes, pools over sites), floored at floor x
    its median and at tiny. Returns predict(evidence, spec) -> Gaussian."""
    v = _var([(np.asarray(y, float) - np.asarray(point(e, s), float)) ** 2 for e, s, y in calib], floor, tiny, pooled)
    return lambda e, s: Gaussian(point(e, s), v, grid)


def cv_spread(fit, cases, k=4, floor=0.05, tiny=1e-12, pooled=False, grid=None, seed=0):
    """Train a point predictor on all its cases and give it a spread from k-fold cross-validation on the same cases, so
    no extra data is needed (k = len(cases) is leave-one-out). fit(cases) ->
    point(evidence, spec); cases: (evidence, spec, observed). Returns predict(evidence, spec) -> Gaussian."""
    folds = np.array_split(np.random.default_rng(seed).permutation(len(cases)), min(k, len(cases)))
    r2 = []
    for f in folds:
        point = fit([c for i, c in enumerate(cases) if i not in set(f)])
        r2 += [(np.asarray(cases[i][2], float) - np.asarray(point(cases[i][0], cases[i][1]), float)) ** 2 for i in f]
    full, v = fit(cases), _var(r2, floor, tiny, pooled)
    return lambda e, s: Gaussian(full(e, s), v, grid)
