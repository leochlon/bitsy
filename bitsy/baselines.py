"""A support-set baseline for engineering: an in-context field learner to set against a trained surrogate."""
import numpy as np
from .laws import Gaussian


class Kernel:
    """An in-context field learner: Nadaraya-Watson averaging of the support
    examples' fields, weighted by design similarity, with the bandwidth chosen by leave-one-out prediction on the support
    set and a spread from the leave-one-out residuals floored at floor x their median. It solves no Gram system, uses no
    geometric descriptors, and its spread is the same for every query. Support example:
    (design vector, field over shared sites). With fewer than two examples it returns the prior. Use as
    Fn(Kernel(sites, grid=...))."""

    def __init__(self, sites, scales=(0.05, 0.1, 0.2, 0.4, 0.8, 1.6), floor=0.05, prior=(0.0, 1.0), grid=None):
        self.sites, self.scales, self.floor, self.prior, self.grid = sites, scales, floor, prior, grid

    @staticmethod
    def _w(d2, h):
        a = -d2 / (2 * h * h)
        a = np.exp(a - a.max(-1, keepdims=True))
        return a / a.sum(-1, keepdims=True)

    def __call__(self, evidence, spec):
        if len(evidence) < 2:
            return Gaussian(np.full(self.sites, self.prior[0]), np.full(self.sites, self.prior[1]), self.grid)
        X = np.array([np.atleast_1d(e[0]) for e in evidence], float)
        F = np.array([e[1] for e in evidence], float)
        D = ((X[:, None] - X[None]) ** 2).sum(-1)
        np.fill_diagonal(D, np.inf)
        res = {h: F - self._w(D, h) @ F for h in self.scales}
        h = min(self.scales, key=lambda h: (res[h] ** 2).mean())
        v = (res[h] ** 2).mean(0)
        v = np.maximum(v, self.floor * np.median(v))
        mu = self._w(((X - np.atleast_1d(np.asarray(spec, float))) ** 2).sum(-1), h) @ F
        return Gaussian(mu, v, self.grid)
