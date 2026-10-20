"""Fields: an in-context support set vs a trained surrogate, on a family of designs with surface fields.

A design is 3 numbers; its field is 64 values along a surface. The support set reads 32 past designs in context. The
surrogate is trained on n designs, with an error spread from cross-validation on its own training cases. Both are
scored on 200 held-out designs. Then: which past design does a prediction rely on, and how fast does the support set
learn?

    python examples/fields.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from bitsy import Fn, laws, baselines, core, gate, bakeoff, diagnose

S = np.linspace(0, 1, 64)                                           # 64 sites along the surface
GRID = (-4.0, 4.0, 0.01)                                            # fields are scored on 0.01-wide bins


def field(x):
    return (x[0] * np.sin(2 * np.pi * S) + (1 + 0.5 * x[1]) * np.cos(np.pi * S) * x[2]
            + 0.8 * np.tanh(2 * x[0] * x[1]) * np.exp(-((S - 0.5 - 0.2 * x[2]) / 0.08) ** 2))


def sims(n, rng, noise=0.02):
    """n simulated designs: (design, noisy field)."""
    return [(x, field(x) + noise * rng.normal(size=S.shape)) for x in rng.uniform(-1, 1, (n, 3))]


class Ridge:
    """A trained surrogate: random Fourier features of the design, ridge regression to the field. Its memory is its
    weights, so it ignores the evidence it is passed."""

    def __init__(self, train, m=256, gamma=1.5, lam=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.W, self.b = rng.normal(0, gamma, (3, m)), rng.uniform(0, 2 * np.pi, m)
        X, F = np.array([t[0] for t in train]), np.array([t[1] for t in train])
        P = self.f(X)
        self.B = np.linalg.solve(P.T @ P + lam * np.eye(P.shape[1]), P.T @ F)

    def f(self, X):
        return np.sqrt(2 / self.W.shape[1]) * np.cos(np.atleast_2d(X) @ self.W + self.b)

    def __call__(self, evidence, spec):
        return (self.f(spec) @ self.B)[0]


rng = np.random.default_rng(0)
support, held = sims(32, rng), sims(200, rng)
kernel = Fn(baselines.Kernel(64, grid=GRID))
a = bakeoff.score(kernel, support, held)
fit = lambda cases: Ridge([(c[1], c[2]) for c in cases])
for n in [4, 8, 16, 32, 64, 2000]:
    train = [([], x, y) for x, y in sims(n, np.random.default_rng(n))]
    b = bakeoff.score(Fn(laws.cv_spread(fit, train, grid=GRID)), [], held)
    r = bakeoff.pool(a, b, streams=200)
    winner = 'support set' if r['truth'] == 'A' else 'surrogate'
    print(f'bakeoff n={n:<5d}    support set {a["cost"].mean() / 64:.2f} vs surrogate {b["cost"].mean() / 64:.2f} bits/site: '
          f'{winner} called in {max(r["A"], r["B"])}/200 streams, median {r["median_queries"]:.0f} designs, wrong {r["wrong"]}')

rel, mat = [], []
for _ in range(100):
    past, x = sims(24, rng), rng.uniform(-1, 1, 3)
    dist = [np.sum((p[0] - x) ** 2) for p in past]
    e = diagnose.evidence(kernel, past, x, drop=[int(np.argmin(dist))], matched=[int(np.argmax(dist))])
    rel.append(e['relevant'])
    mat.append(e['matched'])
m, lo, hi = core.ci(np.array(rel) - np.array(mat))
print(f'diagnose.evidence  the nearest past design supplies {m:.2f} more bits/site than the farthest [{lo:.2f}, {hi:.2f}]')

curve, _ = gate.curves(kernel, sims(64, rng))
print(f'gate.curves        support set: {curve[2:8].mean() / 64:.2f} bits/site after 2-7 designs -> '
      f'{curve[32:].mean() / 64:.2f} after 32-63')
