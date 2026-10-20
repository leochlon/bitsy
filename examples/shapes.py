"""3D shapes: which observed points does an occupancy prediction rely on?

A sphere with a bump is seen through 20,000 noisy surface points, grouped into deletable patches. The model gives, for
any point in space, the probability that it lies inside the shape. Deleting the patch that sees the bump removes the
bits behind the bump's predictions; deleting as many points from the far side removes almost none.

    python examples/shapes.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from bitsy import Fn, laws, diagnose, guard

rng = np.random.default_rng(0)
TOP = np.array([0.0, 0.0, 1.0])                                     # the bump points this way


def surface(n=20000, bump=0.3, width=0.12):
    """Noisy points on a unit sphere with a bump of height `bump` around TOP, as (direction, radius)."""
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    angle = np.arccos(np.clip(u @ TOP, -1, 1))
    return u, 1 + bump * np.exp(-(angle / width) ** 2) + 0.004 * rng.normal(size=n)


def patches(u, r, width=0.3):
    """Item 0: the points that see the bump. Item 1: as many points from the far side. Then 7 anchor patches."""
    angle = np.arccos(np.clip(u @ TOP, -1, 1))
    near, far = angle < width, angle > np.pi - width
    k = int(min(near.sum(), far.sum()))
    groups = [np.flatnonzero(near)[:k], np.flatnonzero(far)[:k]] + np.array_split(np.flatnonzero(~near & ~far), 7)
    return [(u[g], r[g]) for g in groups]


def occupancy(evidence, spec, bw=0.05, sharp=0.01):
    """P(inside) at each query point: the surface radius in its direction, from observed points within ~bw radians."""
    U = np.concatenate([e[0] for e in evidence]) if evidence else np.zeros((0, 3))
    R = np.concatenate([e[1] for e in evidence]) if evidence else np.zeros(0)
    uq, rq = spec
    w = np.exp(-(np.arccos(np.clip(uq @ U.T, -1, 1)) / bw) ** 2)
    radius = (w @ R + 1e-3) / (w.sum(1) + 1e-3)                     # no nearby points: the unit sphere
    return laws.Bernoulli(1 / (1 + np.exp(-(radius - rq) / sharp)))


# query sites inside the bump: directions within 0.06 rad of TOP, radii 1.05 to 1.2
theta, phi = rng.uniform(0, 0.06, 256), rng.uniform(0, 2 * np.pi, 256)
sites = (np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], 1), rng.uniform(1.05, 1.2, 256))

shape, items = Fn(occupancy), patches(*surface())
e = diagnose.evidence(shape, items, sites, drop=[0], matched=[1])
print(f'diagnose.evidence  the bump patch supplies {e["relevant"]:.2f} bits/site; as many far-side points {e["matched"]:.4f}')
for name, drop in [('bump patch', [0]), ('far-side patch', [1])]:
    c = guard.claims(shape, items, sites, drop, True)
    print(f'guard.claims       delete the {name}: {c["crossings"]}/{c["confident"]} confident "inside" claims fall '
          f'(need {c["required"]:.2f}, supplied {c["supplied"]:.2f} bits/site)')
