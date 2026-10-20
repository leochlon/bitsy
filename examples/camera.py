"""Cameras: depth for a new view from the views already taken.

Three spheres are photographed by 12 depth cameras on a ring. Each view is a deletable evidence item: its pixels are
back-projected to 3D points. For a new camera the model gives, per pixel, a distribution over 24 depth bins plus
"nothing hit". The curve shows each view getting cheaper to predict as more views are read; deleting the views next to
a new camera costs far more than deleting the views on the opposite side, and its confident depth claims fall with them.

    python examples/camera.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from bitsy import Fn, laws, diagnose, guard, gate

H = W = 24                                                          # pixels per side
EDGES = np.linspace(1.0, 7.0, 25)                                   # 24 depth bins; outcome 24 = nothing hit
CENTRES = (EDGES[:-1] + EDGES[1:]) / 2
SPHERES = [(np.array([0.0, 0.0, 0.0]), 1.0), (np.array([1.2, 0.6, 0.3]), 0.5), (np.array([-0.9, -1.0, -0.2]), 0.6)]


def rays(angle, ring=4.0, height=0.8, fov=np.radians(50)):
    """Camera centre and one unit ray per pixel, for a camera on the ring looking at the origin."""
    c = np.array([ring * np.cos(angle), ring * np.sin(angle), height])
    f = -c / np.linalg.norm(c)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    s = np.tan(fov / 2) * np.linspace(-1, 1, W)
    x, y = np.meshgrid(s, -s)
    d = f + x.reshape(-1, 1) * right + y.reshape(-1, 1) * up
    return c, d / np.linalg.norm(d, axis=1, keepdims=True)


def render(angle):
    """The observed depth bin of every pixel (24 where the ray hits nothing)."""
    c, d = rays(angle)
    t = np.full(len(d), np.inf)
    for centre, radius in SPHERES:
        oc = c - centre
        b = d @ oc
        disc = b * b - (oc @ oc - radius * radius)
        near = -b - np.sqrt(np.maximum(disc, 0))
        t = np.where((disc > 0) & (near > 0), np.minimum(t, near), t)
    return np.where(np.isfinite(t), np.clip(np.digitize(t, EDGES) - 1, 0, 23), 24)


def depth(evidence, angle, sigma=0.1, tau=0.15, kappa=2.0, floor=1e-3):
    """Per-pixel depth law for a camera at `angle`, from views (angle, depth bins) back-projected to 3D points. Points
    near a pixel's ray vote for their depth along it, the nearest surface first; few points near the ray means likely
    nothing is hit."""
    c, d = rays(angle)
    pts = [cam + CENTRES[b[b < 24], None] * dirs[b < 24] for cam, dirs, b in ((*rays(a), b) for a, b in evidence)]
    pts = np.concatenate(pts) if pts else np.zeros((0, 3))
    P = np.full((len(d), 25), 1 / 25)
    if len(pts):
        v = pts - c
        t = d @ v.T                                                 # depth of each point along each pixel's ray
        near = np.exp(-((v * v).sum(1) - t ** 2) / (2 * sigma ** 2)) * (t > 0)
        w = near * np.exp(-(t - t.min(1, keepdims=True)) / tau)     # the nearest surface along the ray wins
        k = np.clip(np.digitize(t, EDGES) - 1, 0, 23)
        hist = np.stack([(w * (k == j)).sum(1) for j in range(24)], 1)
        none = np.exp(-near.sum(1) / kappa)                         # no points near the ray: probably nothing there
        P[:, :24] = (1 - none)[:, None] * hist / np.maximum(hist.sum(1, keepdims=True), 1e-300)
        P[:, 24] = none
    return laws.Categorical((1 - floor) * P + floor / 25)


camera = Fn(depth)
views = [(a, render(a)) for a in 2 * np.pi * np.arange(12) / 12]

curve, _ = gate.curves(camera, views)                                # each view scored from the views before it
print(f'gate.curves        {curve[1] / H / W:.2f} bits/pixel with 1 view read -> {curve[-4:].mean() / H / W:.2f} with 8-11')

new = 2 * np.pi * 0.5 / 12                                           # a new camera between views 0 and 1
e = diagnose.evidence(camera, views, new, drop=[0, 1], matched=[6, 7])
print(f'diagnose.evidence  the 2 neighbouring views supply {e["relevant"]:.2f} bits/pixel; the 2 opposite views {e["matched"]:.2f}')

P = camera.law(views, new).P
band = np.abs(np.arange(25)[None] - P.argmax(1)[:, None]) <= 1     # claim: depth within one bin of the prediction
band[P.argmax(1) == 24] = np.arange(25) == 24                        # or: nothing hit
for name, drop in [('2 neighbouring views', [0, 1]), ('2 opposite views', [6, 7])]:
    g = guard.claims(camera, views, new, drop, band, t=0.9)
    print(f'guard.claims       delete the {name}: {g["crossings"]}/{g["confident"]} confident depth claims fall')
truth, sure = render(new), (P * band).sum(1) >= 0.9
print(f'check              rendered at the new camera, {np.mean(band[sure, truth[sure]]):.0%} of the {sure.sum()} confident '
      f'claims are true')
