"""Chair: both certificates and an item audit on an occupancy network, live.

A chair (seat, backrest, four legs) carries 6 bumps, each present with probability 1/2. The context is 12 patches of 300
noisy surface points in a random order: one around each bump (on the bump if present, on the flat surface beneath it if
not) and 6 from the shared geometry; 300 anchor points on the shared geometry are always read. The query is the
occupancy of 32 random sites inside each bump's volume, so before any patch is read the answers hold exactly 6 bits,
and a window of w patches leaves exactly the bumps outside it unknown. Code lengths are per bump, averaged over its
sites, and mixed with 1% of the uniform law.

The memories: the network reading every patch; the network reading only the last w patches (Fn(window=w)); states
that store k bumps (k bits), read by the true law and by the network given fresh patches of those bumps.
diagnose.items then deletes each patch of one episode: the bump patches should carry the bits, the shared ones none.

    python examples/chair.py                                  # a numpy stand-in that reads the points
    python examples/chair.py --convonet REPO CONFIG CHECKPOINT  # the pretrained network (adapters.ConvONet)

Seed 0 and 64 episodes draw the same episodes as the recorded ConvONet run in examples/certificates.py.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from bitsy import Fn, gate, bakeoff, diagnose
from bitsy.adapters import occupancy, ConvONet

BASE = [((-0.30, -0.04, -0.30), (0.30, 0.04, 0.30)),           # seat, backrest, legs: boxes (min, max), y up
        ((-0.30, 0.04, 0.22), (0.30, 0.46, 0.30)),
        ((-0.30, -0.46, -0.30), (-0.22, -0.04, -0.22)), ((0.22, -0.46, -0.30), (0.30, -0.04, -0.22)),
        ((-0.30, -0.46, 0.22), (-0.22, -0.04, 0.30)), ((0.22, -0.46, 0.22), (0.30, -0.04, 0.30))]
BUMPS = [((-0.22, 0.04, -0.22), (-0.10, 0.12, -0.10)), ((0.10, 0.04, -0.22), (0.22, 0.12, -0.10)),
         ((-0.22, 0.04, -0.02), (-0.10, 0.12, 0.10)), ((0.10, 0.04, -0.02), (0.22, 0.12, 0.10)),
         ((-0.20, 0.18, 0.14), (-0.06, 0.32, 0.22)), ((0.06, 0.18, 0.14), (0.20, 0.32, 0.22))]
BUMP_FACE = [(1, 0.12, 0.04)] * 4 + [(2, 0.14, 0.22)] * 2        # protrusion axis, outer face, flat surface beneath
F, SHARED, R, POINTS, NOISE, EPS = 6, 6, 32, 300, 0.005, 0.01


def faces(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    out = []
    for ax in range(3):
        for sign, side in ((-1.0, lo[ax]), (1.0, hi[ax])):
            o = [i for i in range(3) if i != ax]
            out.append((ax, side, sign, o, lo, hi, (hi[o[0]] - lo[o[0]]) * (hi[o[1]] - lo[o[1]])))
    return out


def inside(p, boxes):
    m = np.zeros(len(p), bool)
    for lo, hi in boxes:
        m |= np.all((p > np.asarray(lo)) & (p < np.asarray(hi)), axis=1)
    return m


def surface(boxes, n, rng, nudge=1e-4):
    """Uniform points on the outer surface of a union of boxes (a face point is kept if a step outward leaves it)."""
    fs = [f for lo, hi in boxes for f in faces(lo, hi)]
    areas = np.array([f[6] for f in fs])
    pts = []
    while sum(len(x) for x in pts) < n:
        idx = rng.choice(len(fs), size=2 * n, p=areas / areas.sum())
        p, normal = np.empty((2 * n, 3)), np.zeros((2 * n, 3))
        for j, (ax, side, sign, o, lo, hi, _) in enumerate(fs):
            sel = idx == j
            k = int(sel.sum())
            if not k:
                continue
            p[sel, ax] = side
            p[sel, o[0]] = rng.uniform(lo[o[0]], hi[o[0]], k)
            p[sel, o[1]] = rng.uniform(lo[o[1]], hi[o[1]], k)
            normal[sel, ax] = sign
        pts.append(p[~inside(p + nudge * normal, boxes)])
    return np.concatenate(pts)[:n]


def region(p, f, margin=0.03):
    return np.all((p >= np.asarray(BUMPS[f][0]) - margin) & (p <= np.asarray(BUMPS[f][1]) + margin), axis=1)


def sample(boxes, n, rng, keep, oversample=40):
    out = []
    while sum(len(x) for x in out) < n:
        p = surface(boxes, oversample * n, rng)
        out.append(p[keep(p)])
    return np.concatenate(out)[:n]


def episode(rng):
    states = rng.random(F) < 0.5
    boxes = BASE + [BUMPS[f] for f in range(F) if states[f]]
    outside = lambda p: ~np.any([region(p, f) for f in range(F)], axis=0)
    anchors = sample(boxes, POINTS, rng, outside)
    patches = [sample(boxes, POINTS, rng, lambda p, f=f: region(p, f)) for f in range(F)]
    patches += [sample(boxes, POINTS, rng, outside) for _ in range(SHARED)]
    order = rng.permutation(len(patches))
    sites = np.stack([rng.uniform(np.asarray(lo) + 0.01, np.asarray(hi) - 0.01, size=(R, 3)) for lo, hi in BUMPS])
    jitter = lambda x: x + rng.normal(0.0, NOISE, x.shape)
    return states, jitter(anchors), [jitter(p) for p in patches], order, sites


def fresh(f, state, rng):
    """A new patch of bump f, drawn from the shape with that bump in the given state."""
    p = sample(BASE + ([BUMPS[f]] if state else []), POINTS, rng, lambda q: region(q, f))
    return p + rng.normal(0.0, NOISE, p.shape)


class Standin:
    """Reads the points like a network would: a bump is present if points lie on its outer face, absent if they lie on
    the flat surface beneath it, and 0.25 otherwise."""

    def __call__(self, points, sites):
        out = np.empty(sites.shape[:-1])
        for f in range(len(sites)):
            ax, outer, flat = BUMP_FACE[f]
            lo, hi = np.asarray(BUMPS[f][0]), np.asarray(BUMPS[f][1])
            o = [i for i in range(3) if i != ax]
            near = np.all((points[:, o] > lo[o] + 0.01) & (points[:, o] < hi[o] - 0.01), axis=1)
            out[f] = (0.97 if np.sum(near & (np.abs(points[:, ax] - outer) < 0.012)) >= 3 else
                      0.03 if np.sum(near & (np.abs(points[:, ax] - flat) < 0.012)) >= 3 else 0.25)
        return out


ap = argparse.ArgumentParser()
ap.add_argument('--convonet', nargs=3, metavar=('REPO', 'CONFIG', 'CHECKPOINT'))
ap.add_argument('--device', default='cpu')
ap.add_argument('--episodes', type=int, default=64)
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
net = ConvONet.load(*a.convonet, device=a.device) if a.convonet else Standin()
rng, windows = np.random.default_rng(a.seed), (2, 4, 6, 8, 10)
code = lambda model, items, query: bakeoff.score(model, items, [query], EPS)['mixed'][0] / R   # bits per bump
full = Fn(occupancy(net))
ell = {k: np.zeros(a.episodes) for k in ['full'] + [f'last {w}' for w in windows] +
       [f'state{k}_{d}' for k in range(F + 1) for d in ('bayes', 'net')]}
best = {w: np.zeros(a.episodes) for w in windows}
for t in range(a.episodes):
    states, anchors, patches, order, sites = episode(rng)
    items = [patches[i] for i in order[::-1]]                     # read last to first: a window holds the first read
    query = ((anchors, sites), np.repeat(states, R))
    ell['full'][t] = code(full, items, query)
    for w in windows:
        ell[f'last {w}'][t] = code(Fn(occupancy(net), window=w), items, query)
        best[w][t] = F - sum(int(i) < F for i in order[:w])      # bumps whose patch is outside the window
    for k in range(F + 1):
        ell[f'state{k}_bayes'][t] = F - k                         # k bumps stored, each other one a fair coin
        ell[f'state{k}_net'][t] = code(full, [fresh(f, states[f], rng) for f in range(k)], query)
    if t == 0:
        first = items, query, order

print(f'{a.episodes} episodes, {"ConvONet" if a.convonet else "stand-in"}: reading every patch costs '
      f'{ell["full"].mean():.2f} bits for the {F} bumps, which hold {F} bits before any patch is read')
said = {'A': 'every patch wins', 'B': 'the window wins', '': 'no call'}
for w in windows:
    c = gate.window(ell['full'], best[w])
    print(f'gate.window    last {w:>2} patches  margin {c["margin"]:+.2f} [{c["lo"]:+.2f}, {c["hi"]:+.2f}] '
          f'{"issued" if c["issued"] else "not issued"}; measured: {said[bakeoff.paired(ell["full"], ell[f"last {w}"])["call"]]}')
for C in range(F):
    c = gate.capacity(ell['full'], np.full(a.episodes, float(F)), C)
    calls = [bakeoff.paired(ell['full'], ell[f'state{k}_{d}'])['call'] for k in range(C + 1) for d in ('bayes', 'net')]
    print(f'gate.capacity  C = {C} bits        margin {c["margin"]:+.2f} [{c["lo"]:+.2f}, {c["hi"]:+.2f}] '
          f'{"issued" if c["issued"] else "not issued"}; measured: beats {calls.count("A")} of the {len(calls)} states')

items, query, order = first
r = diagnose.items(full, items, [query])
kind = ['bump' if int(i) < F else 'shared' for i in order[::-1]]
for name in ('bump', 'shared'):
    rows = [x for x, k in zip(r['items'], kind) if k == name]
    print(f'diagnose.items {name} patches, bits over the {F * R} sites: ' +
          ', '.join(f'{x["value"][0]:+.1f} {x["verdict"]}' for x in rows))
