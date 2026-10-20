"""Airfoil pressure (AirfRANS): an in-context support set vs a trained MLP surrogate, on real RANS simulations.

A design is the inlet speed, the angle of attack and 16 samples of the airfoil's shape; its field is 64 pressure
coefficients along the upper and lower surfaces. Both models get the same 128 training designs and the same
calibration designs for their error spread. Held out: one design from each geometry group of the official full-test
split, in random order, so no queried shape was seen in training or calibration.

    pip install scikit-learn
    python examples/airfrans.py                    # reads examples/data/airfrans-pressure64.npz
"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from bitsy import Fn, laws, baselines, bakeoff, diagnose
from bitsy.adapters import field

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--data', default=os.path.join(HERE, 'data', 'airfrans-pressure64.npz'))
parser.add_argument('--train', type=int, default=128)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

from sklearn.neural_network import MLPRegressor
with np.load(args.data, allow_pickle=False) as z:
    X, Y, records = z['X'], z['Y'], json.loads(str(z['records']))
rng = np.random.default_rng(args.seed)

# splits by geometry group: official full-test groups are only ever queried
test_groups = {r['group'] for r in records if 'full_test' in r['official_tasks']}
groups = sorted({r['group'] for r in records} - test_groups)
rng.shuffle(groups)
calib_groups = set(groups[:len(groups) // 5])
train = np.array([i for i, r in enumerate(records) if r['group'] not in test_groups | calib_groups])
calib = np.array([i for i, r in enumerate(records) if r['group'] in calib_groups])
first = {}
for i, r in enumerate(records):
    if r['group'] in test_groups:
        first.setdefault(r['group'], i)
held = np.array(sorted(first.values()))
rng.shuffle(train)
rng.shuffle(held)
train = train[:args.train]

# normalise with the training designs only
xm, xs = X[train].mean(0), np.maximum(X[train].std(0), 1e-6)
ym, ys = Y[train].mean(0), max(float(Y[train].std()), 1e-6)
X, Y = (X - xm) / xs / np.sqrt(X.shape[1]), (Y - ym) / ys
grid = (-12.0, 12.0, 0.05)
support = list(zip(X[train], Y[train]))

# A: the support set, read in context; B: an MLP trained on the same designs. Same calibration cases for both spreads.
kernel = Fn(baselines.Kernel(Y.shape[1], grid=grid))
point = lambda evidence, x: kernel.law(evidence, x).mu
support_set = Fn(laws.spread(point, [(support, X[i], Y[i]) for i in calib], grid=grid))
mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, early_stopping=True, random_state=args.seed)
mlp.fit(X[train], Y[train])
surrogate = Fn(field(mlp, list(zip(X[calib], Y[calib])), grid))

queries = list(zip(X[held], Y[held]))
a, b = bakeoff.score(support_set, support, queries, L=64), bakeoff.score(surrogate, [], queries, L=64)
print(f'designs            {len(train)} training, {len(calib)} calibration, {len(held)} held-out shapes')
print(f'bits/site          support set {a["cost"].mean() / 64:.3f}, MLP surrogate {b["cost"].mean() / 64:.3f}')
for name, population in [('as a stream', None), ('on these shapes', len(held))]:
    r = bakeoff.race(a, b, robust=True, population=population)
    call = {'A': 'support set', 'B': 'MLP surrogate', '': 'no call'}[r['call']]
    print(f'bakeoff.race       {name}: {call} after {r["queries"]} held-out designs')

# which training designs does the support set rely on for one held-out shape?
x = X[held[0]]
dist = np.array([np.sum((s[0] - x) ** 2) for s in support])
e = diagnose.evidence(kernel, support, x, drop=list(np.argsort(dist)[:8]), matched=list(np.argsort(dist)[-8:]))
print(f'diagnose.evidence  its 8 nearest designs supply {e["relevant"]:.2f} bits/site; the 8 farthest {e["matched"]:.2f}')
