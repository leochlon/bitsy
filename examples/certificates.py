"""Certificates on three pretrained models: replay of 64 recorded episodes each.

Each design draws tasks whose true answer laws are known, so the entropy of the answers is exact per episode:
  TabPFN v2   logistic tasks from a pool of 4,096, d = 4; contexts of 8 to 1,024 rows; a block of 64 query rows.
              Windows: the last 16 or 64 rows. Bounded memories: logistic regression on a grid of 2 or 4 values per
              weight (5, 10 bits) and the best of 2^k fixed classifiers (k = 2, 4, 6, 8, 10 bits).
  ConvONet    a chair with 6 bumps, each present or not; 12 patches of 300 points in random order; the query is the
              occupancy inside each bump. Windows: 2 to 10 patches. Bounded memories: states storing k bumps.
  MVSNet      a card at one of 16 depths, seen by 1 of 4 source views; the query is the card's depth. Windows: 1 to 3
              views. Bounded memories: states storing the top k bits of the depth.
Every model's code length mixes it with 1% of the uniform law. examples/data/certificates-64.npz holds the per-episode
code lengths; this script recomputes every certificate and checks each issued one against the measured winner of
every pair it covers that was built.

    python examples/certificates.py
"""
import os, sys
from statistics import NormalDist
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from bitsy import gate, bakeoff

D = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'certificates-64.npz'))


def settings():
    """(model, kind, setting, certified memory, its code lengths, true-law code lengths, C, {covered: code lengths})."""
    t = lambda k: D['tabpfn__' + k]
    bounded = {'qlr1': 5, 'qlr2': 10, 'codebook2': 2, 'codebook4': 4, 'codebook6': 6, 'codebook8': 8, 'codebook10': 10}
    for N in (8, 16, 32, 64, 128, 256, 512, 1024):
        for W in (16, 64):
            for B in ('logistic', 'xgboost', 'tabpfn') if N > W else ():
                yield ('TabPFN', 'window', f'N={N}, last {W} rows', B, t(f'loss__{N}__{B}'), t(f'bayesW__{N}__{W}'),
                       None, {f'tabpfn_W{W}': t(f'loss__{N}__tabpfn_W{W}')})
        for C in sorted(set(bounded.values())):
            yield ('TabPFN', 'capacity', f'N={N}, C={C} bits', 'tabpfn', t(f'loss__{N}__tabpfn'), t('bayes0'), C,
                   {n: t(f'loss__{N}__{n}') for n, c in bounded.items() if c <= C})
    for name, key, unit, windows, F in (('ConvONet', 'occupancy', 'patches', (2, 4, 6, 8, 10), 6),
                                        ('MVSNet', 'stereo', 'views', (1, 2, 3), 4)):
        g = lambda k: D[f'{key}__{k}']
        for w in windows:
            yield (name, 'window', f'first {w} {unit}', 'net_full', g('loss__net_full'), g(f'bayesW__{w}'), None,
                   {f'net_W{w}': g(f'loss__net_W{w}')})
        for C in range(F):
            yield (name, 'capacity', f'C={C} bits', 'net_full', g('loss__net_full'), g('bayes0'), C,
                   {f'state{k}_{d}': g(f'loss__state{k}_{d}') for d in ('bayes', 'net') for k in range(C + 1)})


def run(z, show=False):
    """Per model: [issued, tested] per kind, and the checks of the issued certificates."""
    tally = {}
    for model, kind, setting, memory, ell, truth, C, covered in settings():
        cert = gate.window(ell, truth, z) if kind == 'window' else gate.capacity(ell, truth, C, z)
        n = tally.setdefault(model, dict(window=[0, 0], capacity=[0, 0], agree=0, none=0, disagree=0))
        n[kind][1] += 1
        if not cert['issued']:
            continue
        n[kind][0] += 1
        calls = {other: bakeoff.paired(ell, x, z)['call'] for other, x in covered.items()}
        for c in calls.values():
            n['agree' if c == 'A' else 'none' if c == '' else 'disagree'] += 1
        if show:
            beats = ', '.join(o if c == 'A' else f'{o} ({c or "no call"})' for o, c in calls.items())
            print(f'  {model:<9}{kind:<9}{setting:<21}{memory:<9}{cert["margin"]:+7.2f} [{cert["lo"]:+7.2f}, '
                  f'{cert["hi"]:+7.2f}]  beats {beats}')
    return tally


def line(t):
    issued = sum(n['window'][0] + n['capacity'][0] for n in t.values())
    tested = sum(n['window'][1] + n['capacity'][1] for n in t.values())
    checks = [sum(n[k] for n in t.values()) for k in ('agree', 'none', 'disagree')]
    return tested, f'{issued}/{tested} issued; checks: {checks[0]} agree, {checks[1]} no call, {checks[2]} disagree'


print('issued certificates, margin in bits [95% interval], and the built memories each one covers:')
tally = run(1.96, show=True)
print()
for model, n in tally.items():
    print(f'{model:<9} window {n["window"][0]}/{n["window"][1]}, capacity {n["capacity"][0]}/{n["capacity"][1]}; '
          f'checks: {n["agree"]} agree, {n["none"]} no call, {n["disagree"]} disagree')
tested, text = line(tally)
print(f'all       {text}')
z = NormalDist().inv_cdf(1 - 0.05 / tested)
print(f'z = {z:.2f}  {line(run(z))[1]}   (at most 5% chance of any false certificate among the {tested}, Bonferroni)')
