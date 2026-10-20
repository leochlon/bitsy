"""The betting test behind bakeoff.race, on simulated streams where the truth is known.

Each query gives a paired difference in code length between two models, bounded below by -c with c = log2(2/0.01),
the bound for a yes/no outcome mixed with 1% of the uniform law. When neither model is better, a call is wrong: the
test keeps wrong calls at or below 5% however long it runs. When one is better by a fixed number of bits per query, it
is called sooner the larger the gap.

    python examples/race.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from bitsy import core

rng = np.random.default_rng(0)
c, streams, T = np.log2(2 / 0.01), 2000, 1024

nulls = {'symmetric': rng.uniform(-c, c, (streams, T)),                                  # mean 0
         'rare large losses': np.where(rng.random((streams, T)) < c / (c + 0.1), 0.1, -c)}  # mean 0, skewed
for name, d in nulls.items():
    call, _, _, _ = core.race(d, -d, c, c, alpha=0.05)
    print(f'no difference ({name}): wrong calls in {np.mean(call != ""):.1%} of {streams} streams of {T} queries')

for gap in [0.5, 0.1]:
    d = gap + 2 * rng.choice([-1.0, 1.0], (streams, T))                                   # +-2 bits of noise per query
    call, stop, _, _ = core.race(d, -d, c, c, alpha=0.05)
    right = [np.mean((call == 'A') & (stop <= n)) for n in (64, 256, 1024)]
    print(f'A better by {gap} bits/query: called right by 64 / 256 / 1024 queries in '
          f'{right[0]:.0%} / {right[1]:.0%} / {right[2]:.0%}; wrong {np.mean(call == "B"):.1%}')
