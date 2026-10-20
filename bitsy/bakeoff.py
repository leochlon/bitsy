"""Bakeoff: two memories on one stream of held-out queries; call the better one with error <= alpha.
'A' means memory A, mixed with eps of the uniform law (at most log2(1/(1-eps)) = 0.015 bits per token at eps = 0.01),
costs less than B on average over the task's queries; 'B' the reverse. Queries must be drawn independently.
The error bound concerns the probability of a wrong call per started stream,
not the fraction wrong conditional on getting a decision. Budgets may end undecided."""
import numpy as np
from .core import CLIP, ci, mixed, race as _race
from .models import LABELS


def score(model, context, queries, eps=0.01, L=None):
    """Code lengths of held-out queries: the prompt reads them with the context, a fine-tune with none.
    L, the longest possible query in tokens, bounds the mixed code length: fix it in advance for race(); the default
    (the longest query here) is valid for pool(), whose queries are drawn from these."""
    tb = model.bits(context, queries)
    lengths = np.array([len(t) for t in tb])
    if not len(lengths) or (lengths < 1).any():
        raise ValueError('need nonempty scored queries')
    if not 0 < eps <= 1 or model.V is None or not np.isfinite(model.V) or model.V < 1:
        raise ValueError('need 0 < eps <= 1 and a finite outcome count')
    if not all(np.isfinite(t).all() and (t >= 0).all() for t in tb):
        raise ValueError('code lengths must be finite and non-negative')
    fixed = L is not None
    L = int(lengths.max()) if L is None else L
    if not isinstance(L, (int, np.integer)) or L < lengths.max():
        raise ValueError('L must be an integer bounding every query length')
    return dict(cost=np.array([t.sum() for t in tb]), mixed=np.array([mixed(t, model.V, eps) for t in tb]),
                c=L * np.log2(model.V / eps), tokens=np.array([len(t) for t in tb]), grid=getattr(model, 'grid', None),
                bound_fixed=fixed)


def labels(model, prompts, truth, names=LABELS, eps=0.01):
    """Code lengths of known labels read from the first output token, for any model with labels() (hosted chat
    included: no echo). prompts[i]'s true label is truth[i], one of names: a class, a multiple-choice letter, or YES/NO
    for a claim with a known answer (guard.prompt). One outcome per query, so c = log2(len(names)/eps) is fixed in
    advance. Label mass on other tokens counts against the model. When top-k truncation leaves a range, cost and mixed
    hold its low end and cost_hi and mixed_hi its high end; race() and pool() then take, for each side being called, the
    end worse for that call, which keeps the error guarantee and costs only power. Exact ranges give single values."""
    names, truth = tuple(names), list(truth)
    if len(truth) != len(prompts) or not prompts or any(y not in names for y in truth):
        raise ValueError('need one true label from names per prompt')
    if not 0 < eps <= 1:
        raise ValueError('need 0 < eps <= 1')
    R = [model.labels(p, names) for p in prompts]
    lo, hi = (np.array([r[y][k] for r, y in zip(R, truth)], float) for k in (0, 1))
    raw = lambda p: -np.log2(np.maximum(p, CLIP))
    mix = lambda p: -np.log2((1 - eps) * p + eps / len(names))
    out = dict(cost=raw(hi), mixed=mix(hi), c=float(np.log2(len(names) / eps)), tokens=np.ones(len(lo)),
               grid=('labels', names), bound_fixed=True)
    if (hi > lo).any():
        out.update(cost_hi=raw(lo), mixed_hi=mix(lo))
    return out


def graded(s, lo=0.0, hi=1.0):
    """Bounded per-query scores, higher is better (e.g. graded answers from a hosted model without logprobs).
    The same betting test, valid because the score difference is bounded below by -(hi - lo)."""
    s = np.asarray(s, float)
    if not np.all((s >= lo) & (s <= hi)):
        raise ValueError(f'scores must lie in [{lo}, {hi}]')
    return dict(cost=hi - s, mixed=hi - s, c=hi - lo, tokens=np.ones(len(s)))


def _d(a, b, idx=None, robust=False):
    if a.get('grid') is not None and b.get('grid') is not None and a['grid'] != b['grid']:
        raise ValueError('the two sides use different outcome grids, so their code lengths are not comparable')
    k = 'mixed' if robust else 'cost'
    values = [np.asarray(s[name], float) for s in (a, b) for name in ('cost', 'mixed')]
    if any(v.ndim != 1 or not v.size or v.shape != values[0].shape or
           not np.isfinite(v).all() or (v < 0).any() for v in values):
        raise ValueError('scores must be finite non-negative vectors of paired queries')
    for s in (a, b):
        for name in ('cost', 'mixed'):
            v = np.asarray(s.get(name + '_hi', s[name]), float)
            if v.shape != values[0].shape or np.isnan(v).any() or (v < np.asarray(s[name], float)).any():
                raise ValueError('a range must hold its low end in cost/mixed and its high end in cost_hi/mixed_hi')
        top = np.asarray(s.get('mixed_hi', s['mixed']), float)
        if not np.isfinite(s['c']) or s['c'] <= 0 or (top > s['c'] + 1e-9).any():
            raise ValueError('declared bound must cover every mixed query cost')
    # Betting on A takes B's low end and A's high end, and the reverse for B: the null mean can only fall.
    da = np.asarray(b[k]) - np.asarray(a.get('mixed_hi', a['mixed']))
    db = np.asarray(a[k]) - np.asarray(b.get('mixed_hi', b['mixed']))
    return (da[None], db[None]) if idx is None else (da[idx], db[idx])


def race(a, b, alpha=0.05, robust=False, population=None):
    """Live: a, b scored on the same queries in arrival order. Returns the call ('A', 'B' or '') and queries used.
    robust=True compares the eps-mixed versions of both memories, which is what you would deploy: an overconfident side
    then cannot lose the call on a few outliers alone. Valid because a mixed code length is non-negative and bounded
    by c.
    population=N targets a fixed finite cohort revealed uniformly without
    replacement. Its result concerns that cohort, not a wider population."""
    for score_ in (a, b):
        if score_.get('bound_fixed') is False and population != len(score_['cost']):
            raise ValueError('fix L before an independent-stream race; inferred L is only valid for a complete finite cohort')
    call, stop, _, _ = _race(*_d(a, b, robust=robust), a['c'], b['c'], alpha, population=population)
    return dict(call=str(call[0]), queries=int(min(stop[0], len(a['cost']))))


def pool(a, b, streams=2000, T=1000, alpha=0.05, seed=0, robust=False):
    """Offline: query streams drawn at random from a scored pool. How often is each called?
    truth compares the costs being tested (raw, or mixed if robust); wrong counts calls whose own hypothesis is false
    on the pool. With ranges (bakeoff.labels), gap is B's low end minus A's high end, gap_hi the reverse, and truth is
    '' unless one side wins at both ends."""
    idx = np.random.default_rng(seed).integers(0, len(a['cost']), size=(streams, T))
    call, stop, _, _ = _race(*_d(a, b, idx, robust), a['c'], b['c'], alpha)
    (da,), (db,) = _d(a, b, robust=robust)
    k = 'mixed' if robust else 'cost'
    gap, gap_hi = b[k] - a.get(k + '_hi', a[k]), b.get(k + '_hi', b[k]) - a[k]
    dec = call != ''
    wrong = ((call == 'A') & (da.mean() <= 0)) | ((call == 'B') & (db.mean() <= 0))
    out = dict(truth='A' if gap.mean() > 0 else ('B' if gap_hi.mean() < 0 else ''), gap=ci(gap),
               A=int((call == 'A').sum()), B=int((call == 'B').sum()), none=int((~dec).sum()), wrong=int(wrong.sum()),
               median_queries=float(np.median(stop[dec])) if dec.any() else None)
    if any(k + '_hi' in s for s in (a, b)):
        out['gap_hi'] = ci(gap_hi)
    return out


def paired(a, b, z=1.96):
    """Fixed sample: which of two memories costs less on the same episodes or queries? a, b: their code lengths. The
    mean of b - a with a normal interval: 'A' if its low end is above zero, 'B' if its high end is below, '' otherwise.
    Fix the sample size in advance; race() is the test you may stop at any time."""
    m, lo, hi = ci(np.asarray(b, float) - np.asarray(a, float), z)
    return dict(call='A' if lo > 0 else 'B' if hi < 0 else '', diff=m, lo=lo, hi=hi)
