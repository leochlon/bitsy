"""Diagnose: is the information missing, or is the model failing to use it?
value and unused need observed outcomes; evidence and spans need none; items reports every context item, with or
without outcomes."""
import numpy as np
from .core import ci, d2
from . import gate, guard


def value(model, context, matched, queries):
    """Bits per query from the context's content, beyond its form. matched: same form, irrelevant content.
    Averaged over contexts, what any context of that form gains (by correcting the model's default guesses) cancels in
    the difference, and a content-free context gains at most that much. For one context the intervals cover query
    noise only."""
    b0, bm, bc = (np.array([t.sum() for t in model.bits(c, queries)]) for c in ([], matched, context))
    return dict(format=ci(b0 - bm), content=ci(bm - bc), total=ci(b0 - bc))


def unused(prompt, tuned, fits):
    """Floor on how far the prompt's predictions sit from the task's true queries: any fine-tune's average advantage
    over the prompt is at most KL(true || prompt), given the context. If the context fits the window,
    that is the prompt's decoding loss: information the model fails to use. If not, it also counts what the window
    dropped. Valid when queries are drawn independently from the task and neither the fine-tune nor its recipe saw
    them. prompt, tuned: bakeoff.score() of the same queries. fits: the memory keeps the whole context, as a transformer
    does inside its window (gate.fit); for a memory that may lose detail, such as a feature grid, pass False. One-sided:
    a fine-tune that loses proves nothing.
    The floor is an approximate one-sided 97.5% bound (normal approximation), not anytime-valid."""
    d = prompt['cost'] - tuned['cost']
    q, t = ci(d), ci(d / prompt['tokens'])
    what = 'decoding loss' if fits else 'storage + decoding loss'
    return dict(floor=max(q[1], 0.0), per_token=max(t[1], 0.0), mean=q[0], mean_per_token=t[0], what=what,
                read=f'{what} >= floor' if q[1] > 0 else 'no evidence either way')


def _avg(x, w):
    return float(np.average(x, weights=w))


def evidence(model, items, spec, drop, matched, w=None):
    """Bits per site that deleted evidence supplies, beyond a matched deletion of the same form, for any model with a
    law (Fn). KL(law with all items || law rebuilt without them) is the model's own valuation of those items: no
    labels needed. drop, matched: indices of the items to delete; w: site weights (e.g. area). The matched
    deletion is what a deletion of that form costs by itself, so 'content' is what the relevant items' content adds.
    One unit (object, scene, car); take intervals across units, never across sites."""
    A = model.law(items, spec)
    I = lambda d: A.kl(model.law([e for i, e in enumerate(items) if i not in set(d)], spec))
    a, b = I(drop), I(matched)
    return dict(relevant=_avg(a, w), matched=_avg(b, w), content=_avg(a, w) - _avg(b, w), sites=a)


def _d2_range(p, q):
    """d2(p||q) over p in [p0, p1], q in [q0, q1]: 0 at the low end if the ranges overlap, else the nearest corners;
    the high end at a corner (d2 is jointly convex)."""
    lo = 0.0 if p[0] <= q[1] and q[0] <= p[1] else float(d2(p[0], q[1]) if p[0] > q[1] else d2(p[1], q[0]))
    return lo, float(max(d2(a, b) for a in p for b in q))


def spans(model, claim, evidence, mask=True, placeholder=guard.PLACEHOLDER):
    """Bits each evidence item supplies to one text claim, for any model with labels() (hosted chat included, no echo):
    the item alone is replaced by the placeholder and the others stay, as evidence() deletes items from a law. bits =
    d2(p||q) for P(YES) with (p) and without (q) the item, as a range over top-k ranges; lift = p - q, also a range.
    No labels needed. The yes/no channel shows only part of what an item moves, and for a language model
    P(YES) is an elicited reading (guard.prompt), not a bound on the answer law's divergence."""
    items = [evidence] if isinstance(evidence, str) else list(evidence)
    p = tuple(model.labels(guard.prompt(claim, items, (), mask, placeholder))['YES'])
    rows = []
    for i in range(len(items)):
        q = tuple(model.labels(guard.prompt(claim, items, [i], mask, placeholder))['YES'])
        rows.append(dict(item=i, q=q, bits=_d2_range(p, q), lift=(p[0] - q[1], p[1] - q[0])))
    return dict(p=p, items=rows)


REASONS = {
    'evidence': 'helps through what it says: real evidence the answer depends on. Keep it, and favour more items like it.',
    'format': 'helps only through how it looks: format or label space, not facts. Any well-formatted example would do, '
              'so a short template can replace it.',
    'misleading': 'changes the answer but makes it worse: the model misreads information it has rather than missing it. '
                  'A model problem, not a data problem, and the case where fine-tuning can win.',
    'dead weight': 'adds nothing beyond the rest: redundant or irrelevant. Cut it, and the prompt gets shorter and '
                   'cheaper with no loss.',
    'helps': 'helps; give it a control of the same form to tell what it says from how it looks.',
    'unclear': 'helps, but these queries do not separate what it says from how it looks; add queries.',
    'redundant': 'worth something together, nothing one at a time: its items repeat each other, so one can go.',
    'moves': 'moves the prediction; with no outcomes, whether it helps or hurts is unknown.',
    'inert': 'does not move the prediction.',
}


def _join(cue, answer):
    return cue + answer if isinstance(cue, str) else list(cue) + list(answer)


def _evaluate(model, jobs):
    """(per-query bits, prediction, context tokens re-read, context tokens) for each (context, queries) job, in order.
    Local re-reads only what follows the prefix shared with the job before; a model with a law gives its laws; any
    other model gives bits only."""
    if hasattr(model, 'sweep'):
        yield from model.sweep(jobs)
    elif hasattr(model, 'law'):
        for context, qs in jobs:
            laws = [model.law(context, spec) for spec, _ in qs]
            yield [None if y is None else np.atleast_1d(L.bits(y)) for L, (_, y) in zip(laws, qs)], laws, None, None
    else:
        for context, qs in jobs:
            yield model.bits(context, qs), None, None, None


def _moved(a, b):
    """Per query, how far the prediction moved from a to b: KL(a || b) in bits, or None if the model gives none."""
    if a is None:
        return None
    if isinstance(a, np.ndarray):
        a, b = a.astype(float), b.astype(float)
        return np.maximum((np.exp(a) * (a - b)).sum(-1) / np.log(2), 0.0)
    return np.array([float(np.sum(x.kl(y))) for x, y in zip(a, b)])


def _sign(x, z, tol):
    """(mean, low, high) and +1 helps, -1 hurts, 0 neither: the mean is beyond tol bits and, over several queries, its
    interval excludes zero. For a single query low and high are None."""
    if len(x) == 1:
        return (float(x[0]), None, None), (int(np.sign(x[0])) if abs(x[0]) > tol else 0)
    m, lo, hi = ci(x, z)
    return (m, lo, hi), (1 if lo > 0 and m > tol else -1 if hi < 0 and m < -tol else 0)


def items(model, context, queries=None, controls=None, groups=(), screen=None, outcomes=True, z=1.96, tol=0.5,
          move=0.05):
    """Which context items do the work, which are dead weight and which mislead the model, in bits per query.

    context: the deletable items of one prompt (passages, retrieved chunks, demonstrations; for Fn, evidence items).
    queries: held-out queries, as for bakeoff.score. With outcomes=False they are prompts without answers (for Fn,
    specs) and only influence is reported. If None, each item is also a query, scored by the others, so no outside labels
    are needed: (cue, answer) demonstrations for a language model, (spec, observed) examples for Fn.
    controls: controls[i] has item i's form with irrelevant content: the same demonstration with a shuffled label, an
    off-topic passage of the same length. Then value = content + form: content is the item against its control, form
    is the control against nothing.
    groups: lists of item indices deleted together. One-at-a-time deletion measures what an item adds beyond the rest,
    so two items that say the same thing each look worthless alone; their group does not.
    screen: for long contexts, delete the items in this many contiguous blocks first and test items one at a time only
    inside blocks that matter. A block that adds nothing can be cut as a whole; effects that cancel inside a block can
    be missed.

    Every item is deleted and the context read again, never masked: masking leaves its information in what follows.
    Per item: value, the code length without it minus with it (positive helps, negative hurts), mean and interval over
    queries; influence, how far it moves the prediction, needing no labels (for a language model, the first answer
    token's distribution, a lower bound on the whole answer's); content and form with controls; a verdict and its
    reason (REASONS). A verdict needs a mean beyond tol bits per query and, with several queries, an interval over
    them that excludes zero (z); per_query shows which question an item serves. move: influence below this many bits counts as not moving the prediction. Cost: one read per
    item, two with controls, one per group or block, plus the full and empty contexts; Local re-reads only what
    follows each deleted item. reads counts them; tokens is (context tokens re-read, context tokens in all reads)."""
    n, local, fn = len(context), hasattr(model, 'sweep'), hasattr(model, 'law')
    own = queries is None
    if own:
        units = [_join(*u) for u in context] if local else list(context)
        alt = None if controls is None else ([_join(*c) for c in controls] if local else list(controls))
        asks, qs = [[u] for u in context], None
    else:
        units, alt, asks, qs = list(context), None if controls is None else list(controls), None, list(queries)
        if not outcomes:
            if not (local or fn):
                raise ValueError('influence without outcomes needs a model that gives its predictions (Local or Fn)')
            qs = [(q, [model.sep]) for q in qs] if local else [(q, None) for q in qs]
    if alt is not None and len(alt) != n:
        raise ValueError('need one control per item')
    keys = list(range(n)) if own else [None]                       # the held-out item, or None for outside queries
    tokens = [0, 0]                                                  # context tokens re-read, and read in all

    def run(variants):
        """Read every variant (name, drop, swap) for every key; returns {(name, key): (cost per query, prediction)}."""
        jobs, names = [], []
        for j in keys:
            for name, drop, swap in [('full', (), None)] + variants:
                if j is not None and name != 'empty' and (j in drop or j == swap):
                    continue                                     # a held-out item is never in its own context
                gone = set(drop) | ({j} if j is not None else set())
                jobs.append(([alt[k] if k == swap else units[k] for k in range(n) if k not in gone],
                             asks[j] if own else qs))
                names.append((name, j))
        out = {}
        for key, (bits, pred, read, size) in zip(names, _evaluate(model, jobs)):
            out[key] = (np.array([b.sum() for b in bits]) if outcomes else None, pred)
            if read is not None:
                tokens[0] += read
                tokens[1] += size
        return out, len(jobs)

    def row(label, res, name, swap=None):
        """Value, influence, content and form of one deletion, pooled over the keys."""
        val, inf, con, frm = [], [], [], []
        for j in keys:
            if (name, j) not in res:
                continue
            base, var = res[('full', j)], res[(name, j)]
            moved = _moved(base[1], var[1])
            if moved is not None:
                inf.extend(moved)
            if outcomes:
                val.extend(var[0] - base[0])
                if swap and (swap, j) in res:
                    con.extend(res[(swap, j)][0] - base[0])
                    frm.extend(var[0] - res[(swap, j)][0])
        r = dict(item=label, influence=float(np.mean(inf)) if inf else None)
        if not outcomes:
            r['verdict'] = 'moves' if inf and np.mean(inf) > move else 'inert'
            return r
        (v, s), c, f = _sign(np.array(val), z, tol), con and _sign(np.array(con), z, tol), frm and _sign(np.array(frm), z, tol)
        r.update(value=v, content=c[0] if c else None, form=f[0] if f else None, per_query=np.array(val))
        r['verdict'] = ('misleading' if s < 0 else 'dead weight' if s == 0 else 'helps' if not c else
                        'evidence' if c[1] > 0 else 'format' if f[1] > 0 else 'unclear')
        return r

    blocks = [list(map(int, b)) for b in np.array_split(np.arange(n), screen)] if screen and screen < n else []
    sets = [list(map(int, g)) for g in groups] + blocks
    res, reads = run([(f'group {k}', tuple(g), None) for k, g in enumerate(sets)] +
                     ([('empty', tuple(range(n)), None)] if outcomes else []))
    quiet = [b for k, b in enumerate(blocks)
             if (lambda r: r['verdict'] in ('dead weight', 'inert') and (r['influence'] or 0) <= move)(
                 row(b, res, f'group {len(groups) + k}'))]
    keep = set(range(n)) - {i for b in quiet for i in b}
    more, extra = run([v for i in sorted(keep, reverse=True)       # late items first: the longest shared prefixes
                       for v in [(f'del {i}', (i,), None)] + ([(f'swap {i}', (), i)] if alt is not None else [])])
    res.update(more)
    rows = []
    for i in range(n):
        if i in keep:
            r = row(i, res, f'del {i}', f'swap {i}' if alt is not None else None)
            r['why'] = REASONS[r['verdict']]
            if r['verdict'] == 'format' and r['content'] and r['content'][0] < -tol:
                r['why'] += ' What it says is worse than a control\'s: fix it or replace it with a template.'
        else:
            r = dict(item=i, influence=None, value=None, content=None, form=None, per_query=None,
                     verdict='dead weight' if outcomes else 'inert',
                     why='its block adds nothing as a whole, so it was not tested one at a time.')
        rows.append(r)
    group_rows = []
    for k, g in enumerate(sets):
        r = row(g, res, f'group {k}')
        if r['verdict'] == 'helps' and all(i in keep and rows[i]['verdict'] == 'dead weight' for i in g):
            r['verdict'] = 'redundant'
        r['why'] = REASONS[r['verdict']]
        group_rows.append(r)
    fits = None
    try:
        if local:
            asked = [model._split(q) for q in ([a[0] for a in asks] if own else qs)]
            fits = gate.fit(model, units, max(len(c) + len(a) for c, a in asked))['fits']
        elif fn:
            fits = gate.fit(model, units)['fits']
    except Exception:
        fits = None
    if fits is False:
        for r in rows + group_rows:
            if r['verdict'] == 'misleading':
                r['why'] += ' The context does not fit the window, so part of this can be what the window drops.'
    total = None
    if outcomes:
        total = _sign(np.concatenate([res[('empty', j)][0] - res[('full', j)][0] for j in keys]), z, tol)[0]
    return dict(items=rows, groups=group_rows, total=total, fits=fits, reads=reads + extra,
                tokens=tuple(tokens) if tokens[1] else None)
