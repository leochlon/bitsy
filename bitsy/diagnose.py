"""Diagnose: is the information missing, or is the model failing to use it?
value and unused need observed outcomes; evidence and spans need none; items reports every context item, with or
without outcomes. When unused finds a decoding loss, prior, order, depth, heads and dilution look for where it lives:
in the output prior, in position, at some depth, or in the heads that route the context to the answer."""
import numpy as np
from .core import ci, d2
from .models import _is_units
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


# ---- where the decoding loss lives
#
# Every readout of a network is a predictor of the answer from the same context, so on held-out queries the difference
# in code length between two of them is exactly the difference in their decoding losses, in expectation, with no need
# for the true distribution. Each function below compares two such predictors.

def _longest(model, queries):
    return max(len(c) + len(a) for c, a in (model._split(q) for q in queries))


def _fits(model, units, queries):
    if hasattr(model, '_split') and not gate.fit(model, list(units), _longest(model, queries))['fits']:
        raise ValueError('the units must fit the window with the longest query, so that rearranging or deleting units '
                         'changes nothing else')


def prior(prompt, tuned, form, z=1.96):
    """Is the fine-tune winning on the prior repair or on the content? prompt, tuned, form: bakeoff.score() of the same
    held-out queries for the prompt (reading the context), a fine-tune on the context, and a fine-tune on a matched
    control with the same form and irrelevant content (both reading nothing). The form fine-tune can learn only the
    format and the usual answers, so its advantage over the prompt is a floor on the decoding loss the prompt spends on
    the prior repair: the model sees the format in its context and does not adopt it. What the full fine-tune adds
    beyond it comes from the content. Valid when no fine-tune saw the queries and the control says nothing about them.
    share is the part of the full advantage the form fine-tune reproduces; the reading calls the prior when the floor is
    above zero and the share is at least half."""
    a, b, f = (np.asarray(s['cost'], float) for s in (prompt, tuned, form))
    if a.ndim != 1 or not (a.shape == b.shape == f.shape):
        raise ValueError('score the same queries for all three')
    total, rep, content = ci(a - b, z), ci(a - f, z), ci(f - b, z)
    share = rep[0] / total[0] if total[0] > 0 else None
    if total[1] <= 0:
        read = 'no evidence the fine-tune beats the prompt'
    elif rep[1] > 0 and share is not None and share >= 0.5:
        read = (f'mostly the prior repair ({share:.0%} of the advantage): the model sees the format in its context and '
                'does not adopt it. Look at the readout (final norm, output head, an output bias) before the network.')
    else:
        read = 'mostly content: the model fails to use what its context says. order(), depth() and heads() look for where.'
    return dict(total=total, prior=rep, content=content, share=share, floor=max(rep[1], 0.0), read=read)


def order(model, context, queries, k=8, seed=0, tuned=None, z=1.96):
    """Decoding loss from position alone. The prompt reads the units in k random orders, and the average of those k
    predictions is a reader whose answer cannot depend on the order it was given. Per query its code length is at most
    the orders' average (Jensen), and on held-out queries the gap is exactly decoding loss that order dependence costs,
    in expectation. A large cost points at the positional encoding. context: units that fit the window. tuned:
    bakeoff.score() of a fine-tune on the same queries, to report the share of its advantage over the given order that
    order alone explains. given_cost compares the order you gave; cost compares a random order."""
    units, qs = list(context), list(queries)
    if not _is_units(units) or len(units) < 2:
        raise ValueError('order needs a context of at least two units')
    _fits(model, units, qs)
    rng = np.random.default_rng(seed)
    perms = [np.arange(len(units))] + [rng.permutation(len(units)) for _ in range(k)]
    cost = np.array([[b.sum() for b in model.bits([units[i] for i in p], qs)] for p in perms])
    given, shuffled = cost[0], cost[1:]
    mix = -(np.logaddexp.reduce(-shuffled * np.log(2), axis=0) - np.log(k)) / np.log(2)
    out = dict(given=ci(given, z), orders=ci(shuffled.mean(0), z), mixed=ci(mix, z), cost=ci(shuffled.mean(0) - mix, z),
               given_cost=ci(given - mix, z), k=k, share=None)
    g = out['given_cost']
    if tuned is not None:
        gap = ci(given - np.asarray(tuned['cost'], float), z)
        out['advantage'] = gap
        out['share'] = g[0] / gap[0] if gap[0] > 0 else None
    if g[1] > 0:
        out['read'] = (f'order alone costs {g[0]:.2f} bits per query (>= {g[1]:.2f}), recovered by averaging over orders'
                       + (f', {out["share"]:.0%} of the fine-tune\'s advantage' if out['share'] is not None else '')
                       + ': look at the positional encoding')
    else:
        out['read'] = 'no evidence that order costs bits'
    return out


def depth(model, context, queries, train=None, rank=None, epochs=4, lr=1e-3, z=1.96):
    """Bits by depth: where the context's bits reach the answer, and where they are lost. The answer is read off the
    residual stream after every layer and scored on these held-out queries; the readouts are trained on `train`,
    (context, queries) jobs from other documents (a tuned lens; rank keeps each readout low-rank), or, without it, are
    the logit lens, which needs no training. The difference in code length between the model's own output and the
    readout after layer k is exactly the decoding loss the layers after k add, in expectation: lost[k] above zero
    means those layers throw usable bits away. If no readout comes close to a fine-tune's code length, the bits never
    fully reach the answer positions, which is routing (heads()). depth(tuned, [], ...) gives a fine-tune's curve to
    compare. Returns bits [layers + 1, queries], the row means, lost per layer and the best layer."""
    lens = model.lens(train, rank=rank, epochs=epochs, lr=lr) if train is not None else None
    B = model.depth_bits(context, queries, lens)
    L = B.shape[0] - 1
    lost = [ci(B[L] - B[k], z) for k in range(L)]
    best = int(np.argmin(B[:L].mean(1)))
    out = dict(bits=B, mean=B.mean(1), layers=[ci(B[k], z) for k in range(L + 1)], lost=lost, best=best,
               lens='tuned' if lens is not None else 'logit')
    out['read'] = (f'the readout after layer {best} beats the model\'s own output by {lost[best][0]:.2f} bits per query '
                   f'(>= {lost[best][1]:.2f}): the layers after it throw usable bits away' if lost[best][1] > 0 else
                   'no layer\'s readout beats the model\'s own output')
    return out


def heads(model, context, queries, units=None, top=10):
    """Which attention heads carry each unit's bits to the answer. For each unit tested and each head, the rise in the
    answer's code length if that head alone lost the unit (attribution patching, Local.routes), next to the measured
    rise when the unit is deleted outright. units: the indices to test, such as the items diagnose.items marks as
    evidence; default all. The heads' first-order estimates need not add up to the deletion, which also moves the MLPs
    and is not linear; heads_sum shows how much they account for. Returns routes [units, layers, heads], exact,
    heads_sum and the top routes as (unit, layer, head, bits)."""
    if not hasattr(model, 'routes'):
        raise ValueError('heads needs a local model (Local)')
    context = list(context)
    _fits(model, context, queries)
    keep = list(range(len(context))) if units is None else [int(u) for u in units]
    R, exact = model.routes(context, queries, keep)
    flat = sorted(((keep[u], l, h, float(R[u, l, h])) for u in range(R.shape[0]) for l in range(R.shape[1])
                   for h in range(R.shape[2])), key=lambda t: -t[3])[:top]
    read = '; '.join(f'unit {u} through layer {l} head {h}: {b:+.2f} bits' for u, l, h, b in flat[:3])
    return dict(units=keep, routes=R, exact=exact, heads_sum=R.sum((1, 2)), top=flat, read=read)


def dilution(model, contexts, queries, targets, heads=None):
    """Attention the heads pay to one unit as the context around it changes, such as 8, 16 and 32 passages that all
    contain it. Softmax attention normalises over the whole context, so every added unit competes for the weight the
    right one gets; this measures it. contexts: lists of units; targets: the index of the same unit in each; queries:
    questions that unit answers; heads: (layer, head) pairs to report, such as the top routes from heads(). Needs
    attention weights: Local(path, attn='eager'). Returns sizes, attention [contexts, layers, heads] (mass on the unit,
    averaged over query positions and queries) and the prompt's mean bits per query for each context."""
    if len(contexts) != len(targets):
        raise ValueError('need the target unit\'s index in every context')
    qs = list(queries)
    A = np.array([np.mean([model.attends(list(c), q, t) for q in qs], 0) for c, t in zip(contexts, targets)])
    out = dict(sizes=[len(c) for c in contexts], attention=A,
               bits=[float(np.mean([b.sum() for b in model.bits(list(c), qs)])) for c in contexts])
    if heads:
        out['heads'] = {(int(l), int(h)): A[:, l, h] for l, h in heads}
        out['read'] = '; '.join(f'layer {l} head {h}: ' + ' -> '.join(f'{v:.3f}' for v in A[:, l, h]) for l, h in heads) \
            + f' of the attention on the unit, with {" -> ".join(map(str, out["sizes"]))} units'
    return out
