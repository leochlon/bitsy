# Prompt or fine-tune: gate and bakeoff

Should knowledge live in the prompt or in the weights? `gate` tells you whether a fine-tune is worth trying before
you train one. `bakeoff` compares a prompt and a fine-tune you have built, on the same held-out queries, and stops as
soon as it is sure which is better.

The worked example is `examples/language.py`: a 512-token passage of Shakespeare is the knowledge, and 64 recall
queries each give 8 tokens of it and score the next 24. The prompt reads the passage; the fine-tune is trained on it
and reads nothing. On GPT-2 it prints:

```
gate.fit        fits: a fine-tune can only win by what the model fails to use from the prompt
gate.curves     16-31 units seen: prompt 7.00, fine-tune 6.14 bits/token -> fine-tune leads
gate.size       the prompt carries >= 4.96 bits of a 12-bit table; a 2-bit state misses it by >= 2.96
bakeoff  1 ep   prompt 1.69 vs fine-tune 6.19 bits/token: prompt called in 500/500 streams after a median 27 queries, wrong 0
bakeoff 16 ep   prompt 1.69 vs fine-tune 0.29 bits/token: fine-tune called in 500/500 streams after a median 83 queries, wrong 0
diagnose.value  content 5.42 bits/token, format 0.44 bits/token
diagnose.unused decoding loss >= 1.25 bits/token: what the model fails to use from its prompt
```

## gate: is a fine-tune worth trying?

```python
from bitsy import Local, gate

m = Local('gpt2')
gate.fit(m, docs, query)                         # do the docs fit the window with the query?
gate.curves(m, units)                            # prompt and fine-tune learning curves, read from the data
gate.size(m, variants, prompts, cands, bits)     # can a state of `bits` bits ever match the prompt?
```

- **fit.** If the context fits the window, the prompt holds all of it, so on average a fine-tune on the same data can
  only win by what the model fails to use from its prompt. If it does not fit, the prompt drops what falls outside,
  and a fine-tune can use it. `fit` returns `tokens`, `window`, `fits`, `times` and `read`, a sentence that says which.
- **curves.** Each unit is scored before the model has seen it, once with the earlier units in the prompt and once
  after online fine-tuning on them. That gives both learning curves from the data itself, with no separate training
  runs. One ordering is noisy: average a few seeds. `gate.bins(icl, tun)` averages the curves over sizes 0, 1, 2-3,
  4-7, ... and says which leads. The fine-tune curve needs a local model; for any other model it is `None`.
- **size.** How much the prompt's outputs reveal about which context it holds, in bits. A fixed state built from the
  context (an adapter, a compiled or compressed prompt) that stores fewer bits must miss the prompt's outputs by at
  least the difference.

## bakeoff: which is better on my data?

```python
from bitsy import bakeoff

tuned = m.tune(docs, epochs=4)
a = bakeoff.score(m, docs, queries, L=32)        # the prompt reads the docs
b = bakeoff.score(tuned, [], queries, L=32)      # the fine-tune reads nothing; L: longest answer, fixed in advance
bakeoff.race(a, b)                               # {'call': 'A' prompt, 'B' fine-tune, '' not sure yet; 'queries': n}
```

A query is a pair `(cue, answer)` whose answer alone is scored. `race` is a betting test on the paired code lengths:
it can be checked after every query and stopped as soon as it calls. A wrong call has probability at most 5% per race
(`alpha`), counting runs that end undecided.

- **Queries must be drawn independently**, or pass `population=N` to test a fixed set of N shuffled once. The result
  then concerns that set, not a wider population.
- **Fix `L` in advance**: the longest answer, in tokens. Mixing each side with 1% of the uniform distribution bounds
  every query's cost, which the bets need. `robust=True` compares those mixed versions, which is what you would
  deploy: an overconfident side then cannot lose the call on a few outliers alone.
- **pool** replays many random query streams from a scored pool and reports how often each side is called, how often
  wrongly, and after how many queries. `examples/language.py` uses it.
- **labels** scores label tasks (classes, multiple-choice letters, YES/NO) from the first output token, so it runs on
  hosted chat models without echo:

  ```python
  bakeoff.labels(m, prompts, truths, names=('positive', 'negative'))
  ```

- **graded** takes bounded scores, higher is better, from any grader. It is the only bakeoff for APIs with no logprobs
  at all, such as Claude's.
- **paired** is the fixed-sample version: the mean difference with a normal interval. Fix the sample size in advance;
  `race` is the test you may stop at any time.

## What the model fails to use: diagnose.unused

```python
diagnose.unused(a, b, fits=True)                 # a, b: bakeoff.score of the prompt and a fine-tune
```

Any fine-tune's average advantage over the prompt is at most what the prompt fails to use. `unused` turns a measured
fine-tune win into a floor on that loss. If the context fits the window it is the prompt's decoding loss: information
the model has and fails to use. If not, it also counts what the window dropped. It is one-sided: a fine-tune that
loses proves nothing. It needs queries that neither the fine-tune nor its recipe saw.

## Other models

`gate.curves`, `gate.size` and `bakeoff` work with any model wrapped with `Fn` (see [Models](models.md)). A unit is
then whatever the model reads, such as a `(spec, observed)` example, and the fine-tune curve is `None`:

```python
k = Fn(baselines.Kernel(sites, grid=g))          # an in-context field learner: the support-set side
s = Fn(laws.cv_spread(fit, cases, grid=g))       # a trained surrogate with its spread from cross-validation
bakeoff.pool(bakeoff.score(k, support, held), bakeoff.score(s, [], held), robust=True)
gate.curves(k, cases)                            # the support set's learning curve
```

## Limits

- Small differences can stay undecided after thousands of queries.
- The 5% holds per race, not across many races.
- `bakeoff.paired` is a fixed-sample test; `race` is the one you may stop early.
- For the bounds that cover memories you did not build, see [Certificates](certificates.md).
