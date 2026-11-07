# Prompt or fine-tune: gate, bakeoff and compile

Should knowledge live in the prompt or in the weights? `gate` tells you whether a fine-tune is worth trying before
you train one. `bakeoff` compares a prompt and a fine-tune you have built, on the same held-out queries, and stops as
soon as it is sure which is better. `Local.compile` (experimental) is a third contestant: it puts a context into the
weights without gradients, and `bakeoff` scores it like a fine-tune.

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

## margin: no worse than, within a tolerance

A plain race says which side costs less. A model that replaces the prompt, or an edit that should leave other text
alone, needs a different claim: no worse than the other side by more than a tolerance. `margin` makes that the test:

```python
bakeoff.race(b, a, margin=24 * 1.0)              # 'A': b costs less than 1 bit/token more than a; 'B': more
bakeoff.race(edited, original, margin=32 * 0.25) # on other text: damage within 0.25 bits/token
```

- With `margin=m`, 'A' means A costs less than m more than B on average, and 'B' means it costs more than m more. Put
  the new model as A. `margin=0`, the default, is the plain comparison, and its calls are unchanged.
- m is in cost units per query: bits for `score`, score points for `graded`. A tolerance per token is m divided by the
  tokens per query, so give the answers one length and multiply.
- Undecided is not a pass: only 'A' shows the new side is within the tolerance. The 5% bound on a wrong call holds as
  before.
- `pool(margin=m)` replays streams with the same two hypotheses. `paired(a, b, margin=m)` is the fixed-sample version.

## compile: a context into the weights, without gradients (experimental)

```python
c = m.compile([(doc, train)], keep, only=m.block('mlp', 'late'))   # jobs: (context, queries); keep: other text
a = bakeoff.score(m, doc, held, L=24)            # the prompt, on queries the solve did not see
b = bakeoff.score(c, [], held, L=24)             # compiled: reads nothing, like a fine-tune
bakeoff.race(b, a, margin=24 * 1.0)              # within 1 bit/token of the prompt?
bakeoff.race(bakeoff.score(c, [], other, L=32), bakeoff.score(m, [], other, L=32), margin=32 * 0.25)   # damage
c.written                                        # the projections changed, the rank, each change's relative size
```

`compile` reads each query twice: with its context, by this model, and without it, by the model being written. For
each selected MLP output projection, shallowest first, it solves in closed form (least squares on the projection's
inputs) for the change that moves the layer's output without the context to its output with it, on the query tokens,
and writes that change before it reads the next layer. No gradients are taken. It returns a new model; this one is
unchanged.

- **keep is required**: texts or token-id lists, read with no context, whose layer outputs the solve holds where this
  model puts them. Without them it moves ordinary text as much as the queries, and the less they cover, the more it
  moves: in the example, 16 keep passages left other text 2.3 bits per token worse, 128 left it 0.17 worse. Use
  hundreds. `weight` is a query token's weight against one keep token (3 by default).
- **only** picks the layers, as in `tune()`; only MLP output projections are written. **rank=r** keeps the rank-r
  part of each change that carries most of its output. Each **epoch** after the first re-solves every layer for the
  gap left: it fits the queries more closely and can move held-out text more, so check both.
- **ridge** is relative to the mean energy of the projection's inputs (1e-4 by default). A larger ridge shrinks each
  change, and the damage with it, at some cost in recall.
- A job `(context, queries, kept)` leaves `kept`, the start of the context, in the prompt: the reading without the
  context reads `kept`, so only the rest goes into the weights. Score that model with `kept` as its context.
- **Score it like a fine-tune**, on queries the solve did not see: against the prompt with a margin, and on other text
  against this model for damage. `c.written['relative']` gives each change's norm over its weights' norm; a change
  several times the weights is worth checking for damage first.

Part 1 of `examples/install.py` writes the passage into GPT-2's last four MLP layers from 64 recall queries, keeps 128 other
passages of the play in place, and scores 416 queries at cue positions the solve did not see, over the same passage,
and 512 short passages from elsewhere in the play:

```
prompt          recall 1.72 bits/token; the original model, reading nothing, 7.32; other text 7.86
compile         4 projections written, changes 0.50-4.14 times their weights' norm
recall          2.47 bits/token. Better than the original: yes after 26 queries
vs prompt       better: no after 139 queries; within 1 bit/token: yes after 382 queries
other text      8.03 bits/token. Within 0.25 bits/token of the original: yes after 405 queries
```

The prompt is better, and the compiled model, reading nothing, is within 1 bit per token of it. Other text gets 0.17
bits per token worse, within the 0.25 tolerance.

Part 2 writes in right answers that no prompt produces. It fine-tunes a copy on 256 passages of the play (one Adam step
each), puts each block's MLP share of that change alone into the original, and scores it on 128 more right answers and
on the model's own answers to them. The output projections of the third of the MLPs whose shares help most are then
solved so that each MLP makes its own share of the fine-tune's change on the training passages and none on 128
passages the model writes itself. 128 new right answers and 640 more such passages judge it:

```
fine-tune       right answers -2.042 bits/token against the original, its own answers +2.222
one MLP         right / own answers, by layer: 0:-0.342/+0.156 ... 5:-0.545/+0.035 ... 11:-0.240/+0.101
solved          MLPs [3, 4, 5, 6]: new right answers -1.307 bits/token (their shares -1.404, the whole fine-tune -1.877). Better than the original: yes after 92 queries
text it writes  solved +0.011 bits/token, fine-tune +2.597. Solved within 0.25 of the original: yes after 491 queries
```

The solved MLPs keep about 70% of the fine-tune's gain on new right answers and move text the model writes by 0.011
bits per token, where the fine-tune moves it by 2.6.

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
- An undecided margin race is not a pass.
- `compile` is experimental. It writes MLP output projections only, and its solve sees neither your held-out queries
  nor the text it may damage: measure both.
- For the bounds that cover memories you did not build, see [Certificates](certificates.md).
