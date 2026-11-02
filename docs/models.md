# Models

bitsy works with three kinds of model: Hugging Face language models on your machine (`Local`), OpenAI-compatible
endpoints (`Hosted`, `Azure`), and any other model that gives a predictive distribution (`Fn`).

## What runs where

What a tool needs from the model:

| The model gives | Runs |
|---|---|
| Weights (`Local`) | Everything |
| Prompt logprobs (vLLM and other echo endpoints) | `bakeoff`, `gate.fit`, `diagnose.value`, `diagnose.unused`, `diagnose.items` (value), `guard` |
| Output-token logprobs (OpenAI, Azure chat) | `guard.strict`, `diagnose.spans`, `bakeoff.labels` |
| A predictive distribution (`Fn`) | `bakeoff`, `gate.curves`, `gate.size`, `diagnose.evidence`, `diagnose.items`, `guard.claims` |
| Point values (most simulation and vision APIs) | Wrap with `laws.spread`, then as above |
| Text only (Claude) | `bakeoff.graded` |
| Code lengths on designed data (any of the above) | `gate.capacity`, `gate.window`, `bakeoff.paired` |

## Language models on your machine: Local

```python
from bitsy import Local

m = Local('gpt2')                                # any Hugging Face causal LM
m = Local('gpt2', sep='\n', device='cuda')       # one context item per line, on a GPU
tuned = m.tune(docs, epochs=4)                   # a full fine-tune; returns a new model, m is unchanged
```

`Local` gives exact probabilities, so every tool runs. Context items are joined by the end-of-text token unless `sep`
names another single token, such as a newline. Install with `pip install -e '.[local]'`.

Text arguments may be strings or token-id lists. A context is a string, a token-id list, or a list of items read as
separate passages. A query is text, or a pair `(cue, answer)` whose answer alone is scored.

## Hosted models: Hosted and Azure

```python
from bitsy import Hosted, Azure

m = Hosted('gpt-4.1-mini')                       # OpenAI-compatible; reads OPENAI_API_KEY
m = Hosted('MODEL', base_url='http://localhost:8000/v1')                 # a vLLM server
m = Azure('chat', endpoint='https://<resource>.openai.azure.com')         # reads AZURE_OPENAI_API_KEY

m.labels(prompt)                                 # {'YES': (lo, hi), 'NO': (lo, hi), 'UNSURE': (lo, hi)}
```

Chat endpoints return probabilities for the tokens they write, not for the prompt. A held-out answer's code length
(`Hosted.bits`) therefore needs echo, which vLLM gives and OpenAI and Azure chat do not.

`labels()` needs only the first output token. A label's probability is the chance that token spells it (`Y`, ` Yes`,
`UN` count for YES, YES, UNSURE). The top 20 tokens give a range: labels inside it are summed exactly, and the mass
outside it bounds the rest. Local models give exact values.

Calls at temperature 0 are cached. Timeouts, refused connections, 429s and 5xx responses are retried with backoff.

## Any other model: Fn

```python
from bitsy import Fn, laws, baselines

m = Fn(predict)                                  # predict(evidence, spec) -> laws.Bernoulli | Categorical | Gaussian
s = Fn(laws.spread(point, held_out, grid=g))     # a point predictor, local or behind an API, with an error spread
s = Fn(laws.cv_spread(fit, cases, grid=g))       # or with its spread from cross-validation on its own cases
k = Fn(baselines.Kernel(sites, grid=g))          # an in-context field learner: the support-set side
m = Fn(predict, empty=lambda spec: prior)        # the prediction once every item is deleted, fixed in advance
m = Fn(predict, window=16)                       # a memory that reads only the last 16 items
```

Evidence is a list of deletable items (points, views, support cases); `spec` is the query, held fixed. A model returns
a predictive distribution per site from `laws`: yes/no (`Bernoulli`), bins (`Categorical`), or values on a fixed grid
(`Gaussian`).

Then:

```python
from bitsy import gate, bakeoff, diagnose, guard

diagnose.evidence(m, items, spec, [i], [j])      # bits deleted items supply, beyond a matched deletion
guard.claims(m, items, spec, [i], claim)         # per-site claims: required vs supplied bits
bakeoff.pool(bakeoff.score(k, support, held), bakeoff.score(s, [], held), robust=True)
gate.curves(k, cases)                            # the support set's learning curve
```

## Adapters

Ready-made wrappers for common models, in `bitsy.adapters`:

```python
from bitsy.adapters import Table, Chronos, field, occupancy, ConvONet, depth

Fn(Table(classifier, classes=[0, 1, 2]))         # any sklearn-style classifier, e.g. TabPFN or XGBoost
Fn(Chronos.load('amazon/chronos-t5-tiny'))       # forecasts on a fixed value grid; pip install -e '.[series]'
Fn(field(regressor, calibration, grid=g))        # a trained surrogate with a calibrated spread
Fn(occupancy(ConvONet.load(repo, config, ckpt))) # P(inside) per site, from patches of points
Fn(depth(net, bins, pixels))                     # MVSNet-style depth volumes -> a law over fixed depth bins per pixel
```

`ConvONet.load` has not been run against the released checkpoint here. `adapters.depth` was checked against the card
design's own reading on synthetic volumes (within 3e-9 bits per pixel), not on MVSNet itself.
