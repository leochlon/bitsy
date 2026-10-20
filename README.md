# bitsy

Should knowledge live in the prompt or in the weights? Is an answer supported by its evidence?

```
gate       don't pay for a fine-tune a prompt can match
bakeoff    prompt vs fine-tune, stop when sure
diagnose   missing information, or wasted information
guard      flag answers the evidence doesn't support
```

For any model that reads context: text, tables, 3D shapes, cameras, fields. Local or hosted. The core is about 1,250
lines of numpy. Anything that gives a predictive distribution plugs in through `Fn`; torch comes in only with the
models you choose to run on your own machine (language models, Chronos, TabPFN, TabICL).

## Quickstart

Which passages of a prompt does the answer depend on? Most of the five minutes is downloading torch and GPT-2 (550 MB).

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu    # optional on Linux: skips the CUDA download
pip install -e '.[local]'
```

```python
from bitsy import Local, diagnose

m = Local('gpt2', sep='\n')                       # any Hugging Face causal LM; one passage per line
passages = ['The code for Bob is 17.', 'The code for Alice is 52.', 'The code for Bob is not 71.',
            'Every code has two digits.', 'Fish swim in the sea.']
r = diagnose.items(m, passages, [('The code for Bob is', ' 17')])      # a question and its true answer
for x in r['items']:
    print(f"{x['value'][0]:+.2f} bits  {x['verdict']:12s}{passages[x['item']]}")
```

```
+3.99 bits  helps       The code for Bob is 17.
-1.74 bits  misleading  The code for Alice is 52.
-1.04 bits  misleading  The code for Bob is not 71.
+0.85 bits  helps       Every code has two digits.
-0.10 bits  dead weight Fish swim in the sea.
```

Each passage is deleted and the prompt read again; its value is the bits it saves on the true answer. The passage with
the answer helps, and two true passages make GPT-2's answer worse. Put in your own passages and question, then add
`controls=` to tell content from form ([below](#what-a-prompt-depends-on)). The last digit can differ across machines.

No torch? `pip install -e . && python examples/certificates.py` runs on numpy alone in about a second.

## Install

```bash
pip install -e .              # numpy only: hosted models and any model you wrap
pip install -e '.[local]'     # + torch, transformers: Hugging Face models on your machine
pip install -e '.[series]'    # + Chronos, for the forecasting adapter
```

## Language models

```python
from bitsy import Local, gate, bakeoff, diagnose, guard

m = Local('gpt2')                                # any Hugging Face causal LM

gate.fit(m, docs, query)                         # do the docs fit the window with the query?
gate.curves(m, units)                            # prompt and fine-tune learning curves, read from the data
gate.size(m, variants, prompts, cands, bits)     # can a state of `bits` bits ever match the prompt?

tuned = m.tune(docs, epochs=4)
a = bakeoff.score(m, docs, queries, L=32)        # the prompt reads the docs
b = bakeoff.score(tuned, [], queries, L=32)      # the fine-tune reads nothing; L: longest answer, fixed in advance
bakeoff.race(a, b)                               # 'A' prompt, 'B' fine-tune, '' not sure yet

diagnose.value(m, docs, matched, queries)        # bits from the docs' content vs their form
diagnose.unused(a, b, fits=True)                 # floor on what the model fails to use from its prompt
diagnose.items(m, passages, queries)             # which passages do the work, which are dead weight or mislead

guard.check(m, claim, evidence)                  # pass, flag, or prior (confident without the evidence)
```

## What a prompt depends on

`diagnose.items` tells you, for any prompt, which pieces of context do the work, which are dead weight and which
mislead the model. Each verdict comes as a number, the bits the item is worth on the actual question, and a reason
that says what to do next:

```
evidence      helps through what it says: real evidence the answer depends on. Keep it, and favour more like it.
format        helps only through how it looks: format or label space, not facts. A short template would do.
misleading    changes the answer but makes it worse: the model misreads information it has rather than missing
              it. A model problem, not a data problem, and the case where fine-tuning can win.
dead weight   adds nothing beyond the rest: redundant or irrelevant. Cut it; the prompt gets shorter and cheaper.
redundant     a group worth something together and nothing one at a time: its items repeat each other.
```

The quickstart's prompt, with a control for each passage:

```python
controls = ['The code for Kim is 45.', 'The code for Lee is 30.', 'The code for Ned is not 94.',
            'Every bird has two wings.', 'Birds fly in the sky.']    # same form, irrelevant content
r = diagnose.items(m, passages, [('The code for Bob is', ' 17')], controls=controls)
```

```
the whole context is worth +8.69 bits per query; 13 reads, re-reading 261 of their 443 context tokens
  item                                 value  influence  content    form   verdict
  The code for Bob is 17.              +3.99       0.58    +3.63   +0.36   evidence
  The code for Alice is 52.            -1.74       0.16    +0.03   -1.77   misleading
  The code for Bob is not 71.          -1.04       0.50    -0.53   -0.51   misleading
  Every code has two digits.           +0.85       0.12    -0.25   +1.10   format
  Fish swim in the sea.                -0.10       0.12    -0.11   +0.01   dead weight
```

On GPT-2, the passage with the answer saves 3.99 bits, almost all through what it says. The rule about two digits
helps only through its form, and any sentence of that shape does as well. Two true passages make the answer worse.
Alice's line costs 1.74 bits and its control costs as much, so any second line of the same shape makes 17 less likely;
the negation costs 1.04 bits. GPT-2 is misreading information it has, not missing information. The sentence about fish
does nothing.

What it measures, per item:

- **Value**: the code length of the true answer without the item minus with it. Positive helps, negative hurts. It
  needs an outcome: a label, an answer seen later, or, for a few-shot prompt, the other demonstrations. Pass
  `queries=None` and each demonstration is scored as a held-out query by the others, so no outside labels are needed.
- **Influence**: how far the item moves the prediction, in bits, with no labels at all. It shows how much an item
  matters, not whether it helps, so it audits a retrieval system or a claim's citations before any answer is checked.
  For a language model it is the change in the first answer token's distribution, a lower bound on the whole answer's.
- **Content and form**: with `controls`, one per item, of the same form with irrelevant content (another name, a
  shuffled label, an off-topic passage of the same length). Content is the item against its control; form is the
  control against nothing; they add up to the value.
- **Misuse**: a negative value. If the whole context fits the window, nothing was lost to truncation, so the model is
  misreading the item rather than missing it. If it does not fit, the reason says so.

Every item is deleted and the context read again, never masked: masking attention leaves the item's information in
the cache entries that follow it.

Leave-one-out measures what an item adds beyond the rest, so items that repeat each other each look worthless. Pass
`groups` to delete them together. In `examples/items.py`, each of eight few-shot demonstrations is worth at most 0.18
bits alone, yet deleting six of them costs 4.43 bits: they repeat each other, so any one can go, but not most of them.

Cost is one read of the context per item, two with controls, one per group, plus the full and empty contexts.

- `Local` keeps the cache of the prefix each read shares with the read before it, and items are deleted from last to
  first, so deleting item i re-reads only what comes after it. `tokens` reports what was re-read.
- `screen=k` deletes k contiguous blocks first and tests items one at a time only inside blocks that matter. A block
  that adds nothing can be cut as a whole. Effects that cancel inside a block can be missed.
- Controls are optional. Without them, `value` and `influence` still separate helpful, harmful and dead items.
- Score fewer queries to screen, then more on the items that matter.
- On hosted models every read is a new call. Reading from the last item to the first lets provider-side prefix
  caching reuse the shared beginning of the prompt.

## Hosted models

```python
from bitsy import Hosted, Azure, guard, diagnose, bakeoff

m = Hosted('gpt-4.1-mini')                       # OpenAI-compatible; base_url= for vLLM and others
m = Azure('chat', endpoint='https://<resource>.openai.azure.com')     # AZURE_OPENAI_API_KEY

m.labels(prompt)                                 # {'YES': (lo, hi), 'NO': (lo, hi), 'UNSURE': (lo, hi)}
guard.strict(m, claim, items, cites=[0])         # is the claim entailed by its cited items?
guard.strict_many(m, claims, items, group=8)     # several claims per call
diagnose.spans(m, claim, items)                  # bits each item supplies to the claim
bakeoff.labels(m, prompts, truths, names=('positive', 'negative'))   # label tasks, prompt vs fine-tuned model
```

Chat endpoints return probabilities for the tokens they write, not for the prompt, so a held-out answer's code length
(`Hosted.bits`) needs echo. `labels()` needs only the first output token. A label's probability is the chance that token
spells it (`Y`, ` Yes`, `UN` count for YES, YES, UNSURE). The top 20 tokens give a range: labels inside it are summed
exactly, and the mass outside it bounds the rest. Local models give exact values.

`guard.strict` uses a strict entailment prompt. Evidence goes in as quoted spans, and questions or instructions inside it
are masked. The baseline replaces the cited items with `[CITED_EVIDENCE_REMOVED]`. A claim passes only at the worst end
of every range. Each flag gives a reason: contradicted, ambiguous, not entailed, low gain, or error. A failed call flags
its claim and the run carries on. Calls at temperature 0 are cached, and timeouts, refused connections, 429s and 5xx
are retried.

## Any other model

```python
from bitsy import Fn, laws, baselines, gate, bakeoff, diagnose, guard

m = Fn(predict)                                  # predict(evidence, spec) -> laws.Bernoulli | Categorical | Gaussian
s = Fn(laws.spread(point, held_out, grid=g))     # a point predictor, local or behind an API, with an error spread
s = Fn(laws.cv_spread(fit, cases, grid=g))       # or with its spread from cross-validation on its own cases
k = Fn(baselines.Kernel(sites, grid=g))          # an in-context field learner: the support-set side
m = Fn(predict, empty=lambda spec: prior)        # the prediction once every item is deleted, fixed in advance
m = Fn(predict, window=16)                       # a memory that reads only the last 16 items

diagnose.evidence(m, items, spec, [i], [j])      # bits deleted items supply, beyond a matched deletion
guard.claims(m, items, spec, [i], claim)         # per-site claims: required vs supplied bits
bakeoff.pool(bakeoff.score(k, support, held), bakeoff.score(s, [], held), robust=True)
gate.curves(k, cases)                            # the support set's learning curve
```

Evidence is a list of deletable items (points, views, support cases); spec is the query, held fixed. A model returns a
predictive distribution per site from `laws`: yes/no, bins, or Gaussian values on a fixed grid.

```python
from bitsy.adapters import Table, Chronos, field, occupancy, ConvONet, depth

Fn(Table(classifier, classes=[0, 1, 2]))         # any sklearn-style classifier, e.g. TabPFN or XGBoost
Fn(Chronos.load('amazon/chronos-t5-tiny'))       # forecasts on a fixed value grid
Fn(field(regressor, calibration, grid=g))        # a trained surrogate with a calibrated spread
Fn(occupancy(ConvONet.load(repo, config, ckpt))) # P(inside) per site, from patches of points
Fn(depth(net, bins, pixels))                     # MVSNet-style depth volumes -> a law over fixed depth bins per pixel
```

## Certificates

`bakeoff` compares the memories you built. On data you design, where the true law of every answer is known, two
certificates also cover the memories you did not build:

```python
from bitsy import gate, bakeoff

gate.capacity(prompt, prior, C=8)    # the prompt beats every memory that stores at most 8 bits about its context
gate.window(memory, best)            # this memory beats every memory that reads only the window
bakeoff.paired(prompt, adapter)      # the measured winner of one pair you did build, on the same episodes
```

Each argument holds one code length per episode, and each episode draws a fresh task, its context and a block of
queries from the design. `prior` is the code length of the answers under their true law before any context is read;
its mean is their entropy, H(Q). `best` is the code length under the true law given only what the window holds; its
mean is H(Q | window).

- **Capacity.** A memory that stores at most C bits about the context costs at least H(Q) - C bits on average,
  whatever reads it: a fine-tune, an adapter, a compiled or compressed prompt of that size. A prompt that costs less
  beats all of them.
- **Window.** A memory that reads only the last W items or tokens costs at least H(Q | window) bits on average,
  whatever it does with them. A memory that costs less, such as a fine-tune on everything, beats all of them.

A certificate is issued when the 95% interval of its paired margin, prior - prompt - C or best - memory, lies above
zero. `bakeoff.paired` then checks it against each memory it covers that you did build.

`examples/certificates.py` replays three pretrained models, 64 episodes each:

```
model      design                                              window   capacity   checks
TabPFN v2  logistic tasks from a pool of 4,096; 8 to 1,024 rows  8/30     17/48      50 of 50 agree
ConvONet   a chair with 6 optional bumps; 12 point patches       2/5      3/6        14 of 14 agree
MVSNet     a card at one of 16 depths; 4 source views            2/3      4/4        22 of 22 agree
```

36 of 96 certificates are issued, and none of their 86 checks is a tie or goes the other way.

- TabPFN reading every row, and logistic regression trained on every row, beat every memory that reads only the last
  16 rows, from 128 rows on. XGBoost gets no window certificate, and nothing is issued against the last 64 rows.
- TabPFN beats every memory of at most 2 bits from 64 rows, 5 bits from 128, 6 bits from 256 and 8 bits at 1,024.
- ConvONet reading all 12 patches beats every memory confined to 2 or 4 of them, and every memory of at most 2 bits.
- MVSNet reading all 4 views beats every memory confined to 1 or 2 views, and every memory of at most 3 bits.
- With z = 3.28, which bounds the chance of any false certificate among the 96 by 5%, 32 are issued and all 71 of
  their checks agree.

TabPFN ran from its v2 classifier checkpoint with 8 estimators, ConvONet from the official shapenet_3plane_partial
checkpoint on CPU, and MVSNet from the original TensorFlow 1 checkpoint (model.ckpt-150000) through a separate Python 2
worker, which is not included here. `examples/chair.py` runs the ConvONet design live, through a numpy stand-in or the
pretrained network, and deletes each patch with `diagnose.items`; seed 0 draws the recorded episodes.

What these results cover:

- The data is designed so that the true laws are exact. On real data H(Q) is unknown.
- A capacity certificate covers only memories smaller than H(Q), the bits in the answers: here about 32, 6 and 4 bits.
  A fine-tune's weights store far more.
- The windows are imposed by the design: TabPFN is given only the last rows, the networks only the first patches or
  views.
- The checks cover the bounded memories that were built: logistic regressions on a grid of 2 or 4 values per weight,
  the best of 2^k fixed classifiers, and states storing k bumps or k bits of depth, read by the true law and by the
  network.
- The intervals are pointwise normal ones, and the same episodes give both the margins and the measured winners.

## Examples

```
python examples/shapes.py        3D shape from surface points: which points a prediction relies on   numpy
python examples/camera.py        depth for a new camera from the views already taken                numpy
python examples/fields.py        support set vs trained surrogate on surface fields                 numpy
python examples/race.py          the betting test on simulated streams: wrong calls and speed       numpy
python examples/airfrans.py      airfoil pressure from RANS simulations: support set vs MLP         scikit-learn
python examples/tables.py        TabPFN and TabICL vs XGBoost on OpenML tables                      tabpfn tabicl xgboost
python examples/language.py      GPT-2: gate, prompt vs fine-tune, diagnose, guard                  '.[local]'
python examples/vitaminc.py      claims against evidence, on a local or hosted model                a model or an API key
python examples/items.py         which context items do the work, mislead, or can be cut            '.[local]'
python examples/certificates.py  certificates on TabPFN, ConvONet and MVSNet: 64 recorded episodes  numpy
python examples/chair.py         both certificates and a patch audit on an occupancy network, live  numpy, or ConvONet
```

Each script explains its setup at the top and prints one line per result. `examples/data/airfrans-pressure64.npz`
holds 1,000 AirfRANS simulations (Bonnet et al., NeurIPS 2022), each reduced to 18 design inputs and 64 surface
pressure coefficients, under the Open Database Licence 1.0. `examples/data/certificates-64.npz` holds the per-episode
code lengths of the three certificate runs. `examples/chair.py` takes about a minute with the stand-in.

## What runs where

```
the model gives                                   runs
weights (Local)                                   everything
prompt logprobs (vLLM and other echo endpoints)   bakeoff, gate.fit, diagnose.value, unused, items (value), guard
output-token logprobs (OpenAI, Azure chat)        guard, diagnose.spans, bakeoff.labels
a predictive distribution (Fn)                    bakeoff, gate.curves, gate.size, diagnose.evidence, items, guard.claims
point values (most simulation and vision APIs)    wrap with laws.spread, then as above
text only (Claude)                                bakeoff.graded
code lengths on designed data (any of the above)  gate.capacity, gate.window, bakeoff.paired
```

## How it works

Everything is a code length: -log2 of the probability the model gave to what actually happened, in bits.

The idea is to delete context and ask again. Deleting part of the context and re-reading the query is held-out
validation done inside one forward pass. The bits a model saves on held-out answers when it reads a context measure what
that context teaches it. Deleting one piece of evidence and asking again measures what that piece is worth to the model,
with no labels needed.

- **Where a context's value comes from.** On average, the bits a context saves on a held-out query are what it says
  about the query, plus how much it corrects the model's default guesses whatever its content, minus what the model
  fails to use.
- **gate.** If the context fits in the window, the prompt holds all of it, so a fine-tune on the same data can only win
  by what the prompt fails to use. `curves` scores each unit before the model has seen it, once with the earlier units
  in the prompt and once after online fine-tuning on them, which gives both learning curves from the data itself.
  `size` measures how much the prompt's outputs reveal about which context it holds; a fixed state with fewer bits (an
  adapter, a compressed prompt) must miss the prompt's outputs by at least the difference. `capacity` and `window`
  compare against the true answers instead: a state of C bits leaves at least H(Q) - C bits of uncertainty about the
  answers, and a reader of the window at least H(Q | window), so a memory below either bound beats every memory of
  that kind.
- **bakeoff.** A betting test on paired code lengths that can be checked after every query and stopped as soon as it
  calls. A wrong call has probability at most 5% per race, counting runs that end undecided. Queries must be drawn
  independently, or pass `population=N` to test a fixed set shuffled once. Mixing each side with 1% of the uniform
  distribution bounds every query's cost, which the bets need.
- **diagnose.** Deleting context is held-out validation. A matched deletion, same form with irrelevant content,
  separates what the content adds from what the format does.
- **guard.** Lifting a claim from probability q to target t takes at least d2(t‖q) bits of evidence, where d2 is the
  binary KL divergence. A claim passes when p ≥ t with the evidence and q < t without it; prior means the model was
  already confident without the evidence.

The derivations, proofs and experiments are in the paper under [Citation](#citation).

## Files

```
core.py        bits, budgets and the betting test
laws.py        Bernoulli, Categorical, Gaussian; error spreads for point predictors
models.py      Local, Hosted, Azure, Fn
gate.py        fit, curves, size, capacity, window
bakeoff.py     score, labels, graded, race, pool, paired
diagnose.py    value, unused, evidence, spans, items
guard.py       check, claims, strict, strict_many
baselines.py   Kernel, an in-context field learner
adapters.py    Table, Chronos, field, occupancy, ConvONet, depth
examples/      the scripts above, the AirfRANS data and the certificate runs
```

## Limits

- Free-text code lengths on chat endpoints need prompt logprobs (echo). OpenAI and Azure chat don't return them; a
  vLLM server does.
- APIs with no logprobs at all, such as Claude's, only run `bakeoff.graded`.
- Hosted label ranges come from the top 20 tokens, so close calls can stay ambiguous. `strict_many(group>1)`
  conditions each answer on the ones above it; `group=1` keeps one claim per call.
- Small differences can stay undecided after thousands of queries. The 5% holds per race, not across many races.
- `guard.check` on a local instruction model needs transformers < 5. `guard.strict` works with both.
- The Guard's 0.95 threshold is a default, not a calibration for your data.
- `diagnose.items` needs code lengths or predictions. On chat endpoints without echo, check claims against their
  evidence with `diagnose.spans` and `guard.strict`. Its verdicts need at least `tol` bits per query (0.5 by
  default) and, over several queries, an interval that excludes zero; an item that matters for one question in
  ten can fall below that on average, so `per_query` shows which question each item serves.
- Certificates need the true law of every answer, so they run on designed data only. Their intervals are pointwise:
  raise `z` when testing many settings. `bakeoff.paired` is a fixed-sample test; `race` is the one you may stop early.
- `ConvONet.load` has not been run against the released checkpoint here. `adapters.depth` was checked against the card
  design's own reading on synthetic volumes (within 3e-9 bits per pixel), not on MVSNet itself.

## Citation

Leon Chlon and Julie Huang. When Thoughts Become Memory: Unifying In-Context Learning and Fine-Tuning. arXiv preprint,
2026.

```bibtex
@misc{chlon2026thoughts,
  title  = {When Thoughts Become Memory: Unifying In-Context Learning and Fine-Tuning},
  author = {Chlon, Leon and Huang, Julie},
  year   = {2026},
  note   = {arXiv preprint}
}
```
