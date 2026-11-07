# bitsy

[Paper](#citation) · [Docs](docs/README.md) · [Examples](#examples) · [Licence](LICENSE)

[![Licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)

bitsy is a tool for measuring, in bits, what a model gets from its context.

It deletes a piece of the context, asks the model again, and measures how much more or less likely the right answer
becomes. Each bit is a factor of two. That one measurement answers four questions:

- **`diagnose`**: which parts of my prompt help, which mislead the model, and which do nothing?
- **`guard`**: is this answer supported by its evidence?
- **`bakeoff`**: prompt, fine-tune or compiled weights, which is better on my data? It stops as soon as it is sure.
- **`gate`**: is a fine-tune worth trying at all?

It works with Hugging Face models on your machine, OpenAI-compatible APIs (OpenAI, Azure, vLLM), and any model you
wrap that gives a predictive distribution: over text, tables, 3D shapes, cameras or fields. The core is about 1,700
lines of Python and needs only numpy; torch comes in only for models you run locally.

## Install

```bash
git clone https://github.com/leochlon/bitsy && cd bitsy
pip install -e .              # numpy only: hosted models and any model you wrap
pip install -e '.[local]'     # adds torch and transformers, for Hugging Face models on your machine
pip install -e '.[series]'    # adds Chronos, for the forecasting adapter
```

On Linux, run `pip install torch --index-url https://download.pytorch.org/whl/cpu` first to skip the CUDA download.

## Quickstart

Which passages of a prompt does GPT-2's answer depend on?

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

Each passage is deleted and the prompt read again. Its value is the bits it saves on the true answer: the passage
with the answer makes 17 about 16 times as likely (3.99 bits). Two true passages make GPT-2's answer worse, and the
sentence about fish does nothing. The first run downloads GPT-2 (550 MB). The last digit can differ across machines.

No torch? `python examples/certificates.py` runs on numpy alone in under a second.

## diagnose: what does the answer depend on?

Give each passage a control with the same form and irrelevant content, and its value splits into what it says
(content) and how it looks (form):

```python
controls = ['The code for Kim is 45.', 'The code for Lee is 30.', 'The code for Ned is not 94.',
            'Every bird has two wings.', 'Birds fly in the sky.']
r = diagnose.items(m, passages, [('The code for Bob is', ' 17')], controls=controls)
```

```
  item                                 value  influence  content    form   verdict
  The code for Bob is 17.              +3.99       0.58    +3.63   +0.36   evidence
  The code for Alice is 52.            -1.74       0.16    +0.03   -1.77   misleading
  The code for Bob is not 71.          -1.04       0.50    -0.53   -0.51   misleading
  Every code has two digits.           +0.85       0.12    -0.25   +1.10   format
  Fish swim in the sea.                -0.10       0.12    -0.11   +0.01   dead weight
```

The rule about two digits helps only through its form, so a short template would do. Alice's line costs as much as
its control, so any second line of that shape makes 17 less likely: GPT-2 is misreading information it has, not
missing it. Each row also says what to do next, in `r['items'][i]['why']`.

More in [docs/diagnose.md](docs/diagnose.md): items that repeat each other, long contexts, and cost. When a prompt
fails to use what it reads, [docs/where.md](docs/where.md) (experimental) narrows down where inside the model the
bits are lost.

## guard: is the answer supported?

```python
from bitsy import Hosted, guard

m = Hosted('gpt-4.1-mini')                       # reads OPENAI_API_KEY; base_url= for vLLM and others
evidence = ['The Eiffel Tower stands in Paris, France.', 'It was completed in 1889.']
r = guard.strict(m, 'The Eiffel Tower is in Rome.', evidence)
r['verdict'], r['why']                           # e.g. ('flag', 'contradicted')
```

guard asks the model twice, with the evidence and with it removed. A claim passes when the evidence is what makes the
model confident (above 0.95 by default). It is flagged when the model is not confident even with the evidence, with a
reason: contradicted, ambiguous or not entailed. It is `prior` when the model was confident without the evidence, so
the evidence is not why. It reads only the first output token, so hosted chat models need no prompt logprobs.

More in [docs/guard.md](docs/guard.md): label ranges on hosted models, grouped claims, and per-site claims.

## bakeoff: prompt, fine-tune or compile?

In `examples/language.py`, `doc` is a 512-token passage of Shakespeare, and each of 64 `queries` gives 8 tokens of it
and scores the next 24. The prompt reads the passage; the fine-tune is trained on it and reads nothing:

```python
a = bakeoff.score(m, doc, queries, L=24)                          # L: the longest answer, fixed in advance
b = bakeoff.score(m.tune([doc], epochs=1), [], queries, L=24)
bakeoff.race(a, b)
# {'call': 'A', 'queries': 27}
```

The prompt is better, and the race called it after 27 of the 64 queries. You can check it after every query and stop
when it calls; a wrong call has probability at most 5%. After 16 epochs the fine-tune is better, 0.29 bits per token
against the prompt's 1.69, but 64 queries are not enough to call it, and the race returns `''`, not sure yet.

`m.compile` (experimental) is a third contestant. It writes the passage into the late MLP layers in closed form, with
no gradients, so that the model reading nothing answers as it does with the passage in its prompt. `keep`, other text
that should stay where it is, is required. In `examples/install.py`, 64 recall queries drive the solve, 416 more ask
the same passage from cue positions the solve did not see, and `other` is 512 short passages from elsewhere in the
play:

```python
c = m.compile([(doc, train)], keep, only=m.block('mlp', 'late'))   # a new model; m is unchanged
a, b = bakeoff.score(m, doc, held, L=24), bakeoff.score(c, [], held, L=24)
bakeoff.race(b, a, margin=24 * 1.0)                      # 'A': c costs less than 1 bit/token more than the prompt
# {'call': 'A', 'queries': 382}
d, e = bakeoff.score(c, [], other, L=32), bakeoff.score(m, [], other, L=32)
bakeoff.race(d, e, margin=32 * 0.25)                     # 'A': other text within 0.25 bits/token of the original
# {'call': 'A', 'queries': 405}
```

A plain race says which side costs less: here the prompt, 1.72 bits per token against 2.47. `margin` asks whether A
costs less than m more than B, in cost per query, so a tolerance per token is multiplied by the tokens per query. 'B'
means it costs more than that, and undecided is not a pass. keep decides the damage: with 128 passages other text
gets 0.17 bits per token worse, with 16 it got 2.3 worse, so check it every time.

## gate: is a fine-tune worth trying?

```python
gate.fit(m, doc, 32)['read']
# 'fits: a fine-tune can only win by what the model fails to use from the prompt'
```

If the context fits the window, the prompt already holds all of it, so a fine-tune can only win by what the model
fails to use. `diagnose.unused` turns a fine-tune's win into a floor on that: after 16 epochs, GPT-2 fails to use at
least 1.25 bits per token of the passage. `gate.curves` reads both learning curves from your data, and `gate.size` says
whether an adapter of a given size could ever match the prompt.

More in [docs/prompt-vs-finetune.md](docs/prompt-vs-finetune.md), and [docs/certificates.md](docs/certificates.md)
for designed data, where a prompt can be shown to beat every memory of a given size.

## Which models run what

| Model | Runs |
|---|---|
| `Local`: Hugging Face weights | everything, and `tune` and `compile` to put a context into the weights |
| `Hosted` with prompt logprobs: vLLM and other echo endpoints | `bakeoff`, `gate.fit`, `diagnose.value`, `diagnose.unused`, `diagnose.items`, `guard` |
| `Hosted`, `Azure`: OpenAI and Azure chat | `guard.strict`, `diagnose.spans`, `bakeoff.labels` |
| `Fn`: any model with a predictive distribution | `bakeoff`, `gate.curves`, `gate.size`, `diagnose.evidence`, `diagnose.items`, `guard.claims` |
| Text only, such as Claude | `bakeoff.graded` |

Point predictors, such as most simulation APIs, are wrapped with `laws.spread`. More in [docs/models.md](docs/models.md).

## Examples

| Script | What it shows | Needs |
|---|---|---|
| `items.py` | which context items do the work, mislead, or can be cut | `.[local]` |
| `language.py` | GPT-2: gate, prompt vs fine-tune, diagnose, guard | `.[local]` |
| `install.py` | GPT-2: a passage compiled into the weights, and a fine-tune's change copied into the MLPs where it helps; each scored for damage to other text | `.[local]` |
| `where.py` | where inside GPT-2 a prompt loses bits (experimental) | `.[local]` |
| `vitaminc.py` | claims against evidence, on a local or hosted model | a model or an API key |
| `race.py` | the betting test on simulated streams: wrong calls and speed | numpy |
| `certificates.py` | certificates on TabPFN, ConvONet and MVSNet: 64 recorded episodes | numpy |
| `chair.py` | both certificates and a patch audit on an occupancy network, live | numpy, or ConvONet |
| `shapes.py` | 3D shape from surface points: which points a prediction relies on | numpy |
| `camera.py` | depth for a new camera from the views already taken | numpy |
| `fields.py` | support set vs trained surrogate on surface fields | numpy |
| `airfrans.py` | airfoil pressure from RANS simulations: support set vs MLP | scikit-learn |
| `tables.py` | TabPFN and TabICL vs XGBoost on OpenML tables | tabpfn, tabicl, xgboost |

Run them as `python examples/<script>`. Each explains its setup at the top and prints one line per result.
`examples/data/airfrans-pressure64.npz` holds 1,000 AirfRANS simulations (Bonnet et al., NeurIPS 2022), each reduced
to 18 design inputs and 64 surface pressure coefficients, under the Open Database Licence 1.0.

## How it works

Every number bitsy reports is a code length: the bits a model spends on what actually happened. Deleting part of the
context and asking again is held-out validation inside one forward pass, so the bits the model loses measure what that
part taught it, with no training. The theory behind each tool is in [docs/how-it-works.md](docs/how-it-works.md) and
the paper.

## What it doesn't do

- Free-text bits on a chat API need prompt logprobs (echo). OpenAI and Azure chat don't return them, so they run
  only `guard.strict`, `diagnose.spans` and `bakeoff.labels`; a vLLM server returns them and runs most tools. APIs
  with no logprobs at all, such as Claude's, run only `bakeoff.graded`.
- The guard's 0.95 threshold is a default, not a calibration for your data.
- Small differences can stay undecided after thousands of queries, and the 5% error holds per race, not across many.
  An undecided margin race is not a pass.
- `compile` is experimental and writes MLP output projections only. Its solve never sees the text it may damage, so
  measure that with a margin race.
- Deleting one item at a time misses items that repeat each other; pass `groups` to delete them together.
- Certificates need the true law of every answer, so they run on designed data only.

The full list is in [docs/how-it-works.md](docs/how-it-works.md#all-limits).

## Citation

```bibtex
@misc{chlon2026thoughts,
  title  = {When Thoughts Become Memory: Unifying In-Context Learning and Fine-Tuning},
  author = {Chlon, Leon and Huang, Julie},
  year   = {2026},
  note   = {arXiv preprint}
}
```

## Licence

[MIT](LICENSE).
