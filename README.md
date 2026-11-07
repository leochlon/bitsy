# bitsy

[Paper](#citation) · [Docs](docs/README.md) · [Examples](#examples) · [Licence](LICENSE)

bitsy measures, in bits, what a model gets from its context. It deletes part of the context, asks the model again, and
measures how much more or less likely the right answer becomes. Each bit is a factor of two. The core is about 1,700
lines of Python and needs only numpy.

| Tool | What it answers | Start with |
|---|---|---|
| [`diagnose`](docs/diagnose.md) | Which parts of my prompt help, which mislead the model, and which do nothing? | `diagnose.items` |
| [`guard`](docs/guard.md) | Is this answer supported by its evidence? | `guard.strict` |
| [`bakeoff`](docs/prompt-vs-finetune.md#bakeoff-which-is-better-on-my-data) | Prompt, fine-tune or compiled weights: which is better on my data? It stops as soon as it is sure. | `bakeoff.score`, `bakeoff.race` |
| [`gate`](docs/prompt-vs-finetune.md#gate-is-a-fine-tune-worth-trying) | Is a fine-tune worth trying at all? | `gate.fit` |
| [`compile`](docs/prompt-vs-finetune.md#compile-a-context-into-the-weights-without-gradients-experimental) | Experimental: writes a context into a local model's MLP output projections in closed form, with no gradients. `bakeoff` scores it like a fine-tune. | `Local.compile` |

The docs also cover [where a prompt loses bits](docs/where.md) inside a local model (experimental),
[certificates](docs/certificates.md) on designed data, and [how it works](docs/how-it-works.md).

## Install

```bash
git clone https://github.com/leochlon/bitsy && cd bitsy
pip install -e .              # numpy only: hosted models and any model you wrap
pip install -e '.[local]'     # adds torch and transformers, for Hugging Face models on your machine
pip install -e '.[series]'    # adds Chronos, for the forecasting adapter
```

Requires Python 3.9+. On Linux, run `pip install torch --index-url https://download.pytorch.org/whl/cpu` first to skip
the CUDA download.

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

Each passage is deleted and the prompt read again: the one with the answer makes 17 about 16 times as likely (3.99
bits), and two true passages make it less likely. The first run downloads GPT-2 (550 MB), and the last digit can differ
across machines. No torch? `python examples/certificates.py` runs on numpy alone in under a second.

## Models

| Model | Runs |
|---|---|
| `Local`: Hugging Face weights | everything, plus `tune` and `compile` |
| `Hosted` with prompt logprobs: vLLM and other echo endpoints | `bakeoff`, `gate.fit`, `diagnose.value`, `diagnose.unused`, `diagnose.items`, `guard` |
| `Hosted`, `Azure`: OpenAI and Azure chat | `guard.strict`, `diagnose.spans`, `bakeoff.labels` |
| `Fn`: any model with a predictive distribution, over text, tables, 3D shapes, cameras or fields | `bakeoff`, `gate.curves`, `gate.size`, `diagnose.evidence`, `diagnose.items`, `guard.claims` |
| Text only, such as Claude | `bakeoff.graded` |

Point predictors are wrapped with `laws.spread`. Adapters and more in [docs/models.md](docs/models.md).

## Examples

Run any with `python examples/<name>.py`. Each explains its setup at the top and prints one line per result.

- **numpy only:** `race` (the betting test), `certificates` (TabPFN, ConvONet and MVSNet, 64 recorded episodes),
  `chair` (an occupancy network, live), `shapes` (which points a 3D shape prediction relies on), `camera` (depth for a
  new view), `fields` (support set vs trained surrogate)
- **`.[local]`, on GPT-2:** `items` (diagnose), `language` (all four tools), `install` (compile, and a fine-tune's change
  copied into the MLPs), `where` (experimental)
- **Other:** `vitaminc` (guard; a model or an API key), `airfrans` (airfoil pressure, support set vs MLP; scikit-learn),
  `tables` (TabPFN and TabICL vs XGBoost; tabpfn, tabicl, xgboost)

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

[MIT](LICENSE). `examples/data/airfrans-pressure64.npz` holds 1,000 AirfRANS simulations (Bonnet et al., NeurIPS
2022), each reduced to 18 design inputs and 64 surface pressure coefficients, under the Open Database Licence 1.0.
