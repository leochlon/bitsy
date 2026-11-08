<div align="center">

# bitsy

**How much did your model learn from its prompt, a fine-tune or a weight edit?<br>
bitsy measures all three in one unit: bits.**

<h3>

[Paper](https://arxiv.org/abs/2610.29419) | [Docs](docs/README.md) | [Examples](examples/)

</h3>

</div>

---

Reading something in the prompt and being trained on it are both ways for a model to learn it. bitsy treats every
prompt, fine-tune, adapter and weight edit as memory and scores it the same way. It takes the model's right answers on
held-out questions, first with nothing, then with the memory. The bits saved are what the memory taught it: each bit
makes the right answer twice as likely.

- **Attribute it**: how many bits each part of a context contributes, including true lines the model misreads
  (`diagnose`), and whether an answer's confidence comes from its evidence (`guard`).
- **Compare memories**: prompt against fine-tune, adapter or edited weights, with a sequential test you can check
  after every question. It stops when it calls a winner, wrong at most 5% of the time per comparison (`bakeoff`).
- **Move it**: write a prompt into the weights, solved in closed form with no gradient steps, while holding other text
  in place, and measure what moved (`compile`).
- **Find what is lost**: a floor on how much of its prompt a model fails to use, and where inside a local model
  ([where](docs/where.md)).
- **Before you train**: whether a fine-tune is worth trying (`gate`), and, on data whose true answers are known, a
  certificate that a prompt beats every memory of at most C bits, such as any small adapter
  ([certificates](docs/certificates.md)).

## Try it

### Which lines of its prompt does a model learn from?

Each line is deleted and the answer scored again. A line's value is the bits it saves on the true answer.

```python
from bitsy import Local, diagnose

m = Local('gpt2', sep='\n')                       # any Hugging Face causal LM; one passage per line
passages = ['The code for Bob is 17.', 'The code for Alice is 52.', 'The code for Bob is not 71.',
            'Every code has two digits.', 'Fish swim in the sea.']
r = diagnose.items(m, passages, [('The code for Bob is', ' 17')])      # a question and its true answer
for x in r['items']:
    print(f"{x['value'][0]:+.2f} bits  {x['verdict']:12s}{passages[x['item']]}")   # [0]: the one question
```

```
+3.99 bits  helps       The code for Bob is 17.
-1.74 bits  misleading  The code for Alice is 52.
-1.04 bits  misleading  The code for Bob is not 71.
+0.85 bits  helps       Every code has two digits.
-0.10 bits  dead weight Fish swim in the sea.
```

The line with the answer makes 17 about 16 times as likely. Two true lines cost bits: the verdict is how GPT-2 uses a
line, not whether it is true. The first run downloads GPT-2 (550 MB); the last digit can vary by machine.

### How much does a fine-tune learn, compared with reading?

A 512-token passage of Shakespeare, either in the prompt or fine-tuned in for one epoch. Each of 64 questions shows 8
tokens of it and scores the next 24.

```python
import numpy as np, urllib.request
from bitsy import bakeoff

url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
ids = np.array(m.enc(urllib.request.urlopen(url).read().decode()))
doc = ids[100000:100512]
queries = [(doc[s:s + 8], doc[s + 8:s + 32]) for s in np.random.default_rng(0).integers(0, 480, 64)]
none = bakeoff.score(m, [], queries, L=24)                        # L: tokens scored per question, fixed in advance
a = bakeoff.score(m, doc, queries, L=24)                          # the passage in the prompt
b = bakeoff.score(m.tune([doc], epochs=1), [], queries, L=24)     # fine-tuned, nothing in the prompt
print([round(float(s['cost'].mean()) / 24, 2) for s in (none, a, b)])   # bits per token
# [7.55, 1.69, 6.19]
bakeoff.race(a, b)                                                # 'A': the first argument is the better memory
# {'call': 'A', 'queries': 27}
```

Reading nothing, GPT-2 needs 7.55 bits per token. The prompt saves 5.86 of them, the fine-tune 1.36, and the test was
sure after 27 of the 64 questions.

### The same memory, moved into the weights

[examples/install.py](examples/install.py) writes the passage into GPT-2's last four MLP layers from 64 questions,
holding 128 other passages in place, then scores 416 new questions with nothing in the prompt. Its core is one call:

```python
c = m.compile([(doc, train)], keep, only=m.block('mlp', 'late'))   # train: 64 questions; keep: 128 other passages
```

| | the passage (bits/token) | other text (bits/token) |
|---|---|---|
| nothing | 7.32 | 7.86 |
| the passage in the prompt | 1.72 | |
| compiled weights, nothing in the prompt | 2.47 | 8.03 |

The weights now hold 87% of what the prompt taught (4.85 of 5.60 bits per token), and other text moved by 0.17. The
test called it within 1 bit per token of the prompt after 382 questions, and other text within 0.25 after 405.

[notebooks/install_a_skill.ipynb](notebooks/install_a_skill.ipynb) does the same with a 183-token system prompt in
Qwen2.5-1.5B-Instruct: the edited model, with no prompt, keeps 97% of what the prompt teaches.
[Open it in Colab](https://colab.research.google.com/github/leochlon/bitsy/blob/main/notebooks/install_a_skill.ipynb).

## Runs on

- [x] Hugging Face models on your machine (`Local`): everything, including fine-tuning and `compile`
- [x] vLLM and other servers that return prompt log-probabilities (`Hosted`): `bakeoff`, `gate.fit`, `guard`,
  `diagnose.value`, `diagnose.unused`, `diagnose.items`
- [x] OpenAI and Azure chat (`Hosted`, `Azure`): `guard.strict`, `diagnose.spans`, `bakeoff.labels`
- [x] Any model that outputs a probability distribution (`Fn`): tables, 3D shapes, cameras, physics fields
- [x] Text-only APIs such as Claude: `bakeoff.graded` on scored answers

Details, and adapters for sklearn-style classifiers such as TabPFN, Chronos forecasts and 3D models, in
[docs/models.md](docs/models.md).

## Install

```bash
git clone https://github.com/leochlon/bitsy && cd bitsy
pip install -e .              # numpy only: hosted models and any model you wrap
pip install -e '.[local]'     # adds torch and transformers, for Hugging Face models on your machine
pip install -e '.[series]'    # adds Chronos, for time series
```

Python 3.9+. On Linux, `pip install torch --index-url https://download.pytorch.org/whl/cpu` first skips the CUDA
download. `python examples/certificates.py` runs on numpy alone in under a second.

## Citation

```bibtex
@misc{chlon2026thoughts,
  title  = {When Thoughts Become Memory: Unifying In-Context Learning and Fine-Tuning},
  author = {Chlon, Leon and Huang, Julie},
  year   = {2026},
  note   = {arXiv preprint}
}
```

[MIT licence](LICENSE). `examples/data/airfrans-pressure64.npz` holds 1,000 AirfRANS simulations (Bonnet et al.,
NeurIPS 2022), each reduced to 18 design inputs and 64 surface pressure coefficients, under the Open Database Licence
1.0.
