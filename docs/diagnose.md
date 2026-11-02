# diagnose: what a prompt depends on

`diagnose.items` tells you, for any prompt, which pieces of context do the work, which are dead weight and which
mislead the model. Each piece gets a number, the bits it is worth on the actual question, and a verdict that says what
to do next.

## Verdicts

| Verdict | What it means | What to do |
|---|---|---|
| `evidence` | Helps through what it says: real evidence the answer depends on. | Keep it, and favour more like it. |
| `format` | Helps only through how it looks: format or label space, not facts. | A short template would do. |
| `misleading` | Changes the answer but makes it worse. The model misreads information it has rather than missing it. | A model problem, not a data problem, and the case where fine-tuning can win. |
| `dead weight` | Adds nothing beyond the rest: redundant or irrelevant. | Cut it; the prompt gets shorter and cheaper. |
| `redundant` | A group worth something together and nothing one at a time: its items repeat each other. | One can go, but not most of them. |

Without controls, a helpful item is reported as `helps`, because content and form cannot be told apart. With controls
whose queries cannot separate the two, it is `unclear`: add queries. Without outcomes (`outcomes=False`), only
influence is measured and an item either `moves` the prediction or is `inert`.

## Content and form

The quickstart's prompt, with a control for each passage. A control has the same form as its passage and irrelevant
content:

```python
from bitsy import Local, diagnose

m = Local('gpt2', sep='\n')
passages = ['The code for Bob is 17.', 'The code for Alice is 52.', 'The code for Bob is not 71.',
            'Every code has two digits.', 'Fish swim in the sea.']
controls = ['The code for Kim is 45.', 'The code for Lee is 30.', 'The code for Ned is not 94.',
            'Every bird has two wings.', 'Birds fly in the sky.']
r = diagnose.items(m, passages, [('The code for Bob is', ' 17')], controls=controls)
```

`examples/items.py` prints the result as a table:

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
Alice's line costs 1.74 bits and its control costs as much, so any second line of the same shape makes 17 less likely.
The negation costs 1.04 bits. GPT-2 is misreading information it has, not missing information. The sentence about fish
does nothing.

## What it measures

For each item:

- **Value**: how many bits the true answer costs without the item, minus with it. Positive helps, negative hurts. It
  needs an outcome: a label, an answer seen later, or, for a few-shot prompt, the other demonstrations. Pass
  `queries=None` and each demonstration is scored as a held-out query by the others, so no outside labels are needed.
- **Influence**: how far the item moves the prediction, in bits, with no labels at all. It shows how much an item
  matters, not whether it helps, so it can audit a retrieval system or a claim's citations before any answer is
  checked. For a language model it is the change in the first answer token's distribution, a lower bound on the whole
  answer's.
- **Content and form**: with `controls`, one per item, of the same form with irrelevant content (another name, a
  shuffled label, an off-topic passage of the same length). Content is the item against its control; form is the
  control against nothing. They add up to the value.
- **Misuse**: a negative value. If the whole context fits the window, nothing was lost to truncation, so the model is
  misreading the item rather than missing it. If it does not fit, the reason says so.

Every item is deleted and the context read again, never masked: masking attention leaves the item's information in
the cache entries that follow it.

The result is a dict. `r['items']` holds one row per item with `value`, `influence`, `content` and `form` (each a
`(mean, lo, hi)` interval over queries, or `None`), `per_query`, `verdict` and `why`, the reason in words.
`r['total']` is what the whole context is worth, `r['reads']` counts the reads and `r['tokens']` the context tokens
re-read.

## Items that repeat each other

Leave-one-out measures what an item adds beyond the rest, so items that repeat each other each look worthless. Pass
`groups` to delete them together. In `examples/items.py`, each of eight few-shot demonstrations is worth at most 0.18
bits alone, yet deleting six of them costs 4.43 bits: they repeat each other, so any one can go, but not most of them.

```python
demos = [(f'{w} ->', f' {w.upper()}') for w in ['cat', 'dog', 'tree', 'house', 'car', 'book', 'bird', 'lamp']]
r = diagnose.items(m, demos, groups=[list(range(6))])       # no queries: each demonstration is scored by the others
```

## Cost

One read of the context per item, two with controls, one per group, plus the full and empty contexts.

- `Local` keeps the cache of the prefix each read shares with the read before it, and items are deleted from last to
  first, so deleting item i re-reads only what comes after it. `tokens` reports what was re-read.
- `screen=k` deletes k contiguous blocks first and tests items one at a time only inside blocks that matter. A block
  that adds nothing can be cut as a whole. Effects that cancel inside a block can be missed.
- Controls are optional. Without them, `value` and `influence` still separate helpful, harmful and dead items.
- Score fewer queries to screen, then more on the items that matter.
- On hosted models every read is a new call. Reading from the last item to the first lets provider-side prefix
  caching reuse the shared beginning of the prompt.

## The other diagnose functions

```python
diagnose.value(m, docs, matched, queries)        # bits from the docs' content vs their form
diagnose.unused(a, b, fits=True)                 # floor on what the model fails to use from its prompt
diagnose.spans(m, claim, items)                  # bits each item supplies to a claim (hosted chat: no echo needed)
diagnose.evidence(m, items, spec, [i], [j])      # bits deleted items supply beyond a matched deletion (Fn models)
```

- `value` compares the context with a matched one of the same form and irrelevant content, and with no context at
  all. `format` is what any context of that form gains; `content` is what this one's content adds.
- `unused` takes two `bakeoff.score` results, the prompt and a fine-tune, and gives a floor on how much the model
  fails to use from its prompt. See [Prompt or fine-tune](prompt-vs-finetune.md).
- `spans` works with any model that has `labels()`, including hosted chat endpoints without echo. See
  [guard](guard.md).
- `evidence` is for models wrapped with `Fn`: points, views, support cases. See [Models](models.md).

## Limits

- `diagnose.items` needs code lengths or predictions. On chat endpoints without echo, check claims against their
  evidence with `diagnose.spans` and `guard.strict`.
- A verdict needs at least `tol` bits per query (0.5 by default) and, over several queries, an interval that excludes
  zero. An item that matters for one question in ten can fall below that on average, so `per_query` shows which
  question each item serves.
