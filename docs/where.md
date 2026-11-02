# Where a prompt loses bits (experimental)

When `diagnose.unused` finds a floor, the context fits the window and the model still fails to use part of it. These
tools ask where inside the model that loss lives. They work on `Local` models only.

Every readout of the network is a predictor of the answer from the same context. On held-out queries, the difference
in code length between two readouts is exactly the difference in their decoding losses, in expectation, with no need
for the true distribution. Each tool compares such readouts, cheapest first.

## The ladder

`a` is the prompt's `bakeoff.score` and `b` a fine-tune's, on the same queries (see
[Prompt or fine-tune](prompt-vs-finetune.md)). `jobs` are `(context, queries)` pairs from other documents.

```python
f = bakeoff.score(m.tune(matched, epochs=4), [], queries, L=32)   # same form, irrelevant content
diagnose.prior(a, b, f)                              # is the fine-tune winning on the prior repair or the content?

diagnose.order(m, passages, queries, k=8, tuned=b)   # decoding loss from position alone

diagnose.depth(m, passages, queries, train=jobs)     # bits read after every layer
diagnose.depth(tuned, [], queries)                   # a fine-tune's curve, to compare (logit lens)

r = diagnose.heads(m, passages, queries, units=[0, 3])            # which heads carry each passage to the answer
e = Local('gpt2', attn='eager')                                   # attention weights
diagnose.dilution(e, [passages[:8], passages[:16], passages], queries, [0, 0, 0],
                  heads=[t[1:3] for t in r['top'][:3]])           # do those heads lose the passage as context grows?

for part in ('attn', 'mlp'):                         # where a reading fix can live, one block at a time
    for layers in ('early', 'middle', 'late'):
        sel = m.block(part, layers)
        reader = m.tune_reading(jobs, only=sel, rank=m.rank_for(sel, 2_000_000))
        diagnose.unused(a, bakeoff.score(reader, passages, queries, L=32), fits=True)
```

`examples/where.py` runs the whole ladder on GPT-2.

## What each step reads, and where it points

| Step | What it measures | Where it points |
|---|---|---|
| `prior` | A fine-tune on a matched control can learn only the format and the usual answers. If it reproduces most of the full fine-tune's advantage, the model sees the format in its context and does not adopt it. | The readout (final norm, output head, an output bias), before the network. |
| `order` | The average of the prompt's predictions over k random orders of the passages cannot depend on order. What it saves over the order you gave is decoding loss caused by position alone. | The positional encoding. |
| `depth` | The answer read off the residual stream after every layer, by a readout trained on other documents (a tuned lens; `rank` keeps it small) or by the logit lens, scored on your queries. If the readout after layer k beats the model's own output, the layers after k throw usable bits away, and `lost[k]` is the decoding loss they add. | Those later layers. If no readout comes close to the fine-tune, the bits never fully reach the answer, which is routing. |
| `heads` | For each passage and head, the rise in the answer's code length if that head alone lost the passage (attribution patching: one backward pass per query, one read per passage), next to the measured rise when the passage is deleted outright. The estimates are first order. | The heads that route each passage. |
| `dilution` | The attention chosen heads pay to one passage as passages are added around it. Softmax attention normalises over the whole context, so each added passage competes for that weight. | Long contexts drowning a passage. |
| `tune_reading` | Adapters trained to read, with the context in the prompt, on other documents, restricted to one block at a time. Each one's advantage over the prompt is another floor on the decoding loss. | The block that recovers the most is where a fix can live. |

`block`, `count` and `rank_for` match the parameter counts of the readers, and `tune` takes the same `only` and `rank`
arguments. `dilution` and `Local.attends` need attention weights: load the model with `Local(path, attn='eager')`.

## Limits

- Train every readout and reader on documents separate from the ones you score, or they can store the answers and the
  floors stop being valid.
- `tune_reading` shows where a fix is enough; it does not show where the fault starts.
- `heads` gives first-order estimates. They need not add up to the measured deletion, which also moves the MLPs and is
  not linear; `heads_sum` shows how much they account for.
- Block names are read from parameter names (Llama, Qwen, Mistral, GPT-2 and GPT-NeoX naming); check that `count()` is
  not zero.
