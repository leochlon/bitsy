# How it works

Everything bitsy reports is a code length: the number of bits the model spends on what actually happened, -log2 of
the probability it gave it. Saving one bit makes the right answer twice as likely; saving four makes it 16 times as
likely.

The idea is to delete context and ask again. Deleting part of the context and re-reading the query is held-out
validation done inside one forward pass. The bits a model saves on held-out answers when it reads a context measure
what that context teaches it. Deleting one piece of evidence and asking again measures what that piece is worth to the
model, with no labels needed.

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

The derivations, proofs and experiments are in the paper (see [Citation](../README.md#citation)).

## Files

| File | Contents |
|---|---|
| `core.py` | bits, budgets and the betting test |
| `laws.py` | Bernoulli, Categorical, Gaussian; error spreads for point predictors |
| `models.py` | Local, Hosted, Azure, Fn |
| `gate.py` | fit, curves, size, capacity, window |
| `bakeoff.py` | score, labels, graded, race, pool, paired |
| `diagnose.py` | value, unused, evidence, spans, items |
| `guard.py` | check, claims, strict, strict_many |
| `baselines.py` | Kernel, an in-context field learner |
| `adapters.py` | Table, Chronos, field, occupancy, ConvONet, depth |
| `examples/` | the example scripts, the AirfRANS data and the certificate runs |

## All limits

- Free-text code lengths on chat endpoints need prompt logprobs (echo). OpenAI and Azure chat don't return them; a
  vLLM server does.
- APIs with no logprobs at all, such as Claude's, only run `bakeoff.graded`.
- Hosted label ranges come from the top 20 tokens, so close calls can stay ambiguous. `strict_many(group>1)`
  conditions each answer on the ones above it; `group=1` keeps one claim per call.
- Small differences can stay undecided after thousands of queries. The 5% holds per race, not across many races.
- `guard.check` on a local instruction model needs transformers < 5. `guard.strict` works with both.
- The guard's 0.95 threshold is a default, not a calibration for your data.
- `diagnose.items` needs code lengths or predictions. On chat endpoints without echo, check claims against their
  evidence with `diagnose.spans` and `guard.strict`. Its verdicts need at least `tol` bits per query (0.5 by
  default) and, over several queries, an interval that excludes zero; an item that matters for one question in
  ten can fall below that on average, so `per_query` shows which question each item serves.
- Certificates need the true law of every answer, so they run on designed data only. Their intervals are pointwise:
  raise `z` when testing many settings. `bakeoff.paired` is a fixed-sample test; `race` is the one you may stop early.
- `ConvONet.load` has not been run against the released checkpoint here. `adapters.depth` was checked against the card
  design's own reading on synthetic volumes (within 3e-9 bits per pixel), not on MVSNet itself.
