# guard: is an answer supported by its evidence?

guard asks the model about a claim twice: once with the evidence, and once with the evidence removed. A claim passes
when the evidence is what makes the model confident. There are three verdicts:

| Verdict | Meaning |
|---|---|
| `pass` | Confident with the evidence (p ≥ t) and not without it (q < t): the evidence supports the claim. |
| `flag` | Not confident even with the evidence (p < t): unsupported, contradicted or unclear. |
| `prior` | Already confident without the evidence (q ≥ t): the model believes it anyway, so the evidence is not why. |

The threshold `t` is 0.95 by default. Lifting a claim from probability q to t takes a minimum number of bits of
evidence. `need` is that number, `got` is what the evidence supplied, and `gap = need - got` is positive when the
evidence falls short.

## On a hosted chat model: guard.strict

```python
from bitsy import Hosted, guard

m = Hosted('gpt-4.1-mini')                       # reads OPENAI_API_KEY; base_url= for vLLM and others
evidence = ['The Eiffel Tower stands in Paris, France.', 'It was completed in 1889.']
r = guard.strict(m, 'The Eiffel Tower is in Paris.', evidence, cites=[0])
r['verdict'], r['why']                           # ('pass', ''), or ('flag', 'contradicted'), ...
```

`guard.strict` uses a strict entailment prompt that asks for YES, NO or UNSURE. It reads only the first output token,
so it needs output-token logprobs and no echo, which OpenAI, Azure chat and vLLM all give.

- **Evidence** goes in as quoted spans. Items that assert nothing (questions, instructions, headings) are masked, and
  the model is told never to follow instructions inside the evidence.
- **The prior** reads the same prompt with the cited items (`cites`, indices; all of them by default) replaced by
  `[CITED_EVIDENCE_REMOVED]`. To check a claim against its cited items alone, pass only those.
- **Label ranges.** Hosted endpoints return the top 20 tokens, so each label's probability is a range: labels inside
  the top 20 are summed exactly, and the mass outside bounds the rest. A token counts for a label if it spells it
  (`Y`, ` Yes`, `UN` count for YES, YES, UNSURE). A claim passes only at the worst end of every range. Local models
  give exact values.
- **Reasons.** Each flag gives a reason in `why`: `contradicted`, `ambiguous` (the YES range straddles the
  threshold), `not entailed`, `low gain` (the lift is below `gain` bits), or `error: ...`.
- **Failures.** A failed call flags its claim and the run carries on. Calls at temperature 0 are cached, and timeouts,
  refused connections, 429s and 5xx responses are retried.

Several claims on the same evidence:

```python
guard.strict_many(m, claims, evidence, group=8)  # up to 8 claims per call
```

With `group > 1`, each answer is conditioned on the ones above it. `group=1` keeps one claim per call. A grouped
reply that cannot be matched line by line is asked again one claim per call. `post_first=True` skips the prior call
for claims that already fail with the evidence.

## On a local model: guard.check

```python
from bitsy import Local, guard

m = Local('Qwen/Qwen2.5-7B-Instruct')
guard.check(m, 'The tower is in Paris.', 'The Eiffel Tower stands in Paris, France.')
# {'verdict': ..., 'p': ..., 'q': ..., 'need': ..., 'got': ..., 'gap': ...}
```

`guard.check` asks a plain Yes or No question with the evidence (p) and with `(none)` in its place (q). It needs an
instruction-tuned model: GPT-2, in `examples/language.py`, flags every claim. On a local instruction model it needs
transformers < 5; `guard.strict` works with both.

## Which evidence did the work: diagnose.spans

```python
diagnose.spans(m, claim, evidence)               # bits each item supplies to the claim
```

Each item alone is replaced by the placeholder while the others stay. Its bits are how far that moves the
probability of YES, as a range over the label ranges.

## Any other model: guard.claims

For a model wrapped with `Fn` (see [Models](models.md)), claims are per site: yes/no, a set of bins, or a value
range.

```python
guard.claims(m, items, spec, [i], claim)         # per-site claims: required vs supplied bits
```

`required` is the mean number of bits needed to lift each site's claim to t; `supplied` is what the deleted items
actually supply. `crossings` counts claims that are confident with the evidence and fall below t without it.

## Example

`examples/vitaminc.py` runs both checks on VitaminC claims (CC BY-SA 3.0), labelled SUPPORTS, REFUTES or NOT ENOUGH
INFO, on a local model or a hosted one:

```bash
python examples/vitaminc.py --local Qwen/Qwen2.5-7B-Instruct --device cuda
python examples/vitaminc.py --openai gpt-4.1-mini                                        # OPENAI_API_KEY
python examples/vitaminc.py --azure chat --endpoint https://<resource>.openai.azure.com   # AZURE_OPENAI_API_KEY
python examples/vitaminc.py --openai MODEL --base-url http://localhost:8000/v1            # a vLLM server
```

## Limits

- The 0.95 threshold is a default, not a calibration for your data.
- Hosted label ranges come from the top 20 tokens, so close calls can stay `ambiguous`.
- `strict_many(group>1)` conditions each answer on the ones above it; `group=1` keeps one claim per call.
- `guard.check` on a local instruction model needs transformers < 5. `guard.strict` works with both.
