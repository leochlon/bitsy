# Certificates

`bakeoff` compares the memories you built. On data you design, where the true law of every answer is known, two
certificates also cover the memories you did not build:

```python
from bitsy import gate, bakeoff

gate.capacity(prompt, prior, C=8)    # the prompt beats every memory that stores at most 8 bits about its context
gate.window(memory, best)            # this memory beats every memory that reads only the window
bakeoff.paired(prompt, adapter)      # the measured winner of one pair you did build, on the same episodes
```

Each argument holds one code length per episode. Each episode draws a fresh task, its context and a block of queries
from the design.

- `prior` is the code length of the answers under their true law before any context is read. Its mean is their
  entropy, H(Q).
- `best` is the code length under the true law given only what the window holds. Its mean is H(Q | window).

## What they certify

- **Capacity.** A memory that stores at most C bits about the context costs at least H(Q) - C bits on average,
  whatever reads it: a fine-tune, an adapter, a compiled or compressed prompt of that size. A prompt that costs less
  beats all of them.
- **Window.** A memory that reads only the last W items or tokens costs at least H(Q | window) bits on average,
  whatever it does with them. A memory that costs less, such as a fine-tune on everything, beats all of them.

A certificate is issued when the 95% interval of its paired margin, prior - prompt - C or best - memory, lies above
zero. `bakeoff.paired` then checks it against each memory it covers that you did build.

## Results

`examples/certificates.py` replays three pretrained models, 64 episodes each. It needs only numpy and runs in under
a second:

```
model      design                                                window   capacity   checks
TabPFN v2  logistic tasks from a pool of 4,096; 8 to 1,024 rows    8/30     17/48      50 of 50 agree
ConvONet   a chair with 6 optional bumps; 12 point patches         2/5      3/6        14 of 14 agree
MVSNet     a card at one of 16 depths; 4 source views              2/3      4/4        22 of 22 agree
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
worker, which is not included here. `examples/data/certificates-64.npz` holds the per-episode code lengths.
`examples/chair.py` runs the ConvONet design live, through a numpy stand-in or the pretrained network, and deletes each
patch with `diagnose.items`; seed 0 draws the recorded episodes. It takes about a minute with the stand-in.

## What these results cover

- The data is designed so that the true laws are exact. On real data H(Q) is unknown.
- A capacity certificate covers only memories smaller than H(Q), the bits in the answers: here about 32, 6 and 4 bits.
  A fine-tune's weights store far more.
- The windows are imposed by the design: TabPFN is given only the last rows, the networks only the first patches or
  views.
- The checks cover the bounded memories that were built: logistic regressions on a grid of 2 or 4 values per weight,
  the best of 2^k fixed classifiers, and states storing k bumps or k bits of depth, read by the true law and by the
  network.
- The intervals are pointwise normal ones, and the same episodes give both the margins and the measured winners.
  Raise `z` when testing many settings.
