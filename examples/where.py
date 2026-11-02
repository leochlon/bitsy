"""Where a prompt loses bits: the localisation ladder on GPT-2 (or any causal LM).

A document of 8 passages is the context; 24 recall queries give 8 tokens of a passage and score the next 12. When a
fine-tune beats the prompt, the context fits the window, so the gap is a floor on what the model fails to use from it.
The ladder then asks where that loss lives: the output prior, position, depth, the heads that route the passages, or
the block where a reading fix can live. Readouts and readers are trained on 12 other documents, never on this one.

    pip install -e '.[local]'
    python examples/where.py                          # GPT-2 from Hugging Face; any causal LM with --model
"""
import argparse, hashlib, os, sys, urllib.request
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from bitsy import Local, bakeoff, diagnose

URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
SHA = '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--model', default='gpt2', help='Hugging Face model ID or local folder')
parser.add_argument('--text', help='a text file (default: tiny Shakespeare, downloaded once)')
parser.add_argument('--device', default='cpu')
args = parser.parse_args()

path = args.text
if path is None:
    path = os.path.join(os.path.expanduser('~'), '.cache', 'bitsy', 'tinyshakespeare.txt')
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    if hashlib.sha256(open(path, 'rb').read()).hexdigest() != SHA:
        raise ValueError(f'{path}: checksum mismatch')

m = Local(args.model, device=args.device)
ids = m.enc(open(path).read())
say = lambda k, v: print(f'{k:12s}{v}', flush=True)


def document(start, n=8, size=48):
    """n passages of `size` tokens from one stretch of the text, and 3 recall queries per passage."""
    passages = [ids[start + i * size:start + (i + 1) * size] for i in range(n)]
    return passages, [(p[o:o + 8], p[o + 8:o + 20]) for p in passages for o in (4, 16, 28)]


passages, qs = document(100000)
matched, _ = document(150000)                                        # same form, irrelevant content
jobs = [document(200000 + 5000 * j) for j in range(12)]              # other documents, for readouts and readers
L = 12

# the gap: does a fine-tune beat the prompt, and by at least how much?
a = bakeoff.score(m, passages, qs, L=L)
tuned = m.tune(passages, epochs=8)
b = bakeoff.score(tuned, [], qs, L=L)
u = diagnose.unused(a, b, fits=True)
say('unused', f'prompt {a["cost"].mean():.2f} vs fine-tune {b["cost"].mean():.2f} bits per query: {u["read"]}'
    + (f' ({u["floor"]:.2f} bits)' if u['floor'] > 0 else ''))

# 1. prior or content
f = bakeoff.score(m.tune(matched, epochs=8), [], qs, L=L)
say('prior', diagnose.prior(a, b, f)['read'])

# 2. position
say('order', diagnose.order(m, passages, qs, k=8, tuned=b)['read'])

# 3. depth: readouts trained on the other documents, scored here
d = diagnose.depth(m, passages, qs, train=jobs, rank=64)
say('depth', d['read'])
say('', 'bits per query after each layer: ' + ' '.join(f'{v:.1f}' for v in d['mean']))

# 4. routing: which heads carry the first two passages to their answers, and does more context dilute them?
h = diagnose.heads(m, passages, qs[:6], units=[0, 1])
say('heads', h['read'])
e = Local(args.model, device=args.device, attn='eager')
top = [t[1:3] for t in h['top'][:2]]
say('dilution', diagnose.dilution(e, [passages[:2], passages[:4], passages], qs[:3], [0, 0, 0], heads=top)['read'])

# 5. where a reading fix can live: one block at a time, at matched parameter counts
budget = min(m.count(m.block(p, l), 8) for p in ('attn', 'mlp') for l in ('early', 'late'))
for part in ('attn', 'mlp'):
    for layers in ('early', 'late'):
        sel = m.block(part, layers)
        r = m.rank_for(sel, budget)
        reader = m.tune_reading(jobs, epochs=1, lr=1e-4, only=sel, rank=r)
        v = diagnose.unused(a, bakeoff.score(reader, passages, qs, L=L), fits=True)
        say(f'{part} {layers}', f'rank {r}, {m.count(sel, r):,} parameters: advantage {v["mean"]:+.2f} bits per query, '
            f'floor {v["floor"]:.2f}')
