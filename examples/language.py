"""Language: prompt vs fine-tune on GPT-2, with all four tools.

A 512-token passage of Shakespeare is the knowledge. 64 recall queries give 8 tokens of it and score the next 24. The
prompt reads the passage; the fine-tune is trained on it and reads nothing.

    pip install -e '.[local]'
    python examples/language.py                       # GPT-2 from Hugging Face; any causal LM with --model
"""
import argparse, hashlib, itertools, os, sys, urllib.request
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from bitsy import Local, gate, bakeoff, diagnose, guard

URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
SHA = '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--model', default='gpt2', help='Hugging Face model ID or local folder')
parser.add_argument('--text', help='a text file (default: tiny Shakespeare, downloaded once)')
args = parser.parse_args()

path = args.text
if path is None:
    path = os.path.join(os.path.expanduser('~'), '.cache', 'bitsy', 'tinyshakespeare.txt')
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    if hashlib.sha256(open(path, 'rb').read()).hexdigest() != SHA:
        raise ValueError(f'{path}: checksum mismatch')

m = Local(args.model)
ids = np.array(m.enc(open(path).read()))
doc, other = ids[100000:100512], ids[200000:200512]                  # the passage, and a matched one of the same form
qs = [(doc[s:s + 8], doc[s + 8:s + 32]) for s in np.random.default_rng(0).integers(0, 480, 64)]
say = lambda k, v: print(f'{k:16s}{v}', flush=True)

# gate: does it fit, which learns faster, and how small can a fine-tuned state be?
say('gate.fit', gate.fit(m, doc, 32)['read'])
icl, tun = gate.curves(m, list(ids[:64 * 5000].reshape(5000, 64)[np.random.default_rng(3).choice(5000, 32, False)]))
lo, hi, a, b, w = gate.bins(icl / 64, tun / 64)[-1]
say('gate.curves', f'{lo}-{hi} units seen: prompt {a:.2f}, fine-tune {b:.2f} bits/token -> {w} leads')
words = [t for t in range(1000, 40000) if (lambda s: s.startswith(' ') and s[1:].isalpha() and s[1:].islower()
                                                     and len(s) >= 5)(m.tok.decode([t]))]
sel = np.random.default_rng(7).choice(words, size=20, replace=False)  # 4 facts, each one of 8 values: 12 bits
keys, cands, comma = sel[:12].reshape(4, 3), sel[12:], m.enc(',')[0]
tables = np.array(list(itertools.product(range(8), repeat=4)))[np.random.default_rng(1).choice(4096, 512, replace=False)]
variants = [[int(x) for s in range(4) for x in list(keys[s]) + [cands[t[s]], comma]] for t in tables]
r = gate.size(m, variants, [list(k) for k in keys], [[int(c)] for c in cands], bits=2)
say('gate.size', f'the prompt carries >= {r["I"]:.2f} bits of a 12-bit table; a 2-bit state misses it by >= {r["miss"]:.2f}')

# bakeoff: prompt vs fine-tune on the recall queries
a = bakeoff.score(m, doc, qs)
for e in [1, 16]:
    b = bakeoff.score(m.tune([doc], epochs=e), [], qs)
    r = bakeoff.pool(a, b, streams=500)
    say(f'bakeoff {e:2d} ep', f'prompt {a["cost"].mean() / 24:.2f} vs fine-tune {b["cost"].mean() / 24:.2f} bits/token: '
        f'{"prompt" if r["truth"] == "A" else "fine-tune"} called in {max(r["A"], r["B"])}/500 streams after a median '
        f'{r["median_queries"]:.0f} queries, wrong {r["wrong"]}')

# diagnose: content vs format, and what the model leaves unused
v = diagnose.value(m, doc, other, qs)
say('diagnose.value', f'content {v["content"][0] / 24:.2f} bits/token, format {v["format"][0] / 24:.2f} bits/token')
u = diagnose.unused(a, b, gate.fit(m, doc, 32)['fits'])
say('diagnose.unused', f'{u["what"]} >= {u["per_token"]:.2f} bits/token: what the model fails to use from its prompt')

# guard: a supported and a contradicted claim
for claim, ev in [('The tower is in Paris.', 'The Eiffel Tower stands in Paris, France.'),
                  ('The tower is in Rome.', 'The Eiffel Tower stands in Paris, France.')]:
    g = guard.check(m, claim, ev)
    say('guard.check', f'{g["verdict"]:5s} p={g["p"]:.2f} q={g["q"]:.2f} gap={g["gap"]:+.2f} bits  "{claim}"')
