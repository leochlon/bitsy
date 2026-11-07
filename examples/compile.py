"""Compile: a passage into GPT-2's weights without gradients, scored against the prompt it replaces.

A 512-token passage of Shakespeare is the context. Recall queries give 8 tokens of it and score the next 24: 64 drive
the solve, and the other 416, cued at positions the solve did not see, are held out. compile writes the MLP output
projections of the last third of the layers so that the model, reading nothing, answers as it does with the passage
in its prompt, while 128 other passages of the play, read with no context, stay where the model puts them. 512 short
passages from elsewhere in the play measure the damage.

    pip install -e '.[local]'
    python examples/compile.py                        # GPT-2 from Hugging Face; any causal LM with --model
"""
import argparse, hashlib, os, sys, urllib.request
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from bitsy import Local, bakeoff

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
doc = ids[100000:100512]                                             # the context
starts = np.random.default_rng(0).permutation(480)
train, held = ([(doc[s:s + 8], doc[s + 8:s + 32]) for s in part] for part in (starts[:64], starts[64:]))
keep = list(ids[200000:216384].reshape(128, 128))                    # other text, to stay where the model puts it
other = list(ids[250000:266384].reshape(512, 32))                    # held-out text of the same kind: the damage
say = lambda k, v: print(f'{k:16s}{v}', flush=True)
verdict = lambda r: f'{dict(A="yes", B="no").get(r["call"], "not sure yet")} after {r["queries"]} queries'

# the prompt and the original model, reading nothing, on the held-out recall queries and on other text
p, o, e = bakeoff.score(m, doc, held, L=24), bakeoff.score(m, [], held, L=24), bakeoff.score(m, [], other, L=32)
say('prompt', f'recall {p["cost"].mean() / 24:.2f} bits/token; the original model, reading nothing, '
    f'{o["cost"].mean() / 24:.2f}; other text {e["cost"].mean() / 32:.2f}')

c = m.compile([(doc, train)], keep, only=m.block('mlp', 'late'))
rel = np.array(list(c.written['relative'].values()))
say('compile', f'{len(rel)} projections written, changes {rel.min():.2f}-{rel.max():.2f} times their weights\' norm')
b, d = bakeoff.score(c, [], held, L=24), bakeoff.score(c, [], other, L=32)
say('recall', f'{b["cost"].mean() / 24:.2f} bits/token. Better than the original: {verdict(bakeoff.race(b, o))}')
say('vs prompt', f'better: {verdict(bakeoff.race(b, p))}; within 1 bit/token: '
    f'{verdict(bakeoff.race(b, p, margin=24 * 1.0))}')
say('other text', f'{d["cost"].mean() / 32:.2f} bits/token. Within 0.25 bits/token of the original: '
    f'{verdict(bakeoff.race(d, e, margin=32 * 0.25))}')
