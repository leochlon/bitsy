"""Install: behaviour written into GPT-2's MLP output projections in closed form, from a prompt and from a fine-tune.

1. A prompt. A 512-token passage of Shakespeare is the context. Recall queries give 8 tokens of it and score the next
   24: 64 drive the solve and 416 are held out. compile writes the MLP output projections of the last third of the
   layers so that the model, reading nothing, answers as it does with the passage in its prompt, while 128 other
   passages, read with no context, stay where the model puts them. 512 short passages from elsewhere measure the damage.
2. Right answers. 512 passages of the play give an 8-token cue and a 24-token right answer; what the model gives now
   is its own greedy continuation. A fine-tune on 256 of them (one Adam step per passage) changes every weight. Where
   that change lives: each block's MLP share of it, alone in the original, is scored on 128 more right answers and on
   the model's own answers to them. The output projections of the third of the MLPs whose shares help most are then
   solved, shallowest first, so that each MLP makes its own share of the fine-tune's change on the 256 training
   passages and none on 128 passages the model writes itself; the fine-tune's attention and other layers are left
   out. The last 128 right answers, and 384 more passages the model writes, judge the result.

    pip install -e '.[local]'
    python examples/install.py                        # GPT-2 from Hugging Face; any causal LM with --model
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
say = lambda k, v: print(f'{k:16s}{v}', flush=True)
verdict = lambda r: f'{dict(A="yes", B="no").get(r["call"], "not sure yet")} after {r["queries"]} queries'
mean = lambda s, L: f'{s["cost"].mean() / L:.2f}'

# 1. a prompt
doc = ids[100000:100512]                                             # the context
starts = np.random.default_rng(0).permutation(480)
train, held = ([(doc[s:s + 8], doc[s + 8:s + 32]) for s in part] for part in (starts[:64], starts[64:]))
keep = list(ids[200000:216384].reshape(128, 128))                    # other text, to stay where the model puts it
other = list(ids[250000:266384].reshape(512, 32))                    # held-out text of the same kind: the damage

p, o, e = bakeoff.score(m, doc, held, L=24), bakeoff.score(m, [], held, L=24), bakeoff.score(m, [], other, L=32)
say('prompt', f'recall {mean(p, 24)} bits/token; the original model, reading nothing, {mean(o, 24)}; '
    f'other text {mean(e, 32)}')
c = m.compile([(doc, train)], keep, only=m.block('mlp', 'late'))
rel = np.array(list(c.written['relative'].values()))
say('compile', f'{len(rel)} projections written, changes {rel.min():.2f}-{rel.max():.2f} times their weights\' norm')
b, d = bakeoff.score(c, [], held, L=24), bakeoff.score(c, [], other, L=32)
say('recall', f'{mean(b, 24)} bits/token. Better than the original: {verdict(bakeoff.race(b, o))}')
say('vs prompt', f'better: {verdict(bakeoff.race(b, p))}; within 1 bit/token: '
    f'{verdict(bakeoff.race(b, p, margin=24 * 1.0))}')
say('other text', f'{mean(d, 32)} bits/token. Within 0.25 bits/token of the original: '
    f'{verdict(bakeoff.race(d, e, margin=32 * 0.25))}')

# 2. right answers
T = m.T
cues = ids[266384:282768].reshape(512, 32)                           # 8-token cue, 24-token right answer
fit, pick, test = cues[:256], cues[256:384], cues[384:]              # fine-tune and solve; choose the MLPs; judge
right = lambda part: [(q[:8], q[8:]) for q in part]
gen = lambda rows, n, **kw: m.lm.generate(T.tensor(rows, device=m.device), attention_mask=T.ones(
    len(rows), len(rows[0]), dtype=T.long, device=m.device), max_new_tokens=n, min_new_tokens=n, pad_token_id=m.sep,
    **kw)[:, -n:].tolist()                                           # the mask: begin and pad can be the same token
own = lambda part: [(q[:8], a) for q, a in zip(part, gen([[m.bos] + list(q[:8]) for q in part], 24, do_sample=False))]
T.manual_seed(0)
written = gen([[m.bos]] * 512, 64, do_sample=True, top_k=50)         # text the model writes: 128 kept, 384 judge
tuned = m.copy()
for i in np.random.default_rng(0).permutation(256):
    tuned.step(list(fit[i]), lr=1e-4)


def share(tuned, sel):
    """The original with the fine-tune's change on the parameters sel names."""
    x, src = m.copy(), dict(tuned.lm.named_parameters())
    with T.no_grad():
        for n, w in x.lm.named_parameters():
            if sel(n):
                w.copy_(src[n])
    return x


def copy_change(m, tuned, sel, rows, keep, ridge=1e-4):
    """A copy of m whose selected MLP output projections, shallowest first, are solved so that each MLP makes its own
    share of the fine-tune's change on rows and none on keep (token-id rows from the begin token; the fit uses
    positions 1 onward). Later MLPs see the earlier ones' changes."""
    # compile() aims at a context; no public method reads an MLP's input with its projection's input, hence _read
    x = m.copy()
    mods, tmods = dict(x.lm.named_modules()), dict(tuned.lm.named_modules())
    with T.no_grad():
        for _, _, name, proj in x._mlp_outputs(sel):
            mlp = name.rsplit('.', 1)[0]
            G, B = 0, 0
            for data in (rows, keep):
                for u, k in x._read(data, [1] * len(data), [(mods[mlp], 'in'), (proj, 'in')], mods[mlp]):
                    k = k.double()
                    G = G + k.T @ k
                    if data is rows:
                        B = B + (tmods[mlp](u) - mods[mlp](u)).double().T @ k
            G.diagonal().add_(ridge * float(G.diagonal().mean()))
            dW = T.linalg.solve(G, B.T)                              # [in, out]
            proj.weight += (dW if type(proj).__name__ == 'Conv1D' else dW.T).to(proj.weight.dtype)
    return x


sets = [right(pick), own(pick)]
base = [bakeoff.score(m, [], s, L=24) for s in sets]
change = lambda x: [float((bakeoff.score(x, [], s, L=24)['cost'] - b['cost']).mean() / 24) for s, b in zip(sets, base)]
a, b = change(tuned)
say('fine-tune', f'right answers {a:+.3f} bits/token against the original, its own answers {b:+.3f}')
n_layers = m.lm.config.num_hidden_layers
by_layer = [change(share(tuned, m.block('mlp', l))) for l in range(n_layers)]
say('one MLP', 'right / own answers, by layer: ' + ' '.join(f'{l}:{r:+.3f}/{o:+.3f}' for l, (r, o) in enumerate(by_layer)))
best = sorted(np.argsort([r for r, _ in by_layer])[:max(1, n_layers // 3)].tolist())

x = copy_change(m, tuned, m.block('mlp', best), [[m.bos] + list(q) for q in fit], [[m.bos] + t for t in written[:128]])
score = lambda y, s, L: bakeoff.score(y, [], s, L=L)
t0, w0 = score(m, right(test), 24), score(m, written[128:], 64)
gap = lambda y: f'{(score(y, right(test), 24)["cost"] - t0["cost"]).mean() / 24:+.3f}'
t1, w1 = score(x, right(test), 24), score(x, written[128:], 64)
say('solved', f'MLPs {best}: new right answers {(t1["cost"] - t0["cost"]).mean() / 24:+.3f} bits/token (their shares '
    f'{gap(share(tuned, m.block("mlp", best)))}, the whole fine-tune {gap(tuned)}). Better than the original: '
    f'{verdict(bakeoff.race(t1, t0))}')
say('text it writes', f'solved {(w1["cost"] - w0["cost"]).mean() / 64:+.3f} bits/token, fine-tune '
    f'{(score(tuned, written[128:], 64)["cost"] - w0["cost"]).mean() / 64:+.3f}. Solved within 0.25 of the '
    f'original: {verdict(bakeoff.race(w1, w0, margin=64 * 0.25))}')
