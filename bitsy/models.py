"""Adapters. Language models (Local, Hosted) have: window, V, n(text), bits(context, queries), yes(prompt),
labels(prompt, names); Local also has copy(), step(text, lr), tune(texts, ...), dist(context, prompts, cands), and, for
finding where in the network a prompt loses bits: block(), count(), rank_for(), tune_reading(), hidden(), lens(),
depth_bits(), routes() and attends(). Any other model (Fn) has law(evidence, spec), bits(context, queries) and
dist(context, specs).

Text arguments may be strings or token-id lists. A context is a string, a token-id list, or a list
of units (strings or token-id lists), which are read as separate passages. A query is text, or a pair
(cue, answer) whose answer alone is scored.

labels() reads only the first output token, so it needs output logprobs (OpenAI or Azure chat, vLLM, a local model)
and no echo. A label's probability is the probability that the first output token spells it (label_of)."""
import copy, json, math, os, re, time, urllib.error, urllib.request, urllib.parse
import numpy as np

YES, NO = ('Yes', ' Yes', 'yes', ' yes'), ('No', ' No', 'no', ' no')
LABELS = ('YES', 'NO', 'UNSURE')
RETRY = (408, 409, 425, 429, 500, 502, 503, 504)
PARTS = ('attn', 'mlp', 'norm', 'embed', 'head')
_LAYER = re.compile(r'(?:^|\.)(?:layers|h|blocks|layer)\.(\d+)\.')
_ATTN_OUT = re.compile(r'(?:^|\.)(?:layers|h|blocks|layer)\.(\d+)\.(?:self_attn|attn|attention|self_attention)\.'
                       r'(?:o_proj|c_proj|dense|out_proj)$')


def _part(name):
    """The part of a transformer a parameter belongs to, from its name: attn, mlp, norm, embed, head or other."""
    n = name.lower()
    if re.search(r'(?:^|\.)(?:ln_\w+|\w*norm\w*)\.', n):
        return 'norm'
    if re.search(r'(?:^|\.)(?:lm_head|embed_out)\.', n):
        return 'head'
    if re.search(r'(?:^|\.)(?:\w*attn\w*|\w*attention\w*)\.', n):
        return 'attn'
    if re.search(r'(?:^|\.)(?:mlp|ffn|feed_forward|block_sparse_moe|experts?)\.', n):
        return 'mlp'
    if re.search(r'(?:^|\.)(?:embed_tokens|embed_in|wte|wpe|embeddings?|word_embeddings)\.', n):
        return 'embed'
    return 'other'


def _layer(name):
    m = _LAYER.search(name)
    return int(m.group(1)) if m else None


def _final_norm(lm):
    """The norm a causal LM applies before its output head, found by name."""
    for path in ('model.norm', 'transformer.ln_f', 'gpt_neox.final_layer_norm', 'model.final_layernorm',
                 'model.decoder.final_layer_norm', 'backbone.norm_f', 'transformer.norm_f'):
        obj = lm
        for a in path.split('.'):
            obj = getattr(obj, a, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise ValueError('cannot find the final norm of this model, which reading intermediate layers needs')


def _low_rank(T, shape, r, device, dtype, gen):
    """A rank-r additive update W + B A of one weight matrix, starting at zero (B = 0)."""
    class LowRank(T.nn.Module):
        def __init__(self):
            super().__init__()
            self.A = T.nn.Parameter((T.randn(r, shape[1], generator=gen) / math.sqrt(shape[1])).to(device, dtype))
            self.B = T.nn.Parameter(T.zeros(shape[0], r, device=device, dtype=dtype))

        def forward(self, W):
            return W + self.B @ self.A
    return LowRank()


def _translator(T, d, rank, device, gen):
    """h -> h + W h + b, with W full or of rank r, starting at the identity."""
    class Translator(T.nn.Module):
        def __init__(self):
            super().__init__()
            self.b = T.nn.Parameter(T.zeros(d, device=device))
            if rank is None:
                self.W = T.nn.Parameter(T.zeros(d, d, device=device))
            else:
                self.U = T.nn.Parameter(T.zeros(d, rank, device=device))
                self.V = T.nn.Parameter((T.randn(rank, d, generator=gen) / math.sqrt(d)).to(device))

        def forward(self, h):
            return h + (h @ self.W.T if rank is None else (h @ self.V.T) @ self.U.T) + self.b
    return Translator()


def label_of(token, names=LABELS):
    """The label a first output token spells, or None. Case, surrounding space and trailing punctuation are ignored. A
    token that equals a label, or starts exactly one label, counts for it ('Y', ' Yes', 'UN' -> YES, YES, UNSURE), so
    a label split into several tokens is read from its first."""
    t = str(token).strip().rstrip('.,:;!').upper()
    if not t:
        return None
    exact = [n for n in names if n.upper() == t]
    hits = exact or [n for n in names if n.upper().startswith(t)]
    return hits[0] if len(hits) == 1 else None


def ranges(top, names=LABELS):
    """[lo, hi] for P(the first output token spells each label), from the top k (token, logprob) pairs. A label's
    tokens inside the top k are summed exactly; every token outside it shares the mass the top k leave, so the label
    can hold at most that much more. Exact when the top k hold all the mass. Relative to the probabilities the
    endpoint reports."""
    lo, seen = dict.fromkeys(names, 0.0), 0.0
    for tok, lp in top:
        q = math.exp(lp)
        seen += q
        lab = label_of(tok, names)
        if lab is not None:
            lo[lab] += q
    rest = max(0.0, 1.0 - seen)
    return {n: (min(v, 1.0), min(v + rest, 1.0)) for n, v in lo.items()}


def _trim(cache, keep):
    extra = cache.get_seq_length() - keep
    if extra > 0:
        cache.crop(-extra)  # a negative crop removes tokens in transformers 4 and 5


def _is_units(ctx):
    return ctx is not None and not isinstance(ctx, str) and len(ctx) > 0 and not isinstance(ctx[0], (int, np.integer))


class Local:
    """A Hugging Face causal LM on this machine: full access, so every product runs. Context units are joined by the
    end-of-text token unless sep names another single token, such as a newline. attn picks the attention
    implementation; attends() needs attention weights, which attn='eager' exposes."""

    def __init__(self, path, device='cpu', sep=None, attn=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.T, self.device, self._opt, self._adapters = torch, device, None, []
        self.tok = AutoTokenizer.from_pretrained(path)
        kw = {} if attn is None else dict(attn_implementation=attn)
        self.lm = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, **kw).to(device).eval()
        c = self.lm.config
        self.window = getattr(c, 'n_positions', None) or c.max_position_embeddings
        self.V = c.vocab_size
        e, b = self.tok.eos_token_id, self.tok.bos_token_id
        self.bos, self.sep = (b if b is not None else e), (e if e is not None else b)
        if sep is not None:  # join context units with this one token (e.g. '\n') instead of end-of-text
            ids = self.enc(sep)
            if len(ids) != 1:
                raise ValueError('sep must be a single token')
            self.sep = ids[0]
        self._labels = {}

    def enc(self, x):
        return self.tok.encode(x, add_special_tokens=False) if isinstance(x, str) else [int(i) for i in x]

    def n(self, x):
        return len(self.enc(x))

    def copy(self):
        m = copy.copy(self)
        m.lm, m._opt = copy.deepcopy(self.lm), None
        return m

    def _ctx(self, ctx, room, spans=False):
        """[bos] + context in at most `room` tokens: units keep their last whole units, each closed by a
        separator; text keeps its last tokens. spans=True also returns {unit index: (start, end)} for the whole units
        kept, positions in the returned ids, separator excluded."""
        if _is_units(ctx):
            ids, kept = [], []
            for j in range(len(ctx) - 1, -1, -1):
                e = self.enc(ctx[j]) + [self.sep]
                if 1 + len(ids) + len(e) > room:
                    break
                ids = e + ids
                kept.insert(0, j)
            if not ids:  # even the newest unit is too long: keep its last tokens
                e = self.enc(ctx[-1]) + [self.sep]
                ids = e[max(0, len(e) - room + 1):]
            if not spans:
                return [self.bos] + ids
            where, pos = {}, 1
            for j in kept:
                n = len(self.enc(ctx[j]))
                where[j] = (pos, pos + n)
                pos += n + 1
            return [self.bos] + ids, where
        ids = self.enc(ctx) if ctx is not None and len(ctx) else []
        out = [self.bos] + ids[max(0, len(ids) - room + 1):]
        return (out, {}) if spans else out

    def _split(self, q):
        return (self.enc(q[0]), self.enc(q[1])) if isinstance(q, tuple) else ([], self.enc(q))

    def bits(self, context, queries):
        """Per-token bits of each query's answer given the context, teacher-forced; the context is read once
        and cut to fit the window with the longest query."""
        T, qs = self.T, [self._split(q) for q in queries]
        ids = self._ctx(context, self.window - max(len(c) + len(a) for c, a in qs))
        out = []
        with T.no_grad():
            o = self.lm(T.tensor([ids], device=self.device), use_cache=True)
            first, cache = T.log_softmax(o.logits[0, -1].float(), -1), o.past_key_values
            for cue, ans in qs:
                seq = cue + ans
                lp = T.log_softmax(self.lm(T.tensor([seq[:-1]], device=self.device), past_key_values=cache,
                                           use_cache=True).logits[0].float(), -1) if len(seq) > 1 else None
                _trim(cache, len(ids))
                rows = ([first] if not cue else []) + ([lp[i] for i in range(max(len(cue) - 1, 0), len(seq) - 1)]
                                                       if lp is not None else [])
                out.append(np.array([-float(r[a]) / math.log(2) for r, a in zip(rows, ans)]))
        return out

    def sweep(self, jobs):
        """bits() for many (context, queries) jobs, in order, plus the log-probabilities of each query's first answer
        token. Each context re-reads only what follows the prefix it shares with the context before it, from that
        prefix's cache: delete a late item and only the items after it are read again. Contexts are cut to fit the
        longest query of any job. Yields (bits, firsts, read, tokens) per job; firsts: [queries, vocabulary], natural
        log; read: context tokens actually re-read, out of tokens."""
        T = self.T
        jobs = [(c, [self._split(q) for q in qs]) for c, qs in jobs]
        room = self.window - max(len(c) + len(a) for _, qs in jobs for c, a in qs)
        cache, prev = None, []
        with T.no_grad():
            for context, qs in jobs:
                ids = self._ctx(context, room)
                k = 0
                while k < min(len(ids), len(prev)) - 1 and ids[k] == prev[k]:
                    k += 1                                   # shared prefix; the last token is always re-read
                if cache is None or k == 0:
                    o = self.lm(T.tensor([ids], device=self.device), use_cache=True)
                else:
                    _trim(cache, k)
                    o = self.lm(T.tensor([ids[k:]], device=self.device), past_key_values=cache, use_cache=True)
                cache, first = o.past_key_values, T.log_softmax(o.logits[0, -1].float(), -1)
                bits, firsts = [], []
                for cue, ans in qs:
                    seq = cue + ans
                    lp = T.log_softmax(self.lm(T.tensor([seq[:-1]], device=self.device), past_key_values=cache,
                                               use_cache=True).logits[0].float(), -1) if len(seq) > 1 else None
                    _trim(cache, len(ids))
                    rows = ([first] if not cue else []) + ([lp[i] for i in range(max(len(cue) - 1, 0), len(seq) - 1)]
                                                           if lp is not None else [])
                    bits.append(np.array([-float(r[a]) / math.log(2) for r, a in zip(rows, ans)]))
                    firsts.append(rows[0].cpu().numpy())
                read = len(ids) - k
                prev = ids
                yield bits, np.array(firsts, dtype=np.float32), read, len(ids)

    def step(self, x, lr=3e-5):
        """One Adam step on [bos] + x (online fine-tune). Returns x's per-token bits before the step."""
        T = self.T
        ids = T.tensor([[self.bos] + self.enc(x)], device=self.device)
        if self._opt is None:
            self._opt = T.optim.Adam(self.lm.parameters(), lr=lr)
        for g in self._opt.param_groups:
            g['lr'] = lr
        lp = T.log_softmax(self.lm(ids).logits[0, :-1].float(), -1)
        tok = lp[T.arange(ids.shape[1] - 1), ids[0, 1:]]
        self._opt.zero_grad()
        (-tok.mean()).backward()
        self._opt.step()
        return (-tok.detach() / math.log(2)).cpu().numpy()

    def tune(self, texts, epochs=1, lr=1e-4, chunk=256, batch=2, seed=0, only=None, rank=None):
        """Fine-tune with Adam. Each epoch cuts every text into pieces of at most `chunk` tokens at a random
        offset, covering it once, `batch` pieces per step, each read as [bos] + piece. texts: one text or a list.
        only: which parameters train. None trains all of them; otherwise a selector from block(), a name substring or a
        list of them, or any function of the parameter name. rank: train rank-r updates of the selected weight matrices
        (an adapter; vectors such as biases and norm scales train in full), merged into the weights at the end, so the
        result reads like any other model. count() gives the trainable parameters, rank_for() a rank that matches a
        budget. Returns a new model; this one is unchanged."""
        T, m, rng = self.T, self.copy(), np.random.default_rng(seed)
        opt = T.optim.Adam(m.lm.parameters() if only is None and rank is None else m._trainable(only, rank, seed), lr=lr)
        docs = [self.enc(x) for x in (texts if _is_units(texts) else [texts])]
        for _ in range(epochs):
            pieces = []
            for d in docs:
                o = int(rng.integers(0, chunk))
                cuts = sorted(set(c for c in [0] + list(range(o, len(d), chunk)) + [len(d)] if 0 <= c <= len(d)))
                pieces += [d[a:b] for a, b in zip(cuts[:-1], cuts[1:]) if b - a >= 2]
            rng.shuffle(pieces)
            for i in range(0, len(pieces), batch):
                grp = pieces[i:i + batch]
                ids = T.full((len(grp), chunk + 1), self.bos, device=self.device)
                w = T.zeros((len(grp), chunk + 1), device=self.device)
                for r, p in enumerate(grp):
                    ids[r, 1:1 + len(p)] = T.tensor(p)
                    w[r, :1 + len(p)] = 1
                lp = T.log_softmax(m.lm(ids).logits[:, :-1].float(), -1)
                tok = lp.gather(-1, ids[:, 1:, None])[..., 0]
                opt.zero_grad()
                (-(tok * w[:, 1:]).sum() / w[:, 1:].sum()).backward()
                opt.step()
        m._merge()
        return m

    # ---- where in the network a prompt loses bits

    def block(self, part=None, layers=None):
        """A selector for tune(only=...), tune_reading() and count(): the parameters of one part, or a tuple of parts,
        from PARTS ('attn', 'mlp', 'norm', 'embed', 'head'), in some layers: an index, a list or range of indices, or
        'early', 'middle' or 'late', a third of the depth each. None means every part, or every layer. Selecting layers
        leaves out the embeddings, the final norm and the head, which sit in no layer. Parts are read from parameter
        names (Llama, Qwen, Mistral, GPT-2 and GPT-NeoX naming); check count() is not zero."""
        parts = None if part is None else ((part,) if isinstance(part, str) else tuple(part))
        if parts is not None and any(p not in PARTS for p in parts):
            raise ValueError(f'parts are {PARTS}')
        L = self.lm.config.num_hidden_layers
        if layers is None:
            keep = None
        elif isinstance(layers, str):
            thirds = dict(zip(('early', 'middle', 'late'), np.array_split(np.arange(L), 3)))
            if layers not in thirds:
                raise ValueError("layers must be an index, a list of indices, or 'early', 'middle' or 'late'")
            keep = {int(i) for i in thirds[layers]}
        else:
            keep = {int(layers)} if isinstance(layers, (int, np.integer)) else {int(i) for i in layers}

        def select(name):
            return (parts is None or _part(name) in parts) and (keep is None or _layer(name) in keep)
        return select

    def _chosen(self, only):
        """(name, parameter, module, attribute) for each parameter the selection names. A head tied to the input
        embedding appears only under the embedding's name, so selecting the head selects that shared matrix."""
        sel = (lambda name: True) if only is None else only if callable(only) else (
            lambda name, pats=((only,) if isinstance(only, str) else tuple(only)): any(p in name for p in pats))
        mods = dict(self.lm.named_modules())
        out = []
        for name, p in self.lm.named_parameters():
            mname, attr = name.rsplit('.', 1) if '.' in name else ('', name)
            if sel(name):
                out.append((name, p, mods[mname], attr))
        head, emb = self.lm.get_output_embeddings(), self.lm.get_input_embeddings()
        if head is not None and emb is not None and head.weight is emb.weight and not any(p is emb.weight for _, p, _, _ in out):
            hname = next((n for n, mod in mods.items() if mod is head), None)
            if hname is not None and sel(hname + '.weight'):
                ename = next(n for n, mod in mods.items() if mod is emb)
                out.append((ename + '.weight', emb.weight, emb, 'weight'))
        return out

    def _matrix(self, p, mod):
        """A weight matrix an adapter can update: two-dimensional and not an embedding table."""
        return p.ndim == 2 and not isinstance(mod, self.T.nn.Embedding)

    def _sizes(self, only):
        """(parameters that train in full under an adapter, rows + columns summed over the adapted matrices, total)."""
        vec = rows = total = 0
        for _, p, mod, _ in self._chosen(only):
            total += p.numel()
            if p.ndim == 1:
                vec += p.numel()
            elif self._matrix(p, mod):
                rows += p.shape[0] + p.shape[1]
        return vec, rows, total

    def count(self, only=None, rank=None):
        """Parameters that tune() or tune_reading() train with this selection: the selected parameters in full, or,
        with rank, rank-r updates of the selected weight matrices plus the selected vectors (embedding tables are left
        out under an adapter). Match it across blocks before comparing them, so the comparison is about where the
        parameters sit and not how many there are."""
        vec, rows, total = self._sizes(only)
        return int(total if rank is None else vec + rank * rows)

    def rank_for(self, only, budget):
        """The largest rank whose adapter on this selection trains at most `budget` parameters (at least 1)."""
        vec, rows, _ = self._sizes(only)
        if rows == 0:
            raise ValueError('the selection holds no weight matrix to adapt')
        return max(1, int((budget - vec) // rows))

    def _trainable(self, only, rank=None, seed=0):
        """Freeze everything, then open the selection: in full, or through rank-r updates that _merge() folds back."""
        T = self.T
        for p in self.lm.parameters():
            p.requires_grad_(False)
        params, self._adapters, gen = [], [], T.Generator().manual_seed(seed)
        for _, p, mod, attr in self._chosen(only):
            if rank is None or p.ndim == 1:
                if not p.requires_grad:
                    p.requires_grad_(True)
                    params.append(p)
            elif self._matrix(p, mod):
                from torch.nn.utils import parametrize
                up = _low_rank(T, tuple(p.shape), rank, p.device, p.dtype, gen)
                parametrize.register_parametrization(mod, attr, up)
                getattr(mod.parametrizations, attr).original.requires_grad_(False)
                params += [up.A, up.B]
                self._adapters.append((mod, attr))
        if not params:
            raise ValueError('the selection trains no parameters')
        return params

    def _merge(self):
        """Fold any adapters into their weights and make every parameter trainable again."""
        if self._adapters:
            from torch.nn.utils import parametrize
            for mod, attr in self._adapters:
                parametrize.remove_parametrizations(mod, attr, leave_parametrized=True)
        self._adapters = []
        for p in self.lm.parameters():
            p.requires_grad_(True)

    def tune_reading(self, jobs, epochs=1, lr=1e-4, only=None, rank=None, seed=0):
        """Fine-tune how the model reads. jobs: (context, queries) as in sweep(), from documents other than the ones
        you will test on. Each query is read with its context in the prompt and the loss falls on its answer tokens
        only, so the model learns to use a context and cannot store the answers to your test questions. A tuned reader
        is one more predictor from the same context: its average advantage over the prompt on held-out documents is a
        floor on the prompt's decoding loss, so restricting `only` to one block at a time shows where a fix can live.
        only, rank: as in tune(). One step per job, its queries' answer tokens weighted equally. Returns a new model;
        this one is unchanged."""
        T, m, rng = self.T, self.copy(), np.random.default_rng(seed)
        params = m.lm.parameters() if only is None and rank is None else m._trainable(only, rank, seed)
        opt = T.optim.Adam(params, lr=lr)
        jobs = [(c, [m._split(q) for q in qs]) for c, qs in jobs]
        room = m.window - max(len(c) + len(a) for _, qs in jobs for c, a in qs)
        for _ in range(epochs):
            for j in rng.permutation(len(jobs)):
                context, qs = jobs[j]
                ids, total = m._ctx(context, room), sum(len(a) for _, a in qs)
                opt.zero_grad()
                for cue, ans in qs:
                    seq = ids + cue + ans
                    lp = T.log_softmax(m.lm(T.tensor([seq[:-1]], device=m.device)).logits[0].float(), -1)
                    pos = T.arange(len(ids) + len(cue) - 1, len(seq) - 1, device=m.device)
                    (-lp[pos, T.tensor(ans, device=m.device)].sum() / total).backward()
                opt.step()
        m._merge()
        return m

    def hidden(self, context, queries):
        """The residual stream at every depth, at each position that predicts an answer token, with the context read
        once. Per query: (states [answer tokens, layers + 1, width], answer token ids, the model's own per-token
        bits). State k is the stream after k layers; the last is what the head reads (after the final norm in Hugging
        Face models), so depth_bits() scores that row with the model's own logits."""
        T, qs = self.T, [self._split(q) for q in queries]
        ids = self._ctx(context, self.window - max(len(c) + len(a) for c, a in qs))
        out = []
        with T.no_grad():
            cache = None
            if len(ids) > 1:
                cache = self.lm(T.tensor([ids[:-1]], device=self.device), use_cache=True).past_key_values
            o = self.lm(T.tensor([ids[-1:]], device=self.device), past_key_values=cache, use_cache=True,
                        output_hidden_states=True)
            cache, last = o.past_key_values, T.stack([h[0, -1] for h in o.hidden_states])
            first = T.log_softmax(o.logits[0, -1].float(), -1)
            for cue, ans in qs:
                seq, states, rows = cue + ans, ([last] if not cue else []), ([first] if not cue else [])
                if len(seq) > 1:
                    o2 = self.lm(T.tensor([seq[:-1]], device=self.device), past_key_values=cache, use_cache=True,
                                 output_hidden_states=True)
                    hs = T.stack([h[0] for h in o2.hidden_states], 1)
                    lp = T.log_softmax(o2.logits[0].float(), -1)
                    for i in range(max(len(cue) - 1, 0), len(seq) - 1):
                        states.append(hs[i])
                        rows.append(lp[i])
                _trim(cache, len(ids))
                out.append((T.stack(states).float().cpu().numpy(), np.array(ans),
                            np.array([-float(r[a]) / math.log(2) for r, a in zip(rows, ans)])))
        return out

    def lens(self, jobs, rank=None, epochs=4, lr=1e-3, batch=256, seed=0):
        """A readout for every layer, trained to predict answer tokens from the residual stream after that layer (a
        tuned lens): h -> head(norm(h + W h + b)), W full or of rank r, starting from the logit lens (W = 0, b = 0).
        jobs: (context, queries) from documents other than the ones you will score, so a readout cannot store your
        test answers. Returns one readout per layer, for depth_bits()."""
        T = self.T
        data = [r for c, qs in jobs for r in self.hidden(c, qs)]
        H = T.tensor(np.concatenate([h for h, _, _ in data]))
        y = T.tensor(np.concatenate([a for _, a, _ in data]))
        norm, head = _final_norm(self.lm), self.lm.get_output_embeddings()
        flags = [p.requires_grad for p in self.lm.parameters()]
        for p in self.lm.parameters():
            p.requires_grad_(False)
        out, gen = [], T.Generator().manual_seed(seed)
        try:
            for k in range(H.shape[1] - 1):
                tr = _translator(T, H.shape[2], rank, self.device, gen)
                opt = T.optim.Adam(tr.parameters(), lr=lr)
                for _ in range(epochs):
                    for idx in T.randperm(len(y), generator=gen).split(batch):
                        x, t = H[idx, k].to(self.device), y[idx].to(self.device)
                        loss = T.nn.functional.cross_entropy(head(norm(tr(x))).float(), t)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                out.append(tr.eval())
        finally:
            for p, f in zip(self.lm.parameters(), flags):
                p.requires_grad_(f)
        return out

    def depth_bits(self, context, queries, lens=None):
        """Code length of each query's answer read after every layer: [layers + 1, queries]. Row k reads the stream
        after k layers through its readout from lens() (None: the logit lens, head(norm(h)), which needs no training);
        the last row is the model's own output, exactly."""
        T, rows = self.T, self.hidden(context, queries)
        norm, head = _final_norm(self.lm), self.lm.get_output_embeddings()
        depth = rows[0][0].shape[1]
        if lens is not None and len(lens) != depth - 1:
            raise ValueError('need one readout per layer')
        B = np.zeros((depth, len(rows)))
        with T.no_grad():
            for q, (states, ans, bits) in enumerate(rows):
                S, a = T.tensor(states, device=self.device), T.tensor(ans, device=self.device)
                for k in range(depth - 1):
                    h = S[:, k] if lens is None else lens[k](S[:, k])
                    lp = T.log_softmax(head(norm(h)).float(), -1)
                    B[k, q] = -float(lp[T.arange(len(a), device=self.device), a].sum()) / math.log(2)
                B[depth - 1, q] = float(bits.sum())
        return B

    def _attn_outputs(self):
        """The output projection of every layer's attention, whose input is the heads' outputs side by side."""
        found = sorted((int(m.group(1)), name) for name, _ in self.lm.named_modules() if (m := _ATTN_OUT.search(name)))
        if [i for i, _ in found] != list(range(self.lm.config.num_hidden_layers)):
            raise ValueError('cannot find one attention output projection per layer by name')
        mods = dict(self.lm.named_modules())
        return [mods[name] for _, name in found]

    def routes(self, context, queries, units=None):
        """Bits each attention head carries from each context unit to the answers, by attribution patching. One
        backward pass per query gives every head's gradient at the query positions (the last context token through the
        answer); one read per deleted unit gives each head's output there without the unit. Their product, summed over
        positions, estimates how much the answer's code length would rise if that head alone lost the unit: a
        first-order estimate, exact for small changes. Returns (routes [units, layers, heads], exact [units]), means
        over queries, where exact is the measured rise when the unit is deleted for every head at once. The context must
        fit the window, so deleting a unit removes only that unit."""
        T = self.T
        if not _is_units(context):
            raise ValueError('routes needs a context of units, such as passages')
        context, qs = list(context), [self._split(q) for q in queries]
        keep = list(range(len(context))) if units is None else [int(u) for u in units]
        room = self.window - max(len(c) + len(a) for c, a in qs)
        projs, H = self._attn_outputs(), self.lm.config.num_attention_heads
        seen = [None] * len(projs)

        def grab(i):
            def hook(mod, args):
                if args[0].requires_grad:
                    args[0].retain_grad()
                seen[i] = args[0]
            return hook

        def answer_bits(o, n_ctx, cue, ans):
            lp = T.log_softmax(o.logits[0].float(), -1)
            pos = T.arange(n_ctx + len(cue) - 1, n_ctx + len(cue) + len(ans) - 1, device=self.device)
            return -lp[pos, T.tensor(ans, device=self.device)].sum() / math.log(2)

        handles = [mod.register_forward_pre_hook(grab(i)) for i, mod in enumerate(projs)]
        flags = [p.requires_grad for p in self.lm.parameters()]
        for p in self.lm.parameters():
            p.requires_grad_(False)
        out, exact = np.zeros((len(keep), len(projs), H)), np.zeros(len(keep))
        try:
            ids, emb = self._ctx(context, room), self.lm.get_input_embeddings()
            for cue, ans in qs:
                n = len(cue) + len(ans)
                with T.enable_grad():
                    x = emb(T.tensor([(ids + cue + ans)[:-1]], device=self.device)).detach().requires_grad_(True)
                    b0 = answer_bits(self.lm(inputs_embeds=x), len(ids), cue, ans)
                    b0.backward()
                clean = [s.detach()[0, -n:] for s in seen]
                grad = [s.grad[0, -n:] for s in seen]
                for r, u in enumerate(keep):
                    cut = self._ctx([c for j, c in enumerate(context) if j != u], room)
                    with T.no_grad():
                        b1 = answer_bits(self.lm(T.tensor([(cut + cue + ans)[:-1]], device=self.device)), len(cut), cue, ans)
                        exact[r] += float(b1) - float(b0)
                        for i in range(len(projs)):
                            d = ((seen[i][0, -n:] - clean[i]) * grad[i]).reshape(n, H, -1).sum((0, 2))
                            out[r, i] += d.float().cpu().numpy()
        finally:
            for h in handles:
                h.remove()
            for p, f in zip(self.lm.parameters(), flags):
                p.requires_grad_(f)
        return out / len(qs), exact / len(qs)

    def attends(self, context, query, target):
        """Attention each head pays to one context unit from the query's positions (the last context token through
        the answer): [layers, heads], each entry the attention mass on the unit's tokens averaged over those positions.
        Needs attention weights: Local(path, attn='eager')."""
        T, (cue, ans) = self.T, self._split(query)
        ids, where = self._ctx(context, self.window - len(cue) - len(ans), spans=True)
        if target not in where:
            raise ValueError('the target unit is not in what the window keeps')
        s, e = where[target]
        with T.no_grad():
            o = self.lm(T.tensor([(ids + cue + ans)[:-1]], device=self.device), output_attentions=True)
        if not o.attentions or o.attentions[0] is None:
            raise ValueError("this model's attention hides its weights: load it with Local(path, attn='eager')")
        n = len(cue) + len(ans)
        return np.array([a[0, :, -n:, s:e].sum(-1).mean(-1).float().cpu().numpy() for a in o.attentions])

    def dist(self, context, prompts, cands):
        """Next-token law after context + each prompt, lumped into the candidates' first tokens and 'other'
        -> [len(prompts), len(cands)+1]."""
        T, c = self.T, [self.enc(x)[0] for x in cands]
        if len(set(c)) < len(c):
            raise ValueError('candidates must start with distinct tokens')
        ps = [self.enc(p) for p in prompts]
        ids = self._ctx(context, self.window - max(map(len, ps)))
        rows = []
        with T.no_grad():
            cache = self.lm(T.tensor([ids], device=self.device), use_cache=True).past_key_values
            for p in ps:
                pr = T.softmax(self.lm(T.tensor([p], device=self.device), past_key_values=cache,
                                       use_cache=True).logits[0, -1].double(), -1)
                _trim(cache, len(ids))
                pc = pr[c].cpu().numpy()
                rows.append(np.append(pc, max(1.0 - pc.sum(), 1e-12)))
        return np.array(rows)

    def yes(self, prompt):
        """P(Yes) / (P(Yes) + P(No)) for the next token after the prompt."""
        T = self.T
        ids = (self.tok.apply_chat_template([{'role': 'user', 'content': prompt}],
                                          tokenize=True, add_generation_prompt=True)
               if getattr(self.tok, 'chat_template', None) else [self.bos] + self.enc(prompt))
        with T.no_grad():
            p = T.softmax(self.lm(T.tensor([ids], device=self.device)).logits[0, -1].double(), -1)
        y = sum(float(p[i]) for i in {self.enc(s)[0] for s in YES})
        n = sum(float(p[i]) for i in {self.enc(s)[0] for s in NO})
        return y / (y + n)

    def _chat(self, prompt):
        """The prompt as one user turn plus the generation prompt if the tokenizer has a chat template, else [bos] +
        prompt. return_dict=False keeps a plain id list in transformers 4 and 5."""
        if not getattr(self.tok, 'chat_template', None):
            return [self.bos] + self.enc(prompt)
        ids = self.tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True,
                                           add_generation_prompt=True, return_dict=False)
        return list(ids['input_ids'] if hasattr(ids, 'keys') else ids)

    def _label_ids(self, names, size):
        """Vocabulary ids whose token spells each label (label_of), cached per label set."""
        key = (names, size)
        if key not in self._labels:
            words = self.tok.batch_decode([[i] for i in range(min(size, len(self.tok)))])
            ids = {n: [] for n in names}
            for i, w in enumerate(words):
                lab = label_of(w, names)
                if lab is not None:
                    ids[lab].append(i)
            self._labels[key] = {n: np.array(v, dtype=int) for n, v in ids.items()}
        return self._labels[key]

    def labels(self, prompt, names=LABELS):
        """Exact P(the first output token spells each label), summed over the whole vocabulary: lo = hi."""
        T, names = self.T, tuple(names)
        with T.no_grad():
            p = T.softmax(self.lm(T.tensor([self._chat(prompt)], device=self.device)).logits[0, -1].double(), -1)
        p = p.cpu().numpy()
        return {n: (float(p[i].sum()),) * 2 for n, i in self._label_ids(names, len(p)).items()}


class Fn:
    """Any model that reads evidence, local or hosted: predict(evidence, spec) -> a law from laws.py, rebuilt from
    scratch on every call (a stateless API call is). evidence: a list of deletable items (observations, views, support
    examples); spec: the query, held fixed and passed in full (geometry, conditions, sites). A scored query is
    (spec, observed). window: it reads only the last `window` items (None: every item). empty(spec) -> law, if
    given, is the prior used when no item is left; declare it per dataset in advance, never from the items."""

    def __init__(self, predict, window=None, empty=None):
        self.predict, self.window, self.V, self.grid, self.empty = predict, window, None, None, empty

    def law(self, evidence, spec):
        evidence = [] if evidence is None else list(evidence)
        if self.window is not None:
            evidence = evidence[max(0, len(evidence) - self.window):]
        return self.empty(spec) if self.empty is not None and not evidence else self.predict(evidence, spec)

    def bits(self, context, queries):
        out, grids = [], set()
        for spec, y in queries:
            L = self.law(context, spec)
            grids.add((L.V, str(L.grid)))
            out.append(np.atleast_1d(L.bits(y)))
        if len(grids) > 1:
            raise ValueError('all queries must share one outcome grid')
        self.V, self.grid = grids.pop() if grids else (None, None)
        return out

    def dist(self, context, specs, cands=None):
        return np.concatenate([np.atleast_2d(self.law(context, s).probs()) for s in specs])


class Hosted:
    """An OpenAI-compatible endpoint. yes() and labels() need chat logprobs (OpenAI returns the top 20; top sets k).
    bits() needs prompt logprobs through /completions with echo (e.g. a vLLM server).
    n() is approximate (4 characters per token) unless you pass the provider's tokenizer as count.
    Calls at temperature 0 are cached per instance (cache=False turns this off). Timeouts, refused connections and
    HTTP 408/409/425/429/5xx are retried `retries` times, waiting backoff, 2 x backoff, ... seconds."""

    def __init__(self, model, base_url='https://api.openai.com/v1', key=None, window=None, V=None, count=None,
                 top=20, retries=2, backoff=1.0, cache=True):
        self.model, self.url = model, base_url.rstrip('/')
        self.key = key or os.environ.get('OPENAI_API_KEY', '')
        self.window, self.V, self._count = window, V, count or (lambda s: len(s) // 4 + 1)
        self.headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}
        self.query = ''
        self.top, self.retries, self.backoff = top, max(0, int(retries)), backoff
        self._cache = {} if cache else None

    def _send(self, path, body):
        req = urllib.request.Request(self.url + path + self.query, json.dumps(body).encode(), self.headers)
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code not in RETRY or attempt == self.retries:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == self.retries:
                    raise
            time.sleep(self.backoff * 2 ** attempt)

    def _post(self, path, body):
        key = path + json.dumps(body, sort_keys=True)
        keep = self._cache is not None and body.get('temperature') == 0
        if keep and key in self._cache:
            response = self._cache[key]
        else:
            response = self._send(path, body)
            if keep:
                if len(self._cache) >= 4096:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[key] = response
        self.returned_model = response.get('model')
        return response

    def _ask(self, prompt, max_tokens):
        return self._post('/chat/completions', dict(model=self.model, messages=[dict(role='user', content=prompt)],
                                                    max_tokens=max_tokens, temperature=0, logprobs=True,
                                                    top_logprobs=self.top))

    def labels(self, prompt, names=LABELS):
        """Ranges for P(the first output token spells each label), from its top-k output logprobs (ranges())."""
        top = self._ask(prompt, 1)['choices'][0]['logprobs']['content'][0]['top_logprobs']
        return ranges([(t['token'], t['logprob']) for t in top], names)

    def labels_many(self, prompt, n, names=LABELS):
        """Ranges for n labels answered one per line in a single reply (grouped claims). Each is read at the first
        token of its line, so it is conditioned on the answers above it. Raises unless the reply holds exactly n
        lines, each a label whose text matches the token it was read from; callers then ask one claim per call."""
        r = self._ask(prompt, 4 * n + 4)['choices'][0]
        out, read, start = [], [], True
        for c in r['logprobs']['content']:
            tok = c['token']
            if start and tok.strip():
                lab = label_of(tok, names)
                if lab is None:
                    raise ValueError(f'reply line does not start with a label: {tok!r}')
                out.append(ranges([(t['token'], t['logprob']) for t in c['top_logprobs']], names))
                read.append(lab)
                start = False
            if '\n' in tok:
                start = True
        lines = [line.strip() for line in str(r['message'].get('content') or '').splitlines() if line.strip()]
        if len(out) != n or [label_of(line, names) for line in lines] != read:
            raise ValueError(f'expected {n} label lines, read {read} from {lines}')
        return out

    def n(self, x):
        return self._count(x)

    def yes(self, prompt):
        r = self._post('/chat/completions', dict(model=self.model, messages=[dict(role='user', content=prompt)],
                                                 max_tokens=1, temperature=0, logprobs=True, top_logprobs=20))
        top = r['choices'][0]['logprobs']['content'][0]['top_logprobs']
        y = sum(math.exp(t['logprob']) for t in top if t['token'].strip().lower() == 'yes')
        n = sum(math.exp(t['logprob']) for t in top if t['token'].strip().lower() == 'no')
        return y / (y + n) if y + n > 0 else float('nan')

    def bits(self, context, queries):
        ctx = '\n\n'.join(context) if _is_units(context) else (context or '')
        out = []
        for q in queries:
            pre, ans = (ctx + q[0], q[1]) if isinstance(q, tuple) else (ctx, q)
            lp = self._post('/completions', dict(model=self.model, prompt=pre + ans, echo=True, logprobs=1,
                                                 max_tokens=1, temperature=0))['choices'][0]['logprobs']
            sel = [l for o, l in zip(lp['text_offset'], lp['token_logprobs']) if len(pre) <= o < len(pre + ans)]
            if not sel or any(l is None for l in sel):
                raise ValueError('endpoint returned no prompt logprobs for the answer')
            out.append(-np.array(sel) / math.log(2))
        return out


class Azure(Hosted):
    """Azure OpenAI deployment using an API key. Guard needs a deployment with chat logprobs.

    endpoint is the resource root or its /openai URL; model is the deployment name.
    Defaults read AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY. The tested chat
    deployment does not expose prompt logprobs, so bits() is unavailable there.
    """

    def __init__(self, model, endpoint=None, key=None, api_version='2025-04-01-preview', **kwargs):
        endpoint = endpoint or os.environ.get('AZURE_OPENAI_ENDPOINT', '')
        key = key or os.environ.get('AZURE_OPENAI_API_KEY', '')
        if not endpoint or not key or not model:
            raise ValueError('Azure needs an endpoint, deployment name and AZURE_OPENAI_API_KEY')
        root = endpoint.rstrip('/')
        if root.endswith('/openai/v1'):
            root = root[:-3]
        if not root.endswith('/openai'):
            root += '/openai'
        super().__init__(model, root + '/deployments/' + urllib.parse.quote(model, safe=''), key=key, **kwargs)
        self.headers = {'Content-Type': 'application/json', 'api-key': key}
        self.query = '?' + urllib.parse.urlencode({'api-version': api_version})
