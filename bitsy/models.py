"""Adapters. Language models (Local, Hosted) have: window, V, n(text), bits(context, queries), yes(prompt),
labels(prompt, names); Local also has copy(), step(text, lr), tune(texts, ...), dist(context, prompts, cands). Any other
model (Fn) has law(evidence, spec), bits(context, queries) and dist(context, specs).

Text arguments may be strings or token-id lists. A context is a string, a token-id list, or a list
of units (strings or token-id lists), which are read as separate passages. A query is text, or a pair
(cue, answer) whose answer alone is scored.

labels() reads only the first output token, so it needs output logprobs (OpenAI or Azure chat, vLLM, a local model)
and no echo. A label's probability is the probability that the first output token spells it (label_of)."""
import copy, json, math, os, time, urllib.error, urllib.request, urllib.parse
import numpy as np

YES, NO = ('Yes', ' Yes', 'yes', ' yes'), ('No', ' No', 'no', ' no')
LABELS = ('YES', 'NO', 'UNSURE')
RETRY = (408, 409, 425, 429, 500, 502, 503, 504)


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
    end-of-text token unless sep names another single token, such as a newline."""

    def __init__(self, path, device='cpu', sep=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.T, self.device, self._opt = torch, device, None
        self.tok = AutoTokenizer.from_pretrained(path)
        self.lm = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(device).eval()
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

    def _ctx(self, ctx, room):
        """[bos] + context in at most `room` tokens: units keep their last whole units, each closed by a
        separator; text keeps its last tokens."""
        if _is_units(ctx):
            ids = []
            for u in reversed(ctx):
                e = self.enc(u) + [self.sep]
                if 1 + len(ids) + len(e) > room:
                    break
                ids = e + ids
            if not ids:  # even the newest unit is too long: keep its last tokens
                e = self.enc(ctx[-1]) + [self.sep]
                ids = e[max(0, len(e) - room + 1):]
            return [self.bos] + ids
        ids = self.enc(ctx) if ctx is not None and len(ctx) else []
        return [self.bos] + ids[max(0, len(ids) - room + 1):]

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

    def tune(self, texts, epochs=1, lr=1e-4, chunk=256, batch=2, seed=0):
        """Full fine-tune with Adam. Each epoch cuts every text into pieces of at most `chunk` tokens at a random
        offset, covering it once, `batch` pieces per step, each read as [bos] + piece. texts: one text or a list.
        Returns a new model; this one is unchanged."""
        T, m, rng = self.T, self.copy(), np.random.default_rng(seed)
        opt = T.optim.Adam(m.lm.parameters(), lr=lr)
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
        return m

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
