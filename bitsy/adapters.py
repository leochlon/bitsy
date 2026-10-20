"""Optional adapters for tables, one-step Chronos forecasts, field regressors, occupancy networks and multi-view
stereo."""
import hashlib, os, sys
import numpy as np
from . import laws


class Table:
    """Wrap a sklearn-style classifier. Evidence is (features, label); spec is features.

    Fix the class vocabulary before scoring. A changed evidence set always refits the
    classifier; repeated queries reuse that fit. Empty evidence gives a uniform prior.
    """

    def __init__(self, classifier, classes):
        self.classifier = classifier
        self.classes = np.asarray(classes)
        self._key = None

    def _law(self, p):
        result = laws.Categorical(p)
        result.grid = ('classes', tuple(self.classes.tolist()))
        return result

    def __call__(self, evidence, spec):
        query = np.atleast_2d(np.asarray(spec, float))
        p = np.ones((len(query), len(self.classes))) / len(self.classes)
        if not evidence:
            return self._law(p)
        X = np.asarray([x for x, _ in evidence], float)
        y = np.asarray([y for _, y in evidence])
        if not np.isin(y, self.classes).all():
            raise ValueError('evidence contains a label outside the fixed vocabulary')
        if len(np.unique(y)) == 1:
            p[:] = 1e-12
            p[:, self.classes == y[0]] = 1
            return self._law(p)
        key = hashlib.sha256(X.tobytes() + repr(y.tolist()).encode()).digest()
        if key != self._key:
            self.classifier.fit(X, y)
            self._key = key
        q = self.classifier.predict_proba(query)
        p[:] = 1e-12
        for j, label in enumerate(self.classifier.classes_):
            p[:, self.classes == label] = q[:, j, None]
        return self._law(p)


class Chronos:
    """Original Chronos-T5, exact next-step probabilities on a fixed value grid.

    Evidence is (time index, value); spec is the next time index. Deleted values
    remain missing at their original times. All numeric output tokens are decoded
    and rebinned into fixed physical edges, so comparisons share the same outcomes
    even when the context's scale changes. Pad/EOS mass is conditioned away.
    The observed label for Fn.bits is bin(value), including the two tail bins.
    """

    def __init__(self, pipeline, edges):
        self.pipeline = pipeline
        self.edges = np.asarray(edges, float)
        if self.edges.ndim != 1 or not np.all(np.diff(self.edges) > 0):
            raise ValueError('edges must be a strictly increasing vector')

    @classmethod
    def load(cls, path='amazon/chronos-t5-tiny', device='cpu', edges=None, **kwargs):
        from chronos import ChronosPipeline
        import torch
        pipe = ChronosPipeline.from_pretrained(path, device_map=device, torch_dtype=torch.float32, **kwargs)
        return cls(pipe, np.linspace(-8, 8, 65) if edges is None else edges)

    def bin(self, value):
        return np.searchsorted(self.edges, value, side='right')

    def inputs(self, evidence, spec):
        import torch
        if spec < 1 or any(i < 0 or i >= spec for i, _ in evidence):
            raise ValueError('history indices must precede the positive forecast index')
        if len({i for i, _ in evidence}) != len(evidence):
            raise ValueError('history indices must be unique')
        length = self.pipeline.model.config.context_length
        start = max(0, spec - length)
        values = torch.full((1, spec - start), torch.nan)
        for i, value in evidence:
            if i >= start:
                values[0, i - start] = float(value)
        return self.pipeline.tokenizer.context_input_transform(values)

    def __call__(self, evidence, spec):
        return self.batch([(evidence, spec)])[0]

    def batch(self, cases):
        """Independent one-step queries, left-padded for one model forward pass."""
        import torch
        prepared = [self.inputs(evidence, spec) for evidence, spec in cases]
        width = max(ids.shape[1] for ids, _, _ in prepared)
        ids = torch.cat([torch.nn.functional.pad(i, (width - i.shape[1], 0),
                                                value=self.pipeline.model.config.pad_token_id)
                         for i, _, _ in prepared])
        mask = torch.cat([torch.nn.functional.pad(m, (width - m.shape[1], 0), value=False)
                          for _, m, _ in prepared])
        scale = torch.cat([s for _, _, s in prepared])
        model = self.pipeline.model.model
        with torch.no_grad():
            output = model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device),
                           decoder_input_ids=torch.full((len(cases), 1), model.config.decoder_start_token_id,
                                                        device=model.device))
            first = self.pipeline.model.config.n_special_tokens
            p = torch.softmax(output.logits[:, 0, first:].float(), -1).cpu().numpy()
        tokens = torch.arange(first, first + p.shape[1]).reshape(1, 1, -1).expand(len(cases), -1, -1)
        values = self.pipeline.tokenizer.output_transform(tokens, scale).numpy()[:, 0]
        results = []
        for row, bins in zip(p, self.bin(values)):
            result = laws.Categorical(np.bincount(bins, weights=row, minlength=len(self.edges) + 1))
            result.grid = ('value-bins', tuple(self.edges.tolist()))
            results.append(result)
        return results


def field(regressor, calibration, grid):
    """A fitted regressor -> Gaussian field, calibrated on disjoint (design, field) cases.

    The regressor must predict one field per design. Evidence is ignored because
    it is a trained surrogate; compare against Fn(baselines.Kernel(...)).
    """
    point = lambda evidence, spec: regressor.predict(np.atleast_2d(spec))[0]
    return laws.spread(point, [([], x, y) for x, y in calibration], grid=grid)


def occupancy(net):
    """A point-cloud occupancy network -> yes/no per site. net(points [n, 3], sites [..., 3]) -> P(inside) per site,
    such as ConvONet below. Evidence: point patches [k, 3]; spec: the sites [s, 3], or (points always read, sites)."""
    def predict(evidence, spec):
        fixed, sites = spec if isinstance(spec, tuple) else (None, spec)
        parts = [np.asarray(p, float).reshape(-1, 3) for p in ([] if fixed is None else [fixed]) + list(evidence)]
        points = np.concatenate(parts) if parts else np.zeros((0, 3))
        return laws.Bernoulli(np.asarray(net(points, np.asarray(sites, float)), float).ravel())
    return predict


class ConvONet:
    """A pretrained Convolutional Occupancy Network (the official repository) as net(points, sites) -> P(inside), in the
    network's frame: the unit cube centred at the origin."""

    def __init__(self, model, device='cpu'):
        import torch
        self.model, self.torch, self.device = model.to(device).eval(), torch, device

    @classmethod
    def load(cls, repo, config, checkpoint, device='cpu', sha256=None):
        """repo: a clone of the repository, set up as its README says; config: relative to it and matching the
        checkpoint, e.g. configs/pointcloud/shapenet_3plane_partial.yaml. The released checkpoints hold NumPy scalars,
        so they load with weights_only=False, which runs the file's pickle: pass sha256 to check the file first."""
        import torch
        checkpoint, here = os.path.abspath(checkpoint), os.getcwd()
        if sha256 is not None:
            with open(checkpoint, 'rb') as f:
                if hashlib.sha256(f.read()).hexdigest() != sha256:
                    raise ValueError('the checkpoint does not match sha256')
        sys.path.insert(0, os.path.abspath(repo))
        from src import config as cfg
        try:
            os.chdir(repo)                    # configs name the configs they inherit from relative to the repository
            model = cfg.get_model(cfg.load_config(config, 'configs/default.yaml'), device=device, dataset=None)
        finally:
            os.chdir(here)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state['model'] if 'model' in state else state)
        return cls(model, device)

    def __call__(self, points, sites):
        t = self.torch
        with t.no_grad():
            c = self.model.encode_inputs(t.as_tensor(np.asarray(points, np.float32))[None].to(self.device))
            q = t.as_tensor(np.asarray(sites, np.float32).reshape(-1, 3))[None].to(self.device)
            return self.model.decode(q, c).probs[0].cpu().numpy().reshape(np.shape(sites)[:-1])


def depth(net, bins, pixels=None):
    """A multi-view stereo network -> a law per pixel over fixed depth bins, plus one outcome for any other depth.
    net(images, cameras, depth_min, interval, n) -> [n, h, w], the probability of each depth hypothesis
    depth_min + interval * i at each pixel, image 0 the reference: MVSNet's probability volume. Evidence: source views
    (image, camera); spec: (reference image, reference camera, depth_min, interval, n). bins: depth intervals
    [(lo, hi), ...], fixed, so specs with different hypotheses share outcomes; a hypothesis counts for the bin that holds
    it. pixels: the output pixels to read, an index into [h, w] (a mask, or a pair of slices); all by default. The
    observed outcome is a bin's index, or len(bins) for none of them."""
    bins = np.asarray(bins, float)
    where = (slice(None),) + (() if pixels is None else pixels if isinstance(pixels, tuple) else (pixels,))

    def predict(evidence, spec):
        image, camera, start, step, n = spec
        P = np.asarray(net([image] + [v[0] for v in evidence], [camera] + [v[1] for v in evidence], start, step, n),
                       float)[where].reshape(n, -1)
        z = start + step * np.arange(n)
        mass = ((z >= bins[:, :1]) & (z < bins[:, 1:])).astype(float) @ P
        return laws.Categorical(np.vstack([mass, np.maximum(1 - mass.sum(0), 0)]).T)
    return predict
