"""Tables: in-context foundation models (TabPFN, TabICL) vs XGBoost trained on the same rows.

Every model gets the same 128 training rows. TabPFN and TabICL read them in context, with no training; XGBoost is
trained on them. Every other distinct row of the dataset is a held-out query, revealed in a random order, and each pair
of models is raced on the bits it spends on the true labels. Repeated feature vectors are dropped first, so no query
also sits in the training rows.

    pip install tabpfn==2.2.1 tabicl==2.2.0 xgboost scikit-learn
    python examples/tables.py                         # OpenML phoneme (1489) and spambase (44)
    python examples/tables.py --openml 11 --device cuda
"""
import argparse, itertools, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from bitsy import Fn, laws, bakeoff
from bitsy.adapters import Table


def models(seed, device):
    out = {}
    try:
        from huggingface_hub import hf_hub_download
        from tabpfn import TabPFNClassifier
        path = hf_hub_download('Prior-Labs/TabPFN-v2-clf', 'tabpfn-v2-classifier-finetuned-zk73skhh.ckpt',
                               revision='f851f2a3c941544733b712d8c0f96dfae9b28862')   # the open v2 checkpoint, pinned
        out['TabPFN-v2'] = TabPFNClassifier(model_path=path, device=device, n_estimators=2, random_state=seed)
    except ImportError as e:
        print(f'skipping TabPFN-v2: {e}')
    try:
        from tabicl import TabICLClassifier
        out['TabICL'] = TabICLClassifier(checkpoint_version='tabicl-classifier-v2-20260212.ckpt', device=device,
                                         random_state=seed)
    except ImportError as e:
        print(f'skipping TabICL: {e}')
    try:
        from xgboost import XGBClassifier
        out['XGBoost'] = XGBClassifier(n_estimators=100, max_depth=3, n_jobs=4, random_state=seed)
    except ImportError as e:
        print(f'skipping XGBoost: {e}')
    return out


def run(data_id, zoo, args):
    from sklearn.datasets import fetch_openml
    data = fetch_openml(data_id=data_id, as_frame=False)
    X = np.asarray(data.data, float)
    _, y = np.unique(data.target, return_inverse=True)
    _, first = np.unique(X, axis=0, return_index=True)
    keep = np.sort(first)
    X, y = X[keep], y[keep]
    order = np.random.default_rng(args.seed).permutation(len(y))
    train, held = order[:args.train], order[args.train:args.train + args.queries]
    support, classes = list(zip(X[train], y[train])), np.unique(y)
    print(f'\n{data.details["name"]} (OpenML {data_id}): {len(train)} training rows, {len(held)} held-out rows')
    scores = {}
    for name, clf in zoo.items():
        P = Table(clf, classes)(support, X[held]).P
        scores[name] = bakeoff.score(Fn(lambda e, i, P=P: laws.Categorical(P[i])), [], list(enumerate(y[held])), L=1)
        print(f'  {name:10s} accuracy {np.mean(P.argmax(1) == y[held]):.1%}   {scores[name]["cost"].mean():.3f} bits/row')
    for (na, a), (nb, b) in itertools.combinations(scores.items(), 2):
        who = {'A': na, 'B': nb, '': 'no call'}
        r, f = bakeoff.race(a, b, robust=True), bakeoff.race(a, b, robust=True, population=len(held))
        print(f'  {na} vs {nb}: {who[r["call"]]} after {r["queries"]} rows '
              f'(on these rows: {who[f["call"]]} after {f["queries"]})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--openml', type=int, nargs='+', default=[1489, 44], help='OpenML dataset IDs')
    parser.add_argument('--train', type=int, default=128)
    parser.add_argument('--queries', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    zoo = models(args.seed, args.device)
    for data_id in args.openml:
        run(data_id, zoo, args)
