"""Claims against evidence: the Guard on VitaminC fact-verification rows, with a local or hosted model.

Each row is a claim and a sentence of evidence, labelled SUPPORTS, REFUTES or NOT ENOUGH INFO; a pass should mean
SUPPORTS. guard.check asks Yes or No. guard.strict asks YES, NO or UNSURE under a strict entailment prompt and reads
ranges from the top 20 output tokens, so a hosted chat model needs no echo. Rows come from the VitaminC test split
(CC BY-SA 3.0): 64 consecutive rows at offsets 0, 100 and 200.

    python examples/vitaminc.py --local Qwen/Qwen2.5-7B-Instruct --device cuda
    python examples/vitaminc.py --azure chat --endpoint https://<resource>.openai.azure.com   # AZURE_OPENAI_API_KEY
    python examples/vitaminc.py --openai gpt-4.1-mini                                        # OPENAI_API_KEY
    python examples/vitaminc.py --openai MODEL --base-url http://localhost:8000/v1            # a vLLM server
"""
import argparse, json, os, sys, urllib.request
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from bitsy import Local, Hosted, Azure, guard

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
which = parser.add_mutually_exclusive_group(required=True)
which.add_argument('--local', help='Hugging Face instruction model ID or folder')
which.add_argument('--azure', help='Azure OpenAI deployment name')
which.add_argument('--openai', help='model name on an OpenAI-compatible endpoint')
parser.add_argument('--endpoint', help='Azure resource URL (default: AZURE_OPENAI_ENDPOINT)')
parser.add_argument('--base-url', default='https://api.openai.com/v1')
parser.add_argument('--device', default='cpu')
parser.add_argument('--pages', type=int, default=3)
parser.add_argument('--rows', type=int, default=64)
parser.add_argument('--t', type=float, default=0.95, help='target probability for a pass')
args = parser.parse_args()


def page(i, n):
    url = ('https://datasets-server.huggingface.co/rows?dataset=tals%2Fvitaminc&config=default&split=test'
           f'&offset={100 * i}&length={min(n, 100)}')
    with urllib.request.urlopen(url, timeout=60) as r:
        return [item['row'] for item in json.load(r)['rows']]


model = (Local(args.local, device=args.device) if args.local else
         Azure(args.azure, endpoint=args.endpoint) if args.azure else Hosted(args.openai, base_url=args.base_url))
rows = [r for i in range(args.pages) for r in page(i, args.rows)]
supported = np.array([r['label'] == 'SUPPORTS' for r in rows])
print(f'{len(rows)} claims, {supported.sum()} supported by their evidence')
for name, check in [('guard.check', guard.check), ('guard.strict', guard.strict)]:
    out = [check(model, r['claim'], r['evidence'], t=args.t) for r in rows]
    passed = np.array([o['verdict'] == 'pass' for o in out])
    errors = sum(str(o.get('why', '')).startswith('error') for o in out)
    print(f'{name:13s} accuracy {np.mean(passed == supported):.1%}: passed {(passed & supported).sum()}/{supported.sum()} '
          f'supported claims and {(passed & ~supported).sum()}/{(~supported).sum()} unsupported'
          + (f'; {errors} failed calls flagged' if errors else ''))
