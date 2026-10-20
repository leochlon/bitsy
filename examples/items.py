"""Which context items do the work, which are dead weight, and which mislead the model?

1. A retrieval prompt and one question. Each passage is deleted in turn and the context read again, and each is
   swapped for a control of the same form with irrelevant content. Value is the bits the passage saves on the true
   answer; content and form split it.
2. A few-shot prompt with no outside labels. Each demonstration is scored as a held-out query by the others.
3. A long context, screened in blocks first, so only the blocks that matter are tested one passage at a time.

    pip install -e '.[local]'
    python examples/items.py                          # GPT-2; any Hugging Face causal LM with --model
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from bitsy import Local, diagnose

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--model', default='gpt2')
args = parser.parse_args()
m = Local(args.model, sep='\n')                                     # one item per line


def show(title, labels, r):
    t = r['tokens']
    print(f'\n{title}\nthe whole context is worth {r["total"][0]:+.2f} bits per query; {r["reads"]} reads, '
          f're-reading {t[0]} of their {t[1]} context tokens')
    print(f'  {"item":34s}{"value":>8s}{"influence":>11s}{"content":>9s}{"form":>8s}   verdict')
    for row in r['items'] + r['groups']:
        name = labels[row['item']] if isinstance(row['item'], int) else f'items {row["item"][0]}-{row["item"][-1]} together'
        cell = lambda x: f'{x[0]:+.2f}' if x else '-'
        infl = '-' if row['influence'] is None else f'{row["influence"]:.2f}'
        print(f'  {name[:34]:34s}{cell(row["value"]):>8s}{infl:>11s}{cell(row["content"]):>9s}{cell(row["form"]):>8s}'
              f'   {row["verdict"]}')


# 1. what does this answer depend on?
passages = ['The code for Bob is 17.', 'The code for Alice is 52.', 'The code for Bob is not 71.',
            'Every code has two digits.', 'Fish swim in the sea.']
controls = ['The code for Kim is 45.', 'The code for Lee is 30.', 'The code for Ned is not 94.',
            'Every bird has two wings.', 'Birds fly in the sky.']
r = diagnose.items(m, passages, [('The code for Bob is', ' 17')], controls=controls)
show('1. "The code for Bob is" -> 17', passages, r)
for row in r['items']:
    print(f'  {passages[row["item"]]:34s}{row["why"]}')

# 2. which demonstrations does a few-shot prompt need?
words = ['cat', 'dog', 'tree', 'house', 'car', 'book', 'bird', 'lamp']
demos = [(f'{w} ->', f' {w.upper()}') for w in words]
r = diagnose.items(m, demos, groups=[list(range(6))])
show('2. uppercase demonstrations, each scored by the others', [f'{w} -> {w.upper()}' for w in words], r)

# 3. a long context: screen 4 blocks, then test passages one at a time only where it matters
filler = [f'Item {k} is stored on shelf {k % 7}.' for k in range(20)]
long = filler[:9] + ['The code for Bob is 17.'] + filler[9:]
r = diagnose.items(m, long, [('The code for Bob is', ' 17')], screen=4)
tested = [row for row in r['items'] if row['value'] is not None]
print(f'\n3. {len(long)} passages in 4 blocks: {r["reads"]} reads instead of {len(long) + 2}, re-reading '
      f'{r["tokens"][0]} of their {r["tokens"][1]} context tokens; {len(tested)} passages tested one at a time')
for row in tested:
    if row['verdict'] != 'dead weight':
        print(f'  {long[row["item"]]:34s}{row["value"][0]:+.2f} bits   {row["verdict"]}')
