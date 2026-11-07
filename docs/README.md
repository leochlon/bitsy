# bitsy docs

- [diagnose](diagnose.md): which parts of a prompt help, mislead or do nothing; verdicts, groups, long contexts, cost.
- [guard](guard.md): whether an answer is supported by its evidence, on hosted, local and other models.
- [Prompt or fine-tune](prompt-vs-finetune.md): `gate` before you train, `bakeoff` after, `margin` for "no worse
  than", `compile` (experimental) and `diagnose.unused`.
- [Where a prompt loses bits](where.md) (experimental): the prior, position, depth, heads and the block where a
  reading fix can live, for local models.
- [Certificates](certificates.md): on designed data, beat every memory of a given size or window, including ones
  you did not build.
- [Models](models.md): `Local`, `Hosted`, `Azure`, `Fn`, `laws` and the adapters, and which tools each runs.
- [How it works](how-it-works.md): the idea, the theory behind each tool, the source files and every limit.
