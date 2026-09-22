# Technical report

LaTeX source for the OpenJev technical report, arXiv-ready.

```bash
cd report && latexmk -pdf main.tex     # -> main.pdf
latexmk -C                             # clean build artifacts
```

Needs only standard TeX Live packages. Without `latexmk`:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main            # only when refs.bib changes
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

`main.bbl` is committed, so two `pdflatex` passes are enough for a clean
build as long as no citation was added or removed.

## Layout

```
main.tex            document, packages, section order
sections/           one file per section
refs.bib            bibliography, \bibliographystyle{plain}
figures/            plots, copied from ../docs/plots
```

Every entry in `refs.bib` was verified against the arXiv API. Do not add
one from memory; verify it the same way the file header describes.

## Which backbone the report treats as primary

The decoder (Qwen3-1.7B, rank-16 LoRA, merged) is the anchor architecture
throughout: abstract, introduction, architecture and results all lead with
it. The encoder (ModernBERT-base) is the documented alternative. Its
held-out results live in the ablations section, as does the argument for
why it is retained. If a future result flips that, the order to change is
abstract, introduction, `sections/method.tex`, `sections/results.tex`,
then the backbone recommendation in `sections/discussion.tex`.

## Style

No em dashes and no en dashes as punctuation, including inside LaTeX.
Check with:

```bash
grep -rn -- '---' sections/ main.tex
```

Literal `--` inside `\texttt{}` is a command-line flag, not punctuation,
and should stay.

## Keeping it in step with the repo

The report is the archival account and the source of truth for results.
`docs/` is user-facing documentation: how to run things, how to build a
dataset, what to reach for. Where the two overlap they should not drift.

| claim | source of truth |
|---|---|
| every experiment and its verdict | [`sections/ablations.tex`](sections/ablations.tex) |
| every number with its confidence interval | [`sections/results.tex`](sections/results.tex) |
| how to reproduce a number | [`docs/evaluation.md`](../docs/evaluation.md) |

When a result changes, update the report first, then any figure in
`docs/` that quotes it. Figures
are copies rather than symlinks so a build is reproducible from a clone;
refresh them with:

```bash
cp ../docs/plots/*.png figures/
```
