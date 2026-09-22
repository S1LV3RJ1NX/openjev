# arXiv submission

`openjev-arxiv.tar.gz` is the submission bundle, verified to compile from
scratch in an empty directory: 37 pages, zero undefined references.

Rebuild it after any change to the report:

```bash
cd report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
tar czf arxiv/openjev-arxiv.tar.gz main.tex main.bbl refs.bib sections figures
```

## The one thing that breaks submissions

**`main.bbl` must be in the tarball.** arXiv does not reliably run BibTeX,
and without the `.bbl` this paper renders with 21 undefined citations and
question marks where every reference should be. We checked: removing it
produces exactly that. Regenerate it with `bibtex main` before packing,
because a stale `.bbl` silently ships the previous version's references.

Nothing else is unusual. No custom `.sty` or `.cls`, all stock packages,
figures are PNG.

## Metadata to paste into the submission form

- **Primary category**: cs.CL. Cross-list cs.LG.
- **License**: CC BY 4.0 matches the repository's Apache-2.0 code and lets
  people reuse the figures.
- **Comments field**: note the code, models and datasets, and say the
  repository is the live source of truth so the arXiv version reads as a
  snapshot rather than competing with it.

## Before you upload

First-time submitters to cs.CL usually need an **endorsement** from an
existing arXiv author. This is an administrative step, not a review, but
it blocks upload and is worth arranging in advance rather than discovering
at the point of submitting.

Moderation is typically one to two business days. A submission made after
14:00 US Eastern on a weekday appears the following announcement cycle.
