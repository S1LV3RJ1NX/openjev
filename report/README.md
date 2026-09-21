# Technical report

LaTeX source for the OpenJev technical report, arXiv-ready.

```bash
cd report && latexmk -pdf main.tex     # -> main.pdf
latexmk -C                             # clean build artifacts
```

Needs only standard TeX Live packages. References are inline in
`sections/references.tex` as a `thebibliography` block, so there is no
`.bib` step and no `bibtex` pass.

## Layout

```
main.tex            document, packages, section order
sections/           one file per section
figures/            plots, copied from ../docs/plots
```

## Keeping it in step with the repo

The report is the archival account; `docs/` is the user-facing
documentation. They overlap deliberately in two places and should not
drift:

| claim | source of truth |
|---|---|
| every experiment and its verdict | [`docs/ablations.md`](../docs/ablations.md) |
| every number with its command | [`docs/results.md`](../docs/results.md) |

When a result changes, update those two first, then the report. Figures
are copies rather than symlinks so a build is reproducible from a clone;
refresh them with:

```bash
cp ../docs/plots/*.png figures/
```
