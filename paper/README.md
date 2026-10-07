# Paper: IEEE conference format (6 pages)

`main.tex` + `refs.bib` + `figures/*.pdf` (IEEEtran, `\documentclass[conference]{IEEEtran}`).

## Open in Overleaf

1. Overleaf → **New Project → Upload Project** → choose `paper_overleaf.zip`
   (built by the command below, or take the copy in `D:\PBL\Co_work_Claude\`).
2. Overleaf compiles with pdfLaTeX + BibTeX automatically. The result must be **6 pages**.

## Rebuild the figures and the zip

```bash
python paper/make_figures.py      # vector PDFs from results/*.csv
bash paper/build.sh               # local compile (portable MiKTeX on D:)
```

## Before submission

- Add author e-mail addresses / ORCID in the `\author{...}` block if the venue asks for them.
- Confirm the author list and order with all co-authors, including Dr. Chirag Joshi.
- References [8] and [9] are unpublished internal documents. Keep them, or replace them with the published versions once they exist.
- The text reports this repository's reimplementation results (`results/`), not the archived V1–V6 study numbers.
