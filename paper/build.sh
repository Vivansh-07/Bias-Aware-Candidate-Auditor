#!/usr/bin/env bash
# Local build with the portable MiKTeX on D: (Overleaf builds the same sources automatically).
B=/d/Tools/miktex/portable/texmfs/install/miktex/bin/x64
export TEMP='D:\Tools\miktex\tmp' TMP='D:\Tools\miktex\tmp' PATH="$B:$PATH"
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
bibtex main > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
grep -E "Output written|^!" main.log
grep -E "Overfull|undefined" main.log | head
