#!/usr/bin/env bash
# Build every draft: python plot components first (the TikZ wrappers include
# their PDFs), then the standalone TikZ pictures, then PNG previews.
#   ./build.sh            build everything
#   ./build.sh tikz       only the TikZ pictures
#   ./build.sh py         only the python components
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
mkdir -p out
what=${1:-all}

if [[ $what == all || $what == py ]]; then
  for s in py/ga2_predictor_panel.py py/a8_algorithm_panels.py py/a9_pmc_panels.py py/a3_straggler_factor.py py/a5_fidelity_ratio.py py/a2_cumulative_completions.py; do
    echo "== $s"; $PY "$s" || echo "!! $s failed (continuing)"
  done
fi

if [[ $what == all || $what == tikz ]]; then
  for t in tikz/*.tex; do
    echo "== $t"
    ( cd tikz && latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory=../out "$(basename "$t")" > "../out/$(basename "${t%.tex}").buildlog" 2>&1 ) \
      || { echo "!! $t failed, see out/$(basename "${t%.tex}").buildlog"; grep -m3 -A3 '^!' "out/$(basename "${t%.tex}").log" || true; }
  done
  rm -f out/*.aux out/*.fls out/*.fdb_latexmk out/*.out
fi

for p in out/*.pdf; do
  pdftoppm -png -r 110 -singlefile "$p" "${p%.pdf}"
done
echo "previews:"; ls out/*.png
