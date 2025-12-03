# Convenience script to build the paper.
# On Ubuntu/Mint run 'sudo apt install pandoc pandoc-citeproc'
# to install dependencies.

pandoc paper.md --pdf-engine=pdflatex --bibliography=paper.bib -o paper.pdf
