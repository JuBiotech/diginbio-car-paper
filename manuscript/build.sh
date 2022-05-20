#!/bin/bash
echo "Generating draft PDF"
pdflatex sn-article
echo "Generating bibliography files"
bibtex sn-article
echo "Regenerating PDF, this time with bibliography"
pdflatex sn-article
echo "Regenerating PDF, this time with citations"
pdflatex sn-article
