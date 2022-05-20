FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y \
    texlive \
    texlive-lang-german \
    texlive-lang-english \
    texlive-latex-extra \
    texlive-science \
    texlive-fonts-extra \
    texlive-bibtex-extra \
    biber \
    texlive-xetex \
    latexmk \
    pandoc \
    pandoc-citeproc
RUN apt-get install ttf-mscorefonts-installer -y
RUN mkdir /wd
WORKDIR /wd


# See documentation here: https://boisgera.github.io/pandoc/markdown/
# cd manuscript
# docker build -t pandocstuff .
# docker run --rm -it -v %cd%:/wd pandocstuff pandoc /wd/main.md -o /wd/main.docx --number-sections --bibliography=literature.bib --csl=literature.csl
# docker run --rm -it -v %cd%:/wd pandocstuff pandoc /wd/main.md --pdf-engine=xelatex -o /wd/main.pdf --number-sections --bibliography=literature.bib --csl=literature.csl
# docker run --rm -it -v %cd%:/wd --workdir=/wd/manuscript pandocstuff bash build.sh
