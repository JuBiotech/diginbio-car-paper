FROM ubuntu:latest
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


# docker run --rm -it -v %cd%:/wd pandocstuff pandoc /wd/methods.md -o /wd/methods.docx
# docker run --rm -it -v %cd%:/wd pandocstuff pandoc /wd/methods.md --pdf-engine=xelatex -o /wd/methods.pdf
