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
    texlive-font-utils \
    texlive-bibtex-extra \
    biber \
    texlive-xetex \
    latexmk
RUN apt-get install ttf-mscorefonts-installer -y
RUN mkdir /wd
WORKDIR /wd

# To build the manuscript locally, one can use a Docker container based on this Dockerfile.
# Steps:
#             cd manuscript
#             docker build -t carcontainer .
#             cd ..
# On Windows: docker run --rm -it -v %cd%:/wd --workdir=/wd/manuscript carcontainer bash build.sh
# On Linux:   docker run --rm -it -v $(pwd):/wd --workdir=/wd/manuscript carcontainer bash build.sh
