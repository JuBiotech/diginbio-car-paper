[![DOI](https://zenodo.org/badge/456422792.svg)](https://zenodo.org/badge/latestdoi/456422792)

This repository contains the data, code and manuscript of *Control of Parallelized Bioreactors II - Probabilistic Quantification of Carboxylic Acid Reductase Activity for Bioprocess Optimization* (2022), N. von den Eichen, M. Osthege, M. Dölle, L. Bromig, W. Wiechert, M. Oldiges, D. Weuster-Botz.

# Contents
- `/data` contains the raw experimental data
- `/manuscript` includes the manuscript TeX files and figures
- `/code` contains Jupyter notebooks and Python modules for the data processing pipeline

# Installation of the Environment
An `environment.yml` file is provided.
You can create an environment for running the data analysis with

```bash
conda env create -f environment.yml
```

# Running the analysis
Each step of the data analysis pipeline is a function in `/code/pipeline.py`.

We ran these steps in a parallelized setup on a cluster.
The important part is to run the tasks in the order found in `code/pipeline_run_all.py`.
